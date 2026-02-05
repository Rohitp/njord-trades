"""
Symbol Discovery Service - Orchestration layer for symbol discovery.

Runs all pickers, applies ensemble combination, and persists results to database.
"""

from datetime import datetime
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.models import DiscoveredSymbol, PickerSuggestion, Watchlist
from src.services.discovery.ensemble import EnsembleCombiner
from src.services.discovery.pickers.base import PickerResult
from src.services.discovery.pickers.fuzzy import FuzzyPicker
from src.services.discovery.pickers.llm import LLMPicker
from src.services.discovery.pickers.metric import MetricPicker
from src.services.market_data.service import MarketDataService
from src.utils.logging import get_logger

log = get_logger(__name__)


class SymbolDiscoveryService:
    """
    Orchestration service for symbol discovery.

    Coordinates:
    1. Running all enabled pickers (Metric, Fuzzy, LLM)
    2. Applying EnsembleCombiner to merge results
    3. Persisting DiscoveredSymbol and PickerSuggestion records
    4. Updating Watchlist with top-ranked symbols
    """

    def __init__(self, db_session: AsyncSession | None = None):
        """
        Initialize SymbolDiscoveryService.

        Args:
            db_session: Database session for persistence (required for full functionality)
        """
        self.db_session = db_session

        # Initialize pickers based on config
        self.pickers = {}
        if "metric" in settings.discovery.enabled_pickers:
            self.pickers["metric"] = MetricPicker()
        if "fuzzy" in settings.discovery.enabled_pickers:
            self.pickers["fuzzy"] = FuzzyPicker(db_session=db_session)
        if "llm" in settings.discovery.enabled_pickers:
            self.pickers["llm"] = LLMPicker(db_session=db_session)

        self.ensemble = EnsembleCombiner()
        self.market_data = MarketDataService()  # For capturing prices at suggestion time

    async def run_discovery_cycle(
        self,
        context: dict | None = None,
        update_watchlist: bool = True,
    ) -> dict:
        """
        Run a complete discovery cycle.

        Args:
            context: Optional context for pickers (portfolio state, market conditions)
            update_watchlist: If True, update Watchlist with top symbols

        Returns:
            Dictionary with:
                - discovered_symbols: List of DiscoveredSymbol records
                - picker_suggestions: List of PickerSuggestion records
                - ensemble_results: Final ranked list from EnsembleCombiner
                - watchlist_updates: Number of symbols added to watchlist
        """
        log.info("discovery_cycle_starting")

        if not self.pickers:
            log.warning("discovery_no_pickers_enabled")
            return {
                "discovered_symbols": [],
                "picker_suggestions": [],
                "ensemble_results": [],
                "watchlist_updates": 0,
            }

        # Run all pickers in parallel using asyncio.gather
        import asyncio

        async def run_picker(picker_name: str, picker):
            """Run a single picker with error handling."""
            try:
                log.info("discovery_picker_starting", picker=picker_name)
                results = await picker.pick(context=context)
                log.info(
                    "discovery_picker_complete",
                    picker=picker_name,
                    results_count=len(results),
                )
                return picker_name, results
            except Exception as e:
                log.error(
                    "discovery_picker_error",
                    picker=picker_name,
                    error=str(e),
                    exc_info=True,
                )
                return picker_name, []  # Return empty results on error

        # Run all pickers concurrently
        tasks = [
            run_picker(picker_name, picker)
            for picker_name, picker in self.pickers.items()
        ]
        results_list = await asyncio.gather(*tasks)

        # Convert results to dictionary
        picker_results = {name: results for name, results in results_list}

        # Apply ensemble combination
        ensemble_results = self.ensemble.combine_from_dict(picker_results)

        # Persist results if database session available
        discovered_symbols = []
        picker_suggestions = []
        watchlist_updates = 0

        if self.db_session:
            try:
                # Persist DiscoveredSymbol records
                discovered_symbols = await self._persist_discovered_symbols(
                    picker_results, self.db_session
                )

                # Persist PickerSuggestion records
                picker_suggestions = await self._persist_picker_suggestions(
                    picker_results, self.db_session
                )

                # Update watchlist if requested
                if update_watchlist:
                    watchlist_updates = await self._update_watchlist(
                        ensemble_results, self.db_session
                    )

                # Commit all changes
                await self.db_session.commit()

                log.info(
                    "discovery_cycle_complete",
                    discovered_count=len(discovered_symbols),
                    suggestions_count=len(picker_suggestions),
                    ensemble_count=len(ensemble_results),
                    watchlist_updates=watchlist_updates,
                )
            except Exception as e:
                log.error("discovery_persistence_error", error=str(e), exc_info=True)
                await self.db_session.rollback()
                raise

        return {
            "discovered_symbols": discovered_symbols,
            "picker_suggestions": picker_suggestions,
            "ensemble_results": ensemble_results,
            "watchlist_updates": watchlist_updates,
        }

    async def _persist_discovered_symbols(
        self,
        picker_results: dict[str, List[PickerResult]],
        session: AsyncSession,
    ) -> List[DiscoveredSymbol]:
        """
        Persist DiscoveredSymbol records for all picker results.

        Deduplicates by (symbol, picker_name, discovered_at date) to avoid
        inserting duplicate records on repeated cycles.

        Args:
            picker_results: Dict mapping picker names to their results
            session: Database session

        Returns:
            List of created DiscoveredSymbol records
        """
        from sqlalchemy import select, func, cast, Date

        discovered_at = datetime.now()
        discovered_date = discovered_at.date()
        discovered_symbols = []

        for picker_name, results in picker_results.items():
            for result in results:
                symbol = result.symbol.upper()

                # Check if this (symbol, picker_name, date) already exists
                existing = await session.execute(
                    select(DiscoveredSymbol)
                    .where(DiscoveredSymbol.symbol == symbol)
                    .where(DiscoveredSymbol.picker_name == picker_name)
                    .where(
                        cast(DiscoveredSymbol.discovered_at, Date) == discovered_date
                    )
                )
                if existing.scalar_one_or_none():
                    log.debug(
                        "discovered_symbol_duplicate_skipped",
                        symbol=symbol,
                        picker=picker_name,
                        date=discovered_date,
                    )
                    continue

                discovered_symbol = DiscoveredSymbol(
                    symbol=symbol,
                    picker_name=picker_name,
                    score=float(result.score),
                    reason=result.reason,
                    picker_metadata=result.metadata,
                    discovered_at=discovered_at,
                )
                session.add(discovered_symbol)
                discovered_symbols.append(discovered_symbol)

        return discovered_symbols

    async def _persist_picker_suggestions(
        self,
        picker_results: dict[str, List[PickerResult]],
        session: AsyncSession,
    ) -> List[PickerSuggestion]:
        """
        Persist PickerSuggestion records for forward return tracking.

        Deduplicates by (symbol, picker_name, suggested_at date) to avoid
        inserting duplicate records on repeated cycles.

        Args:
            picker_results: Dict mapping picker names to their results
            session: Database session

        Returns:
            List of created PickerSuggestion records
        """
        from sqlalchemy import select, cast, Date

        suggested_at = datetime.now()
        suggested_date = suggested_at.date()
        suggestions = []

        # Collect all unique symbols to fetch prices in batch
        all_symbols = set()
        for results in picker_results.values():
            for result in results:
                all_symbols.add(result.symbol.upper())

        # Fetch current prices for all symbols (for realistic paper trading)
        symbol_prices = {}
        if all_symbols:
            try:
                quotes = await self.market_data.get_quotes(list(all_symbols))
                for quote in quotes:
                    if quote and quote.price:
                        symbol_prices[quote.symbol.upper()] = quote.price
            except Exception as e:
                log.warning(
                    "discovery_price_fetch_failed",
                    error=str(e),
                    symbols=list(all_symbols),
                )
                # Continue without prices - suggestions will have suggested_price=None

        for picker_name, results in picker_results.items():
            for result in results:
                symbol = result.symbol.upper()

                # Check if this (symbol, picker_name, date) already exists
                existing = await session.execute(
                    select(PickerSuggestion)
                    .where(PickerSuggestion.symbol == symbol)
                    .where(PickerSuggestion.picker_name == picker_name)
                    .where(cast(PickerSuggestion.suggested_at, Date) == suggested_date)
                )
                if existing.scalar_one_or_none():
                    log.debug(
                        "picker_suggestion_duplicate_skipped",
                        symbol=symbol,
                        picker=picker_name,
                        date=suggested_date,
                    )
                    continue

                # Get price at suggestion time (if available)
                suggested_price = symbol_prices.get(symbol)

                suggestion = PickerSuggestion(
                    symbol=symbol,
                    picker_name=picker_name,
                    score=float(result.score),
                    reason=result.reason,
                    suggested_at=suggested_at,
                    suggested_price=suggested_price,
                )
                session.add(suggestion)
                suggestions.append(suggestion)

        return suggestions

    async def _update_watchlist(
        self,
        ensemble_results: List[PickerResult],
        session: AsyncSession,
    ) -> int:
        """
        Update Watchlist with top-ranked symbols from ensemble.

        Args:
            ensemble_results: Final ranked list from EnsembleCombiner
            session: Database session

        Returns:
            Number of symbols added to watchlist
        """
        # Get top N symbols (up to max_watchlist_size)
        top_symbols = ensemble_results[: settings.discovery.max_watchlist_size]

        if not top_symbols:
            return 0

        # Get current watchlist
        current_watchlist = await session.execute(
            select(Watchlist).where(Watchlist.active == True)
        )
        current_symbols = {w.symbol for w in current_watchlist.scalars().all()}

        # Add new symbols that aren't already in watchlist
        added_count = 0
        for result in top_symbols:
            symbol = result.symbol.upper()
            if symbol not in current_symbols:
                # Check if symbol was previously in watchlist (inactive)
                existing = await session.execute(
                    select(Watchlist).where(Watchlist.symbol == symbol)
                )
                existing_watchlist = existing.scalar_one_or_none()

                if existing_watchlist:
                    # Reactivate existing watchlist entry
                    existing_watchlist.active = True
                    existing_watchlist.removed_at = None
                    existing_watchlist.source = "ensemble"
                    existing_watchlist.updated_at = datetime.now()
                else:
                    # Create new watchlist entry
                    watchlist_entry = Watchlist(
                        symbol=symbol,
                        source="ensemble",
                        active=True,
                        added_at=datetime.now(),
                    )
                    session.add(watchlist_entry)

                added_count += 1
                current_symbols.add(symbol)  # Prevent duplicates in this batch

        log.info("watchlist_updated", added_count=added_count, total_symbols=len(top_symbols))
        return added_count

    async def get_active_watchlist(self, session: AsyncSession) -> List[str]:
        """
        Get list of active watchlist symbols.

        Args:
            session: Database session

        Returns:
            List of symbol strings
        """
        result = await session.execute(
            select(Watchlist).where(Watchlist.active == True).order_by(Watchlist.added_at)
        )
        watchlist = result.scalars().all()
        return [w.symbol for w in watchlist]

    async def remove_from_watchlist(
        self,
        symbol: str,
        session: AsyncSession,
    ) -> bool:
        """
        Remove a symbol from the watchlist (mark as inactive).

        Args:
            symbol: Symbol to remove
            session: Database session

        Returns:
            True if symbol was removed, False if not found
        """
        result = await session.execute(
            select(Watchlist).where(
                Watchlist.symbol == symbol.upper(),
                Watchlist.active == True,
            )
        )
        watchlist_entry = result.scalar_one_or_none()

        if watchlist_entry:
            watchlist_entry.active = False
            watchlist_entry.removed_at = datetime.now()
            watchlist_entry.updated_at = datetime.now()
            await session.commit()
            return True

        return False

