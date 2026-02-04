"""
Trading cycle runner service.

Orchestrates the execution of trading cycles by:
1. Loading portfolio state from database
2. Invoking the LangGraph workflow
3. Returning results

Usage:
    runner = TradingCycleRunner()

    # Scheduled cycle for watchlist
    result = await runner.run_scheduled_cycle(["AAPL", "MSFT", "GOOGL"])

    # Event-triggered cycle for single symbol
    result = await runner.run_event_cycle("TSLA")
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.models import Position, PortfolioState, SystemState
from src.services.persistence import CyclePersistenceService
from src.utils.logging import get_logger
from src.workflows.graph import _db_session_context, trading_graph
from src.workflows.state import PortfolioSnapshot, TradingState

log = get_logger(__name__)


class TradingCycleRunner:
    """
    Service for running trading cycles.

    Handles the setup and teardown around the LangGraph workflow,
    including loading portfolio state and preparing the initial TradingState.
    """

    def __init__(self, db_session: AsyncSession | None = None):
        """
        Initialize the runner.

        Args:
            db_session: Optional database session. If not provided,
                        uses default portfolio snapshot.
        """
        self.db_session = db_session

    async def run_scheduled_cycle(
        self,
        symbols: list[str] | None = None,
        trace_id: str | None = None,
    ) -> TradingState:
        """
        Run a scheduled trading cycle for multiple symbols.

        This is the regular cycle that runs at configured times
        (e.g., 11:00 and 14:30 EST).

        Args:
            symbols: List of symbols to analyze. Defaults to watchlist from config.

        Returns:
            TradingState with all agent outputs populated

        Raises:
            ValueError: If circuit breaker is active or trading is disabled
        """
        # Check circuit breaker before starting
        if await self._is_circuit_breaker_active():
            raise ValueError("Trading halted by circuit breaker")

        # Use watchlist from config if no symbols provided
        if symbols is None:
            symbols = settings.trading.watchlist_symbols

        log.info(
            "cycle_started",
            cycle_type="scheduled",
            symbols=symbols,
            symbol_count=len(symbols),
        )

        start_time = datetime.now()

        try:
            # Load current portfolio state
            portfolio = await self._load_portfolio_snapshot()

            # Create initial state
            state = TradingState(
                cycle_type="scheduled",
                symbols=symbols,
                portfolio_snapshot=portfolio,
                started_at=start_time,
                trace_id=trace_id,
            )

            # Set db_session in context variable for use in workflow nodes
            if self.db_session:
                _db_session_context.set(self.db_session)

            # Run the workflow (returns dict due to LangGraph serialization)
            result_dict = await trading_graph.ainvoke(state)

            # Convert dict back to TradingState
            result = self._dict_to_state(result_dict)
            result.started_at = start_time  # Preserve original start time

            duration = (datetime.now() - start_time).total_seconds()

            # Persist cycle results to database for audit trail
            await self._persist_cycle_results(result, duration)

            log.info(
                "cycle_completed",
                cycle_id=str(result.cycle_id),
                cycle_type="scheduled",
                duration_seconds=duration,
                signals=len(result.signals),
                execute_decisions=len(result.get_execute_decisions()),
                errors=len(result.errors),
            )

            return result

        except Exception as e:
            log.error("cycle_failed", cycle_type="scheduled", error=str(e), exc_info=True)
            raise

    async def run_event_cycle(
        self,
        trigger_symbol: str,
        trace_id: str | None = None,
    ) -> TradingState:
        """
        Run an event-triggered trading cycle for a single symbol.

        Triggered when a significant price move (>5% in 15 min) is detected.

        Args:
            trigger_symbol: The symbol that triggered the event

        Returns:
            TradingState with all agent outputs populated

        Raises:
            ValueError: If circuit breaker is active or trading is disabled
        """
        # Check circuit breaker before starting
        if await self._is_circuit_breaker_active():
            raise ValueError("Trading halted by circuit breaker")

        # Check auto-resume conditions (but don't auto-reset - requires manual approval)
        if self.db_session:
            from src.services.circuit_breaker import CircuitBreakerService
            circuit_breaker = CircuitBreakerService(self.db_session)
            await circuit_breaker.check_auto_resume()

        log.info("cycle_started", cycle_type="event", trigger_symbol=trigger_symbol)

        start_time = datetime.now()

        try:
            # Load current portfolio state
            portfolio = await self._load_portfolio_snapshot()

            # Create initial state
            state = TradingState(
                cycle_type="event",
                trigger_symbol=trigger_symbol,
                symbols=[trigger_symbol],  # Focus on the triggering symbol
                portfolio_snapshot=portfolio,
                started_at=start_time,
                trace_id=trace_id,
            )

            # Set db_session in context variable for use in workflow nodes
            if self.db_session:
                _db_session_context.set(self.db_session)

            # Run the workflow (returns dict due to LangGraph serialization)
            result_dict = await trading_graph.ainvoke(state)

            # Convert dict back to TradingState
            result = self._dict_to_state(result_dict)
            result.started_at = start_time  # Preserve original start time

            duration = (datetime.now() - start_time).total_seconds()

            # Persist cycle results to database for audit trail
            await self._persist_cycle_results(result, duration)

            log.info(
                "cycle_completed",
                cycle_id=str(result.cycle_id),
                cycle_type="event",
                trigger_symbol=trigger_symbol,
                duration_seconds=duration,
                signals=len(result.signals),
                execute_decisions=len(result.get_execute_decisions()),
                errors=len(result.errors),
            )

            return result

        except Exception as e:
            log.error("cycle_failed", cycle_type="event", trigger_symbol=trigger_symbol, error=str(e), exc_info=True)
            raise

    async def _persist_cycle_results(
        self,
        state: TradingState,
        duration_seconds: float,
    ) -> None:
        """
        Persist cycle results to the events table for audit trail.

        Args:
            state: Complete trading state with all outputs
            duration_seconds: How long the cycle took
        """
        if self.db_session is None:
            log.debug("persistence_skipped", reason="no_db_session")
            return

        try:
            persistence = CyclePersistenceService(self.db_session)
            await persistence.record_full_cycle(state, duration_seconds)
        except Exception as e:
            # Don't fail the cycle if persistence fails - log and continue
            log.error(
                "persistence_failed",
                cycle_id=str(state.cycle_id),
                error=str(e),
            )

    def _dict_to_state(self, data: dict[str, Any]) -> TradingState:
        """
        Convert LangGraph output dict back to TradingState.

        Delegates to TradingState.from_dict() which handles all nested
        object reconstruction. Serialization logic is co-located with
        the schema definitions in state.py.

        Args:
            data: Dictionary from LangGraph output

        Returns:
            Reconstructed TradingState object

        Raises:
            ValueError: If data is invalid or missing required fields
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict from LangGraph, got {type(data).__name__}")

        try:
            # Use strict_timestamps=False by default to allow loading historical data
            # Set to True if you need to preserve original timestamps from archived events
            return TradingState.from_dict(data, strict_timestamps=False)
        except (ValueError, TypeError, KeyError) as e:
            log.error("state_reconstruction_failed", error=str(e), data_keys=list(data.keys()))
            raise ValueError(f"Failed to reconstruct TradingState: {e}") from e

    async def _load_portfolio_snapshot(self) -> PortfolioSnapshot:
        """
        Load current portfolio state from database.

        If no database session is available or portfolio doesn't exist,
        returns a default snapshot with initial capital.

        Returns:
            PortfolioSnapshot with current cash, positions, etc.
        """
        if self.db_session is None:
            # No database - return default with initial capital
            return PortfolioSnapshot(
                cash=settings.trading.initial_capital,
                total_value=settings.trading.initial_capital,
                deployed_capital=0.0,
                peak_value=settings.trading.initial_capital,
                positions={},
                sector_exposure={},
            )

        try:
            # Load portfolio state (single row table)
            result = await self.db_session.execute(
                select(PortfolioState).where(PortfolioState.id == 1)
            )
            portfolio_state = result.scalar_one_or_none()

            if portfolio_state is None:
                # No portfolio in DB yet - use initial capital
                return PortfolioSnapshot(
                    cash=settings.trading.initial_capital,
                    total_value=settings.trading.initial_capital,
                    deployed_capital=0.0,
                    peak_value=settings.trading.initial_capital,
                    positions={},
                    sector_exposure={},
                )

            # Load positions
            # Note: Position model doesn't have portfolio_state_id foreign key
            # Positions are identified by symbol (unique), so we load all positions
            positions_result = await self.db_session.execute(select(Position))
            positions = positions_result.scalars().all()

            # Build positions dict and sector exposure
            positions_dict: dict[str, dict[str, Any]] = {}
            sector_exposure: dict[str, float] = {}

            for pos in positions:
                positions_dict[pos.symbol] = {
                    "quantity": pos.quantity,
                    "value": pos.current_value,
                    "sector": pos.sector,
                    "avg_cost": pos.avg_cost,  # Fixed: model field is avg_cost, not average_cost
                }

                # Accumulate sector exposure
                if pos.sector:
                    sector_exposure[pos.sector] = (
                        sector_exposure.get(pos.sector, 0.0) + pos.current_value
                    )

            return PortfolioSnapshot(
                cash=portfolio_state.cash,
                total_value=portfolio_state.total_value,
                deployed_capital=portfolio_state.deployed_capital,
                peak_value=portfolio_state.peak_value,
                positions=positions_dict,
                sector_exposure=sector_exposure,
            )

        except Exception as e:
            # On any DB error, return safe default
            log.warning("portfolio_load_failed", error=str(e), using_default=True)
            return PortfolioSnapshot(
                cash=settings.trading.initial_capital,
                total_value=settings.trading.initial_capital,
                deployed_capital=0.0,
                peak_value=settings.trading.initial_capital,
                positions={},
                sector_exposure={},
            )

    async def _is_circuit_breaker_active(self) -> bool:
        """
        Check if circuit breaker is active or trading is disabled.

        Returns:
            True if trading should be halted, False otherwise
        """
        if self.db_session is None:
            # No DB session - assume trading is enabled
            return False

        try:
            result = await self.db_session.execute(
                select(SystemState).where(SystemState.id == 1)
            )
            system_state = result.scalar_one_or_none()

            if system_state is None:
                # No system state in DB - assume trading is enabled
                return False

            # Check if trading is disabled or circuit breaker is active
            if not system_state.trading_enabled:
                log.warning("trading_disabled", reason="trading_enabled=false")
                return True

            if system_state.circuit_breaker_active:
                log.warning(
                    "circuit_breaker_active",
                    reason=system_state.circuit_breaker_reason or "unknown",
                )
                return True

            return False

        except Exception as e:
            # On error, log but don't block trading (fail open)
            log.error("circuit_breaker_check_failed", error=str(e))
            return False
