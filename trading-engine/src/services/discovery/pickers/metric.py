"""
MetricPicker - Pure quantitative filters for symbol discovery.

Uses hard filters (volume, spread, market cap, beta) to identify tradable symbols.
No LLM, no scoring - just pass/fail filters.
"""

from typing import List

from src.services.discovery.pickers.base import PickerResult, SymbolPicker
from src.services.discovery.sources.alpaca import AlpacaAssetSource
from src.services.market_data.fundamentals import (
    AlpacaFundamentalsProvider,
    CachedFundamentalsProvider,
    FundamentalsProvider,
)
from src.services.market_data.provider import Quote
from src.services.market_data.service import MarketDataService
from src.utils.logging import get_logger

log = get_logger(__name__)


class MetricPicker(SymbolPicker):
    """
    Pure quantitative filter-based symbol picker.

    Filters symbols based on:
    - Minimum volume threshold
    - Maximum spread threshold
    - Market cap range (min/max)
    - Beta range (volatility)

    All filters must pass for a symbol to be included.
    """

    def __init__(
        self,
        min_volume: int = 1000000,  # 1M shares/day
        max_spread_pct: float = 0.01,  # 1% max spread
        min_market_cap: float = 100_000_000,  # $100M minimum
        max_market_cap: float = 10_000_000_000_000,  # $10T maximum (effectively no max)
        min_beta: float = 0.5,
        max_beta: float = 2.0,
        fundamentals_provider: FundamentalsProvider | None = None,
    ):
        """
        Initialize MetricPicker with filter thresholds.

        Args:
            min_volume: Minimum daily volume (shares)
            max_spread_pct: Maximum bid-ask spread percentage
            min_market_cap: Minimum market capitalization (USD)
            max_market_cap: Maximum market capitalization (USD)
            min_beta: Minimum beta (volatility relative to market)
            max_beta: Maximum beta
            fundamentals_provider: Fundamentals provider (default: CachedFundamentalsProvider with Alpaca fallback)
        """
        self.min_volume = min_volume
        self.max_spread_pct = max_spread_pct
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_beta = min_beta
        self.max_beta = max_beta

        self.asset_source = AlpacaAssetSource()
        self.market_data = MarketDataService()
        # Use cached provider with Alpaca fallback
        self.fundamentals = fundamentals_provider or CachedFundamentalsProvider(
            fallback_provider=AlpacaFundamentalsProvider()
        )

    @property
    def name(self) -> str:
        """Picker name."""
        return "metric"

    async def pick(self, context: dict | None = None) -> List[PickerResult]:
        """
        Pick symbols using quantitative filters.

        Args:
            context: Optional context containing:
                - candidate_symbols: Optional list of symbols to filter (if None, fetches all)
                - max_symbols: Optional limit on number of symbols to process (default: 500)

        Returns:
            List of PickerResult objects (all with score=1.0 if they pass filters)
        """
        import asyncio

        try:
            # Get candidate symbols from context or fetch all
            if context and "candidate_symbols" in context:
                symbols = context["candidate_symbols"]
                log.info("metric_picker_using_candidates", symbol_count=len(symbols))
            else:
                # Get all tradable stocks
                symbols = await self.asset_source.get_stocks()
                log.info("metric_picker_starting", symbol_count=len(symbols))
        except ValueError as e:
            log.warning("metric_picker_no_alpaca", error=str(e))
            return []  # Can't fetch symbols without Alpaca

        # Limit symbols to process (avoid processing thousands)
        max_symbols = context.get("max_symbols", 500) if context else 500
        if len(symbols) > max_symbols:
            log.info("metric_picker_limiting_symbols", total=len(symbols), limit=max_symbols)
            symbols = symbols[:max_symbols]

        # Process symbols in batches using batch API calls to minimize requests
        # Batch size limited to avoid Alpaca API limits (typically 100-200 symbols per request)
        batch_size = 100
        results = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            log.debug(
                "metric_picker_processing_batch",
                batch_num=i // batch_size + 1,
                batch_size=len(batch),
            )

            # Fetch quotes for entire batch in ONE API call
            try:
                quotes = await self.market_data.get_quotes(batch)
                quotes_map = {q.symbol: q for q in quotes}
            except Exception as e:
                log.error("metric_picker_batch_quote_failed", batch_size=len(batch), error=str(e))
                quotes_map = {}

            # Process each symbol with the pre-fetched quotes
            for symbol in batch:
                try:
                    result = await self._check_symbol_with_quote(symbol, quotes_map.get(symbol))
                    if result:
                        results.append(result)
                except Exception as e:
                    log.debug("metric_picker_symbol_error", symbol=symbol, error=str(e))

            # Log progress every 5 batches
            if (i // batch_size + 1) % 5 == 0:
                log.info(
                    "metric_picker_progress",
                    processed=i + len(batch),
                    total=len(symbols),
                    passed=len(results),
                )

        log.info("metric_picker_complete", passed=len(results), total=len(symbols))
        return sorted(results, key=lambda x: x.symbol)  # Sort alphabetically

    async def _check_symbol_with_quote(
        self, symbol: str, quote: "Quote | None"
    ) -> PickerResult | None:
        """
        Check if a single symbol passes filters using a pre-fetched quote.

        Args:
            symbol: Symbol to check
            quote: Pre-fetched quote data (or None if quote failed)

        Returns:
            PickerResult if symbol passes, None otherwise
        """
        try:
            if await self._passes_filters_with_quote(symbol, quote):
                return PickerResult(
                    symbol=symbol,
                    score=1.0,  # Binary: pass or fail
                    reason=self._build_reason(symbol),
                    metadata={"picker": "metric"},
                )
            return None
        except Exception as e:
            log.debug("metric_picker_symbol_error", symbol=symbol, error=str(e))
            return None

    async def _check_symbol(self, symbol: str) -> PickerResult | None:
        """
        Check if a single symbol passes filters (fetches its own quote).

        DEPRECATED: Use _check_symbol_with_quote with batch quotes instead.

        Returns:
            PickerResult if symbol passes, None otherwise
        """
        try:
            quote = await self.market_data.get_quote(symbol)
            return await self._check_symbol_with_quote(symbol, quote)
        except Exception as e:
            log.debug("metric_picker_symbol_error", symbol=symbol, error=str(e))
            return None

    async def _passes_filters_with_quote(self, symbol: str, quote: Quote | None) -> bool:
        """
        Check if symbol passes all quantitative filters using a pre-fetched quote.

        Args:
            symbol: Symbol to check
            quote: Pre-fetched quote data (or None if unavailable)

        Returns:
            True if all filters pass
        """
        try:
            # If no quote, symbol fails
            if quote is None:
                log.debug("metric_picker_no_quote", symbol=symbol)
                return False

            # Volume filter
            if quote.volume is None or quote.volume < self.min_volume:
                log.debug(
                    "metric_picker_volume_failed",
                    symbol=symbol,
                    volume=quote.volume,
                    threshold=self.min_volume,
                )
                return False

            # Spread filter (if bid/ask available and price is valid)
            if quote.bid and quote.ask and quote.price and quote.price > 0:
                spread_pct = abs(quote.ask - quote.bid) / quote.price
                if spread_pct > self.max_spread_pct:
                    log.debug(
                        "metric_picker_spread_failed",
                        symbol=symbol,
                        spread_pct=spread_pct,
                        threshold=self.max_spread_pct,
                    )
                    return False

            # Market cap and beta filters (if fundamentals available)
            fundamentals = await self.fundamentals.get_fundamentals(symbol)
            if fundamentals:
                # Market cap filter
                if fundamentals.market_cap is not None:
                    if fundamentals.market_cap < self.min_market_cap:
                        log.debug(
                            "metric_picker_market_cap_failed",
                            symbol=symbol,
                            market_cap=fundamentals.market_cap,
                            threshold=self.min_market_cap,
                        )
                        return False
                    if fundamentals.market_cap > self.max_market_cap:
                        log.debug(
                            "metric_picker_market_cap_failed",
                            symbol=symbol,
                            market_cap=fundamentals.market_cap,
                            threshold=self.max_market_cap,
                        )
                        return False

                # Beta filter
                if fundamentals.beta is not None:
                    if fundamentals.beta < self.min_beta or fundamentals.beta > self.max_beta:
                        log.debug(
                            "metric_picker_beta_failed",
                            symbol=symbol,
                            beta=fundamentals.beta,
                            min_threshold=self.min_beta,
                            max_threshold=self.max_beta,
                        )
                        return False

            return True

        except Exception as e:
            log.debug("metric_picker_filter_error", symbol=symbol, error=str(e))
            return False

    async def _passes_filters(self, symbol: str) -> bool:
        """
        Check if symbol passes all quantitative filters (fetches its own quote).

        DEPRECATED: Use _passes_filters_with_quote with batch quotes instead.

        Args:
            symbol: Symbol to check

        Returns:
            True if all filters pass
        """
        try:
            quote = await self.market_data.get_quote(symbol)
            return await self._passes_filters_with_quote(symbol, quote)
        except Exception as e:
            log.debug("metric_picker_filter_error", symbol=symbol, error=str(e))
            return False

            # Spread filter (if bid/ask available and price is valid)
            if quote.bid and quote.ask and quote.price and quote.price > 0:
                spread_pct = abs(quote.ask - quote.bid) / quote.price
                if spread_pct > self.max_spread_pct:
                    log.debug(
                        "metric_picker_spread_failed",
                        symbol=symbol,
                        spread_pct=spread_pct,
                        threshold=self.max_spread_pct,
                    )
                    return False

            # Market cap and beta filters (if fundamentals available)
            fundamentals = await self.fundamentals.get_fundamentals(symbol)
            if fundamentals:
                # Market cap filter
                if fundamentals.market_cap is not None:
                    if fundamentals.market_cap < self.min_market_cap:
                        log.debug(
                            "metric_picker_market_cap_failed",
                            symbol=symbol,
                            market_cap=fundamentals.market_cap,
                            threshold=self.min_market_cap,
                        )
                        return False
                    if fundamentals.market_cap > self.max_market_cap:
                        log.debug(
                            "metric_picker_market_cap_failed",
                            symbol=symbol,
                            market_cap=fundamentals.market_cap,
                            threshold=self.max_market_cap,
                        )
                        return False

                # Beta filter
                if fundamentals.beta is not None:
                    if fundamentals.beta < self.min_beta or fundamentals.beta > self.max_beta:
                        log.debug(
                            "metric_picker_beta_failed",
                            symbol=symbol,
                            beta=fundamentals.beta,
                            min_threshold=self.min_beta,
                            max_threshold=self.max_beta,
                        )
                        return False

            return True

        except Exception as e:
            log.debug("metric_picker_filter_error", symbol=symbol, error=str(e))
            return False

    def _build_reason(self, symbol: str) -> str:
        """Build reason string for why symbol passed filters."""
        return f"Passed quantitative filters: volume >= {self.min_volume:,}, spread <= {self.max_spread_pct * 100:.2f}%"
