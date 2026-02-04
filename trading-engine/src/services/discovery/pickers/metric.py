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

        Returns:
            List of PickerResult objects (all with score=1.0 if they pass filters)
        """
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

        results = []
        for symbol in symbols:
            try:
                if await self._passes_filters(symbol):
                    results.append(
                        PickerResult(
                            symbol=symbol,
                            score=1.0,  # Binary: pass or fail
                            reason=self._build_reason(symbol),
                            metadata={"picker": "metric"},
                        )
                    )
            except Exception as e:
                log.debug("metric_picker_symbol_error", symbol=symbol, error=str(e))
                continue  # Skip symbols that error

        log.info("metric_picker_complete", passed=len(results), total=len(symbols))
        return sorted(results, key=lambda x: x.symbol)  # Sort alphabetically

    async def _passes_filters(self, symbol: str) -> bool:
        """
        Check if symbol passes all quantitative filters.

        Args:
            symbol: Symbol to check

        Returns:
            True if all filters pass
        """
        try:
            # Get quote for volume and spread
            quote = await self.market_data.get_quote(symbol)
            
            # Volume filter
            if quote.volume is None or quote.volume < self.min_volume:
                log.debug("metric_picker_volume_failed", symbol=symbol, volume=quote.volume, threshold=self.min_volume)
                return False

            # Spread filter (if bid/ask available and price is valid)
            if quote.bid and quote.ask and quote.price and quote.price > 0:
                spread_pct = abs(quote.ask - quote.bid) / quote.price
                if spread_pct > self.max_spread_pct:
                    log.debug("metric_picker_spread_failed", symbol=symbol, spread_pct=spread_pct, threshold=self.max_spread_pct)
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
        return f"Passed quantitative filters: volume >= {self.min_volume:,}, spread <= {self.max_spread_pct*100:.2f}%"

