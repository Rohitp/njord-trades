"""
Fundamentals provider for market data (sector, beta, market cap).

Provides fundamental data for symbol discovery and risk management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Fundamentals:
    """Fundamental data for a symbol."""

    symbol: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None  # In USD
    beta: float | None = None  # Volatility relative to market
    pe_ratio: float | None = None
    dividend_yield: float | None = None


class FundamentalsProvider(ABC):
    """
    Abstract base class for fundamentals providers.

    Implementations:
    - AlpacaFundamentalsProvider: Uses Alpaca fundamentals API
    - CachedFundamentalsProvider: Uses cached dataset (CSV/PostgreSQL)
    """

    @abstractmethod
    async def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Fundamentals object or None if not found
        """
        pass

    @abstractmethod
    async def get_fundamentals_batch(self, symbols: List[str]) -> Dict[str, Fundamentals]:
        """
        Get fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to Fundamentals (missing symbols omitted)
        """
        pass


class AlpacaFundamentalsProvider(FundamentalsProvider):
    """
    Fundamentals provider using Alpaca fundamentals API.

    Note: Alpaca's fundamentals API may require a paid subscription.
    Falls back gracefully if API is unavailable.
    """

    def __init__(self, client=None):
        """
        Initialize Alpaca fundamentals provider.

        Args:
            client: Alpaca TradingClient instance (creates default if None)
        """
        from src.config import settings
        from src.services.discovery.sources.alpaca import AlpacaAssetSource

        self.alpaca_source = AlpacaAssetSource()
        self.client = client or self.alpaca_source.client

    async def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        """
        Get fundamental data from Alpaca.

        **NOTE**: This is currently a placeholder. The Alpaca fundamentals API
        integration is not yet implemented. Market cap, beta, and sector filters
        in MetricPicker and FuzzyPicker will not be enforced until this is implemented
        or a cached dataset is provided.

        Args:
            symbol: Stock symbol

        Returns:
            Fundamentals object or None if not available
        """
        if not self.client:
            log.debug("alpaca_fundamentals_not_configured", symbol=symbol)
            return None

        try:
            # Alpaca fundamentals API endpoint
            # Note: This may require Alpaca Market Data Pro subscription
            # For now, return None as placeholder until API access is confirmed
            # This means market cap, beta, and sector filters are currently no-ops
            log.debug(
                "alpaca_fundamentals_not_implemented",
                symbol=symbol,
                message="Alpaca fundamentals API integration pending - filters are no-op",
            )
            return None

        except Exception as e:
            log.warning("alpaca_fundamentals_error", symbol=symbol, error=str(e))
            return None

    async def get_fundamentals_batch(
        self, symbols: List[str]
    ) -> Dict[str, Fundamentals]:
        """
        Get fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to Fundamentals
        """
        results = {}
        for symbol in symbols:
            fundamentals = await self.get_fundamentals(symbol)
            if fundamentals:
                results[symbol] = fundamentals

        return results


class CachedFundamentalsProvider(FundamentalsProvider):
    """
    Fundamentals provider using cached dataset.

    Can load from:
    - PostgreSQL table (fundamentals_cache)
    - CSV file
    - In-memory cache

    Falls back to AlpacaFundamentalsProvider if cache miss.
    """

    def __init__(
        self,
        fallback_provider: FundamentalsProvider | None = None,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize cached fundamentals provider.

        Args:
            fallback_provider: Provider to use on cache miss (default: AlpacaFundamentalsProvider)
            cache_ttl_hours: Cache TTL in hours (default: 24)
        """
        self.fallback = fallback_provider or AlpacaFundamentalsProvider()
        self.cache_ttl_hours = cache_ttl_hours
        # In-memory cache (could be replaced with Redis/PostgreSQL)
        self._cache: Dict[str, tuple[Fundamentals, float]] = {}

    async def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        """
        Get fundamental data from cache or fallback.

        Args:
            symbol: Stock symbol

        Returns:
            Fundamentals object or None
        """
        import time

        # Check cache
        if symbol in self._cache:
            fundamentals, cached_at = self._cache[symbol]
            age_hours = (time.time() - cached_at) / 3600
            if age_hours < self.cache_ttl_hours:
                return fundamentals

        # Cache miss or expired - use fallback
        fundamentals = await self.fallback.get_fundamentals(symbol)
        if fundamentals:
            self._cache[symbol] = (fundamentals, time.time())

        return fundamentals

    async def get_fundamentals_batch(
        self, symbols: List[str]
    ) -> Dict[str, Fundamentals]:
        """
        Get fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to Fundamentals
        """
        results = {}
        for symbol in symbols:
            fundamentals = await self.get_fundamentals(symbol)
            if fundamentals:
                results[symbol] = fundamentals

        return results

