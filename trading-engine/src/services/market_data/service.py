"""Market data service with provider fallback logic."""

from src.config import settings
from src.services.market_data.alpaca_provider import AlpacaProvider
from src.services.market_data.provider import (
    MarketDataProvider,
    OHLCV,
    Quote,
    TechnicalIndicators,
)
from src.services.market_data.yfinance_provider import YFinanceProvider
from src.utils.logging import get_logger

log = get_logger(__name__)


class MarketDataService:
    """
    Market data service with automatic provider fallback.

    Uses Alpaca as primary provider, falls back to yfinance on failure.
    """

    def __init__(
        self,
        primary: MarketDataProvider | None = None,
        fallback: MarketDataProvider | None = None,
    ):
        self._primary = primary
        self._fallback = fallback

    @property
    def primary(self) -> MarketDataProvider:
        """Lazy initialization of primary provider."""
        if self._primary is None:
            if settings.alpaca.api_key:
                self._primary = AlpacaProvider()
            else:
                log.warning("alpaca_not_configured", message="Using yfinance as primary")
                self._primary = YFinanceProvider()
        return self._primary

    @property
    def fallback(self) -> MarketDataProvider:
        """Lazy initialization of fallback provider."""
        if self._fallback is None:
            self._fallback = YFinanceProvider()
        return self._fallback

    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote with fallback."""
        return await self._with_fallback(
            lambda p: p.get_quote(symbol),
            f"get_quote({symbol})",
        )

    async def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Get multiple quotes with fallback."""
        return await self._with_fallback(
            lambda p: p.get_quotes(symbols),
            f"get_quotes({symbols})",
        )

    async def get_bars(self, symbol: str, days: int = 200) -> list[OHLCV]:
        """Get historical bars with fallback."""
        return await self._with_fallback(
            lambda p: p.get_bars(symbol, days),
            f"get_bars({symbol}, {days})",
        )

    async def get_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Get technical indicators with fallback."""
        return await self._with_fallback(
            lambda p: p.get_technical_indicators(symbol),
            f"get_technical_indicators({symbol})",
        )

    async def get_indicators_batch(self, symbols: list[str]) -> dict[str, TechnicalIndicators]:
        """Get technical indicators for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = await self.get_technical_indicators(symbol)
            except Exception as e:
                log.error("indicator_fetch_failed", symbol=symbol, error=str(e))
        return results

    async def _with_fallback(self, operation, operation_name: str):
        """Execute operation with fallback on failure."""
        try:
            result = await operation(self.primary)
            log.debug("market_data_success", provider=self.primary.name, operation=operation_name)
            return result
        except Exception as e:
            log.warning(
                "market_data_fallback",
                primary=self.primary.name,
                fallback=self.fallback.name,
                operation=operation_name,
                error=str(e),
            )
            try:
                result = await operation(self.fallback)
                log.debug("market_data_fallback_success", provider=self.fallback.name, operation=operation_name)
                return result
            except Exception as fallback_error:
                log.error(
                    "market_data_failed",
                    operation=operation_name,
                    primary_error=str(e),
                    fallback_error=str(fallback_error),
                )
                raise


# Singleton instance
market_data_service = MarketDataService()
