"""Market data service - Alpaca only."""

from src.config import settings
from src.services.market_data.alpaca_provider import AlpacaProvider
from src.services.market_data.provider import (
    MarketDataProvider,
    OHLCV,
    Quote,
    TechnicalIndicators,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class MarketDataService:
    """
    Market data service using Alpaca as the sole provider.

    No fallback - if a symbol isn't in Alpaca, we can't trade it anyway.
    """

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
    ):
        self._provider = provider

    @property
    def provider(self) -> MarketDataProvider:
        """Lazy initialization of Alpaca provider."""
        if self._provider is None:
            if not settings.alpaca.api_key:
                raise ValueError("Alpaca API keys are required - no fallback provider available")
            self._provider = AlpacaProvider()
        return self._provider

    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote from Alpaca."""
        try:
            result = await self.provider.get_quote(symbol)
            log.debug(
                "market_data_success", provider=self.provider.name, operation=f"get_quote({symbol})"
            )
            return result
        except Exception as e:
            log.error(
                "market_data_failed",
                provider=self.provider.name,
                operation=f"get_quote({symbol})",
                error=str(e),
            )
            raise

    async def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Get multiple quotes from Alpaca."""
        try:
            result = await self.provider.get_quotes(symbols)
            log.debug(
                "market_data_success",
                provider=self.provider.name,
                operation=f"get_quotes({len(symbols)} symbols)",
            )
            return result
        except Exception as e:
            log.error(
                "market_data_failed",
                provider=self.provider.name,
                operation=f"get_quotes({len(symbols)} symbols)",
                error=str(e),
            )
            raise

    async def get_bars(self, symbol: str, days: int = 200) -> list[OHLCV]:
        """Get historical bars from Alpaca."""
        try:
            result = await self.provider.get_bars(symbol, days)
            log.debug(
                "market_data_success",
                provider=self.provider.name,
                operation=f"get_bars({symbol}, {days})",
            )
            return result
        except Exception as e:
            log.error(
                "market_data_failed",
                provider=self.provider.name,
                operation=f"get_bars({symbol}, {days})",
                error=str(e),
            )
            raise

    async def get_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Get technical indicators from Alpaca."""
        try:
            result = await self.provider.get_technical_indicators(symbol)
            log.debug(
                "market_data_success",
                provider=self.provider.name,
                operation=f"get_technical_indicators({symbol})",
            )
            return result
        except Exception as e:
            log.error(
                "market_data_failed",
                provider=self.provider.name,
                operation=f"get_technical_indicators({symbol})",
                error=str(e),
            )
            raise

    async def get_indicators_batch(self, symbols: list[str]) -> dict[str, TechnicalIndicators]:
        """Get technical indicators for multiple symbols (parallelized)."""
        import asyncio

        async def fetch_indicator(symbol: str) -> tuple[str, TechnicalIndicators | None]:
            try:
                indicators = await self.get_technical_indicators(symbol)
                return (symbol, indicators)
            except Exception as e:
                log.debug("indicator_fetch_failed", symbol=symbol, error=str(e))
                return (symbol, None)

        tasks = [fetch_indicator(symbol) for symbol in symbols]
        results_list = await asyncio.gather(*tasks)

        # Filter out None results
        results = {
            symbol: indicators for symbol, indicators in results_list if indicators is not None
        }
        return results


# Singleton instance
market_data_service = MarketDataService()
