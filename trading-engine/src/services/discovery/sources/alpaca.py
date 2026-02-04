"""
Alpaca assets API data source for symbol discovery.

Fetches available trading symbols from Alpaca API for pickers to evaluate.
"""

import asyncio
from typing import List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from src.config import settings
from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

log = get_logger(__name__)


class AlpacaAssetSource:
    """
    Data source for fetching available symbols from Alpaca.

    Provides symbols that can be traded, filtered by asset class and status.
    """

    def __init__(self):
        """Initialize Alpaca asset source."""
        self.client: TradingClient | None = None
        if settings.alpaca.api_key and settings.alpaca.secret_key:
            self.client = TradingClient(
                api_key=settings.alpaca.api_key,
                secret_key=settings.alpaca.secret_key,
                paper=settings.alpaca.paper,
            )

    @retry_with_backoff()
    async def get_tradable_symbols(
        self,
        asset_class: AssetClass = AssetClass.US_EQUITY,
        status: AssetStatus = AssetStatus.ACTIVE,
    ) -> List[str]:
        """
        Get list of tradable symbols from Alpaca.

        Args:
            asset_class: Asset class to filter (default: US_EQUITY)
            status: Asset status to filter (default: ACTIVE)

        Returns:
            List of symbol strings (e.g., ["AAPL", "MSFT", ...])

        Raises:
            ValueError: If Alpaca is not configured
        """
        if not self.client:
            raise ValueError("Alpaca API keys not configured")

        def _sync_fetch():
            request = GetAssetsRequest(asset_class=asset_class, status=status)
            assets = self.client.get_all_assets(request)
            return [asset.symbol for asset in assets]

        symbols = await asyncio.to_thread(_sync_fetch)
        log.info("fetched_alpaca_symbols", count=len(symbols), asset_class=asset_class.value)
        return symbols

    async def get_stocks(self) -> List[str]:
        """
        Get list of active US equity stocks.

        Returns:
            List of stock symbols
        """
        return await self.get_tradable_symbols(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )

