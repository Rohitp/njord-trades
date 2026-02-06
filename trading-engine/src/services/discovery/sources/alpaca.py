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

    async def get_stocks(
        self, exclude_warrants: bool = True, exclude_otc: bool = True, exclude_units: bool = True
    ) -> List[str]:
        """
        Get list of active US equity stocks with optional filtering.

        Args:
            exclude_warrants: Exclude warrants (symbols ending in W) (default: True)
            exclude_otc: Exclude OTC/pink sheet stocks (5-letter symbols ending in F) (default: True)
            exclude_units: Exclude units/paired securities (symbols ending in U) (default: True)

        Returns:
            List of stock symbols

        Note:
            This filters out common problematic symbols that often lack price data:
            - Warrants (e.g., ABCW): Derivative securities, often delisted or inactive
            - OTC stocks (e.g., FNVUF): Foreign/OTC stocks with limited data availability
            - Units (e.g., ABCU): Paired securities that may have data issues
        """
        symbols = await self.get_tradable_symbols(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )

        # Filter out problematic symbols
        filtered = []
        excluded_count = 0
        exclusion_reasons = {"warrants": 0, "otc": 0, "units": 0}

        for symbol in symbols:
            # Exclude warrants (usually end in W)
            if exclude_warrants and symbol.endswith("W") and len(symbol) > 1:
                excluded_count += 1
                exclusion_reasons["warrants"] += 1
                continue

            # Exclude units/paired securities (usually end in U)
            if exclude_units and symbol.endswith("U") and len(symbol) > 1:
                excluded_count += 1
                exclusion_reasons["units"] += 1
                continue

            # Exclude OTC/pink sheet stocks (5-letter tickers ending in F)
            # These are typically foreign companies trading OTC and often have data issues
            if exclude_otc and len(symbol) == 5 and symbol.endswith("F"):
                excluded_count += 1
                exclusion_reasons["otc"] += 1
                continue

            filtered.append(symbol)

        if excluded_count > 0:
            log.info(
                "filtered_problematic_symbols",
                total=len(symbols),
                excluded=excluded_count,
                remaining=len(filtered),
                warrants_excluded=exclusion_reasons["warrants"],
                otc_excluded=exclusion_reasons["otc"],
                units_excluded=exclusion_reasons["units"],
            )

        return filtered
