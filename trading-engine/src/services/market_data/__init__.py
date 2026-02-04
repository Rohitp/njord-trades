"""Market data service with provider adapters."""

from src.services.market_data.fundamentals import (
    AlpacaFundamentalsProvider,
    CachedFundamentalsProvider,
    Fundamentals,
    FundamentalsProvider,
)
from src.services.market_data.provider import MarketDataProvider, Quote, OHLCV, TechnicalIndicators
from src.services.market_data.service import MarketDataService, market_data_service

__all__ = [
    "MarketDataProvider",
    "MarketDataService",
    "market_data_service",
    "Quote",
    "OHLCV",
    "TechnicalIndicators",
    "FundamentalsProvider",
    "AlpacaFundamentalsProvider",
    "CachedFundamentalsProvider",
    "Fundamentals",
]
