"""Market data provider protocol and data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Quote:
    """Current price quote for a symbol."""
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    timestamp: datetime | None = None


@dataclass
class OHLCV:
    """OHLCV bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TechnicalIndicators:
    """Technical indicators for a symbol."""
    symbol: str
    price: float
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    volume_avg_20: float | None = None
    volume_ratio: float | None = None  # current volume / avg volume


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol."""
        pass

    @abstractmethod
    async def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Get current quotes for multiple symbols."""
        pass

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        days: int = 200,
    ) -> list[OHLCV]:
        """Get historical OHLCV bars."""
        pass

    @abstractmethod
    async def get_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Get technical indicators for a symbol."""
        pass
