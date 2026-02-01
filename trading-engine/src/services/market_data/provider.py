"""Market data provider protocol and data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


# =============================================================================
# DATA CLASSES
# =============================================================================
# These classes standardize how market data flows through the system.
#
# WHY WE NEED THEM:
# - Alpaca and yfinance return data in different formats
# - Without standardization, every part of the app would need to handle both formats
# - These classes act as a "common language" that all providers translate into
#
# HOW THEY'RE USED:
# 1. Provider fetches raw data from API (Alpaca JSON, yfinance DataFrame)
# 2. Provider converts raw data into these standard classes
# 3. Rest of app only works with these classes, doesn't know which provider was used
# =============================================================================


@dataclass
class Quote:
    """
    Real-time price snapshot for a stock.

    WHAT IT IS:
        The current price and trading activity for a symbol at a moment in time.
        Think of it as checking the stock ticker on your phone.

    WHEN IT'S USED:
        - Before placing a trade (to get current price)
        - Calculating position value (shares × current price)
        - Checking if price moved significantly (event triggers)

    EXAMPLE:
        Quote(symbol="AAPL", price=150.50, bid=150.45, ask=150.55, volume=1000000)
        Means: AAPL is trading at $150.50, buyers offering $150.45, sellers want $150.55
    """
    symbol: str                     # Stock ticker (e.g., "AAPL", "TSLA")
    price: float                    # Current/last trade price
    bid: float | None = None        # Highest price a buyer is willing to pay right now
    ask: float | None = None        # Lowest price a seller is willing to accept right now
    volume: int | None = None       # Number of shares traded today
    timestamp: datetime | None = None


@dataclass
class OHLCV:
    """
    Historical price bar (candlestick) for one time period.

    WHAT IT IS:
        A summary of all price activity during a period (usually one day).
        The building block for charts and technical analysis.

    WHEN IT'S USED:
        - Calculating moving averages (need many days of close prices)
        - Calculating RSI (need price changes over time)
        - Analyzing volume patterns (is today's volume unusual?)

    EXAMPLE:
        OHLCV(symbol="AAPL", open=148.0, high=152.0, low=147.5, close=150.5, volume=5000000)
        Means: AAPL opened at $148, reached $152, dropped to $147.50, closed at $150.50
    """
    symbol: str                     # Stock ticker
    timestamp: datetime             # When this period started
    open: float                     # First trade price of the period
    high: float                     # Highest price during the period
    low: float                      # Lowest price during the period
    close: float                    # Last trade price of the period
    volume: int                     # Total shares traded during period


@dataclass
class TechnicalIndicators:
    """
    Calculated trading signals derived from historical prices.

    WHAT IT IS:
        Mathematical transformations of price/volume data that help identify
        trends, momentum, and potential trading opportunities.

    WHEN IT'S USED:
        - Data Agent uses these to generate BUY/SELL/HOLD signals
        - Example logic: "Price above SMA_200 AND RSI between 50-70 = uptrend with momentum"

    INDICATORS EXPLAINED:
        SMA (Simple Moving Average):
            - Average price over N days
            - Price above SMA = bullish, below = bearish
            - SMA_20: short-term trend (weeks)
            - SMA_50: medium-term trend (months)
            - SMA_200: long-term trend (year) - the big one institutions watch

        RSI (Relative Strength Index):
            - Momentum oscillator from 0-100
            - Below 30: oversold (potential buy)
            - Above 70: overbought (potential sell)
            - 50-70: healthy uptrend

        Volume Ratio:
            - Today's volume / 20-day average volume
            - Above 2.0: unusual interest, confirms price moves
            - Below 0.5: low interest, moves may not stick
    """
    symbol: str
    price: float
    sma_20: float | None = None         # 20-day moving average
    sma_50: float | None = None         # 50-day moving average
    sma_200: float | None = None        # 200-day moving average
    rsi_14: float | None = None         # 14-day RSI (0-100 scale)
    volume_avg_20: float | None = None  # 20-day average volume
    volume_ratio: float | None = None   # current volume ÷ average volume


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
