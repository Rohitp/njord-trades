"""yfinance market data provider adapter."""

import asyncio
from datetime import datetime, timedelta

import yfinance as yf

from src.services.market_data.provider import (
    MarketDataProvider,
    OHLCV,
    Quote,
    TechnicalIndicators,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class YFinanceProvider(MarketDataProvider):
    """Market data provider using yfinance."""

    @property
    def name(self) -> str:
        return "yfinance"

    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol."""
        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return Quote(
                symbol=symbol,
                price=info.last_price,
                bid=getattr(info, 'bid', None),
                ask=getattr(info, 'ask', None),
                volume=getattr(info, 'last_volume', None),
                timestamp=datetime.now(),
            )

        return await asyncio.to_thread(_fetch)

    async def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Get current quotes for multiple symbols."""
        tasks = [self.get_quote(symbol) for symbol in symbols]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def get_bars(self, symbol: str, days: int = 200) -> list[OHLCV]:
        """Get historical OHLCV bars."""
        def _fetch():
            ticker = yf.Ticker(symbol)
            # Add buffer days for indicator calculation
            start_date = datetime.now() - timedelta(days=days + 50)
            df = ticker.history(start=start_date)

            bars = []
            for idx, row in df.iterrows():
                bars.append(OHLCV(
                    symbol=symbol,
                    timestamp=idx.to_pydatetime(),
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume']),
                ))
            return bars

        return await asyncio.to_thread(_fetch)

    async def get_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Calculate technical indicators from historical data."""
        bars = await self.get_bars(symbol, days=200)

        if not bars:
            raise ValueError(f"No data available for {symbol}")

        closes = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]
        current_price = closes[-1]
        current_volume = volumes[-1]

        # Calculate SMAs
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
        sma_200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None

        # Calculate RSI (14-period)
        rsi_14 = self._calculate_rsi(closes, 14) if len(closes) >= 15 else None

        # Volume analysis
        volume_avg_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
        volume_ratio = current_volume / volume_avg_20 if volume_avg_20 else None

        return TechnicalIndicators(
            symbol=symbol,
            price=current_price,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            rsi_14=rsi_14,
            volume_avg_20=volume_avg_20,
            volume_ratio=volume_ratio,
        )

    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        recent_deltas = deltas[-(period):]

        gains = [d if d > 0 else 0 for d in recent_deltas]
        losses = [-d if d < 0 else 0 for d in recent_deltas]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
