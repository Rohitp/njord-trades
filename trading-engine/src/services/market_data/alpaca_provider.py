"""Alpaca market data provider adapter."""

import re
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from src.config import settings
from src.services.market_data.provider import (
    MarketDataProvider,
    OHLCV,
    Quote,
    TechnicalIndicators,
)
from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

log = get_logger(__name__)


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize symbol format.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        Uppercase, validated symbol
        
    Raises:
        ValueError: If symbol format is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    symbol = symbol.upper().strip()
    
    # Basic validation: alphanumeric, 1-5 characters (most exchanges)
    if not re.match(r'^[A-Z0-9]{1,5}$', symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    return symbol


class AlpacaProvider(MarketDataProvider):
    """Market data provider using Alpaca API."""

    def __init__(self):
        self._client = None

    @property
    def client(self) -> StockHistoricalDataClient:
        """Lazy initialization of Alpaca client."""
        if self._client is None:
            self._client = StockHistoricalDataClient(
                api_key=settings.alpaca.api_key,
                secret_key=settings.alpaca.secret_key,
            )
        return self._client

    @property
    def name(self) -> str:
        return "alpaca"

    @retry_with_backoff()
    async def _fetch_quote(self, symbol: str) -> Quote:
        """Internal method to fetch quote with retry logic."""
        symbol = validate_symbol(symbol)
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        response = self.client.get_stock_latest_quote(request)
        
        if symbol not in response:
            raise ValueError(f"Symbol {symbol} not found in Alpaca response")
        
        quote_data = response[symbol]
        
        # Handle None bid/ask prices
        if quote_data.bid_price is None and quote_data.ask_price is None:
            raise ValueError(f"No price data available for {symbol}")
        elif quote_data.bid_price is None:
            price = quote_data.ask_price
        elif quote_data.ask_price is None:
            price = quote_data.bid_price
        else:
            price = (quote_data.bid_price + quote_data.ask_price) / 2

        return Quote(
            symbol=symbol,
            price=price,
            bid=quote_data.bid_price,
            ask=quote_data.ask_price,
            volume=None,  # Latest quote doesn't include volume
            timestamp=quote_data.timestamp,
        )

    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol."""
        return await self._fetch_quote(symbol)

    @retry_with_backoff()
    async def _fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        """Internal method to fetch multiple quotes with retry logic."""
        validated_symbols = [validate_symbol(s) for s in symbols]
        request = StockLatestQuoteRequest(symbol_or_symbols=validated_symbols)
        response = self.client.get_stock_latest_quote(request)

        quotes = []
        for symbol in validated_symbols:
            if symbol not in response:
                log.warning("symbol_not_found", symbol=symbol, provider="alpaca")
                continue
            
            quote_data = response[symbol]
            
            # Handle None bid/ask prices
            if quote_data.bid_price is None and quote_data.ask_price is None:
                log.warning("no_price_data", symbol=symbol, provider="alpaca")
                continue
            elif quote_data.bid_price is None:
                price = quote_data.ask_price
            elif quote_data.ask_price is None:
                price = quote_data.bid_price
            else:
                price = (quote_data.bid_price + quote_data.ask_price) / 2
            
            quotes.append(Quote(
                symbol=symbol,
                price=price,
                bid=quote_data.bid_price,
                ask=quote_data.ask_price,
                volume=None,
                timestamp=quote_data.timestamp,
            ))
        return quotes

    async def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Get current quotes for multiple symbols."""
        return await self._fetch_quotes(symbols)

    @retry_with_backoff()
    async def _fetch_bars(self, symbol: str, days: int = 200) -> list[OHLCV]:
        """Internal method to fetch bars with retry logic."""
        symbol = validate_symbol(symbol)
        # Add buffer for indicator calculation
        start_date = datetime.now() - timedelta(days=days + 50)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
        )
        response = self.client.get_stock_bars(request)

        if symbol not in response:
            raise ValueError(f"Symbol {symbol} not found in Alpaca response")

        bars = []
        for bar in response[symbol]:
            bars.append(OHLCV(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            ))
        
        if not bars:
            raise ValueError(f"No historical data available for {symbol}")
        
        return bars

    async def get_bars(self, symbol: str, days: int = 200) -> list[OHLCV]:
        """Get historical OHLCV bars."""
        return await self._fetch_bars(symbol, days)

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

    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float | None:
        """
        Calculate RSI indicator using Wilder's smoothing method (industry standard).
        
        Uses exponential smoothing (Wilder's method) instead of simple average.
        This is the standard RSI calculation used by trading platforms.
        """
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # First period: simple average
        gains = [d if d > 0 else 0 for d in deltas[:period]]
        losses = [-d if d < 0 else 0 for d in deltas[:period]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        # Subsequent periods: Wilder's smoothing
        # avg = (prev_avg * (period - 1) + current) / period
        for i in range(period, len(deltas)):
            current_gain = deltas[i] if deltas[i] > 0 else 0
            current_loss = -deltas[i] if deltas[i] < 0 else 0
            
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
