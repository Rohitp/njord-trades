"""Market data endpoints."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.services.market_data import MarketDataService, market_data_service
from src.utils.logging import get_logger

router = APIRouter(prefix="/api/market", tags=["Market Data"])
log = get_logger(__name__)


class QuoteResponse(BaseModel):
    """Quote response schema."""
    symbol: str
    price: float
    bid: float | None
    ask: float | None
    volume: int | None


class TechnicalIndicatorsResponse(BaseModel):
    """Technical indicators response schema."""
    symbol: str
    price: float
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    rsi_14: float | None
    volume_avg_20: float | None
    volume_ratio: float | None


@router.get("/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str) -> QuoteResponse:
    """Get current quote for a symbol."""
    try:
        quote = await market_data_service.get_quote(symbol.upper())
        return QuoteResponse(
            symbol=quote.symbol,
            price=quote.price,
            bid=quote.bid,
            ask=quote.ask,
            volume=quote.volume,
        )
    except Exception as e:
        log.error("quote_fetch_failed", symbol=symbol, error=str(e))
        raise HTTPException(status_code=502, detail=f"Failed to fetch quote: {e}")


@router.get("/indicators/{symbol}", response_model=TechnicalIndicatorsResponse)
async def get_indicators(symbol: str) -> TechnicalIndicatorsResponse:
    """Get technical indicators for a symbol."""
    try:
        indicators = await market_data_service.get_technical_indicators(symbol.upper())
        return TechnicalIndicatorsResponse(
            symbol=indicators.symbol,
            price=indicators.price,
            sma_20=indicators.sma_20,
            sma_50=indicators.sma_50,
            sma_200=indicators.sma_200,
            rsi_14=indicators.rsi_14,
            volume_avg_20=indicators.volume_avg_20,
            volume_ratio=indicators.volume_ratio,
        )
    except Exception as e:
        log.error("indicators_fetch_failed", symbol=symbol, error=str(e))
        raise HTTPException(status_code=502, detail=f"Failed to fetch indicators: {e}")
