"""Custom exceptions for the trading system."""

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse


class TradingSystemError(Exception):
    """Base exception for trading system errors."""
    pass


class CircuitBreakerActiveError(TradingSystemError):
    """Raised when circuit breaker is active and trading is halted."""
    pass


class InsufficientCapitalError(TradingSystemError):
    """Raised when there's insufficient capital for a trade."""
    pass


class PositionLimitExceededError(TradingSystemError):
    """Raised when position limit is exceeded."""
    pass


class SectorLimitExceededError(TradingSystemError):
    """Raised when sector limit is exceeded."""
    pass


async def trading_error_handler(request, exc: TradingSystemError):
    """Convert TradingSystemError to HTTPException."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": str(exc), "type": exc.__class__.__name__}
    )

