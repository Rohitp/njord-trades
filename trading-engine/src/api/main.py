"""FastAPI application setup and configuration."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exceptions import TradingSystemError, trading_error_handler
from src.api.middleware import RequestLoggingMiddleware
from src.api.routers import events, health, market_data, portfolio, trades
from src.utils.logging import get_logger, setup_logging

# Setup logging first
setup_logging()
log = get_logger(__name__)

# OpenAPI tags metadata
tags_metadata = [
    {"name": "Health", "description": "System health checks"},
    {"name": "Events", "description": "Append-only event log for audit trail"},
    {"name": "Portfolio", "description": "Portfolio state and positions"},
    {"name": "Trades", "description": "Trade history and execution"},
    {"name": "Market Data", "description": "Real-time market data and indicators"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    log.info("trading_engine_starting", version="0.1.0")
    yield
    log.info("trading_engine_stopping")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Trading Engine",
        description="Multi-agent LLM-powered trading system API",
        version="0.1.0",
        openapi_tags=tags_metadata,
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Add exception handlers
    app.add_exception_handler(TradingSystemError, trading_error_handler)

    # Include routers
    app.include_router(health.router)
    app.include_router(events.router)
    app.include_router(portfolio.router)
    app.include_router(trades.router)
    app.include_router(market_data.router)

    return app


# Create the app instance
app = create_app()

