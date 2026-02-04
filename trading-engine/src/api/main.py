"""FastAPI application setup and configuration."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exceptions import TradingSystemError, trading_error_handler
from src.api.middleware import RequestLoggingMiddleware
from src.api.routers import capital, cycles, events, health, market_data, metrics, portfolio, scheduler, strategies, system, trades
from src.config import settings
from src.utils.logging import get_logger, setup_logging

# Setup logging first
setup_logging()
log = get_logger(__name__)

# OpenAPI tags metadata
tags_metadata = [
    {"name": "Health", "description": "System health checks"},
    {"name": "System", "description": "Circuit breaker and trading controls"},
    {"name": "Trading Cycles", "description": "Run multi-agent trading cycles"},
    {"name": "Scheduler", "description": "Scheduled job management"},
    {"name": "Events", "description": "Append-only event log for audit trail"},
    {"name": "Portfolio", "description": "Portfolio state and positions"},
    {"name": "Trades", "description": "Trade history and execution"},
    {"name": "Market Data", "description": "Real-time market data and indicators"},
    {"name": "Capital", "description": "Capital events (deposits, withdrawals, P&L)"},
    {"name": "Strategies", "description": "Trading strategy management"},
    {"name": "Metrics", "description": "Prometheus metrics for monitoring"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    log.info("trading_engine_starting", version="0.1.0")

    # Start scheduler if not in test environment
    scheduler_instance = None
    if settings.environment != "test":
        from src.scheduler import start_scheduler, stop_scheduler
        scheduler_instance = start_scheduler()
        log.info("scheduler_integrated", job_count=len(scheduler_instance.get_jobs()))

    yield

    # Stop scheduler
    if scheduler_instance is not None:
        from src.scheduler import stop_scheduler
        stop_scheduler()

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
    app.include_router(system.router)
    app.include_router(cycles.router)
    app.include_router(scheduler.router)
    app.include_router(events.router)
    app.include_router(portfolio.router)
    app.include_router(trades.router)
    app.include_router(market_data.router)
    app.include_router(capital.router)
    app.include_router(strategies.router)
    app.include_router(metrics.router)

    return app


# Create the app instance
app = create_app()

