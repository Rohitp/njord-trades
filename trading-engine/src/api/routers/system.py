"""
System management API endpoints.

Provides endpoints for circuit breaker control and system status.
"""

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_session
from src.database.models import PortfolioState, SystemState
from src.services.circuit_breaker import CircuitBreakerService
from src.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/system", tags=["System"])


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status response."""

    active: bool
    reason: str | None = None
    trading_enabled: bool = True
    current_drawdown_pct: float | None = None


class CircuitBreakerResetRequest(BaseModel):
    """Request to reset circuit breaker."""

    reason: str = Field(
        default="Manual reset via API",
        description="Reason for resetting the circuit breaker",
    )


class CircuitBreakerResetResponse(BaseModel):
    """Response from circuit breaker reset."""

    success: bool
    message: str


@router.get("/circuit-breaker", response_model=CircuitBreakerStatus)
async def get_circuit_breaker_status(
    db: AsyncSession = Depends(get_session),
) -> CircuitBreakerStatus:
    """
    Get current circuit breaker status.

    Returns whether trading is halted and the reason.
    """
    # Get system state
    stmt = select(SystemState).where(SystemState.id == 1)
    result = await db.execute(stmt)
    system_state = result.scalar_one_or_none()

    # Get portfolio for drawdown calculation
    portfolio_stmt = select(PortfolioState).where(PortfolioState.id == 1)
    portfolio_result = await db.execute(portfolio_stmt)
    portfolio = portfolio_result.scalar_one_or_none()

    drawdown = None
    if portfolio and portfolio.peak_value > 0:
        drawdown = ((portfolio.peak_value - portfolio.total_value) / portfolio.peak_value) * 100

    if system_state is None:
        return CircuitBreakerStatus(
            active=False,
            reason=None,
            trading_enabled=True,
            current_drawdown_pct=drawdown,
        )

    return CircuitBreakerStatus(
        active=system_state.circuit_breaker_active,
        reason=system_state.circuit_breaker_reason,
        trading_enabled=system_state.trading_enabled,
        current_drawdown_pct=drawdown,
    )


@router.post("/circuit-breaker/reset", response_model=CircuitBreakerResetResponse)
async def reset_circuit_breaker(
    request: CircuitBreakerResetRequest,
    db: AsyncSession = Depends(get_session),
) -> CircuitBreakerResetResponse:
    """
    Manually reset the circuit breaker.

    Use with caution - only reset when you've addressed the underlying issue.
    """
    service = CircuitBreakerService(db)
    success = await service.reset(request.reason)

    if success:
        log.info("circuit_breaker_reset_via_api", reason=request.reason)
        return CircuitBreakerResetResponse(
            success=True,
            message="Circuit breaker has been reset. Trading is now enabled.",
        )
    else:
        return CircuitBreakerResetResponse(
            success=False,
            message="Circuit breaker was not active or does not exist.",
        )


@router.post("/circuit-breaker/evaluate", response_model=CircuitBreakerStatus)
async def evaluate_circuit_breaker(
    db: AsyncSession = Depends(get_session),
) -> CircuitBreakerStatus:
    """
    Manually trigger circuit breaker evaluation.

    Checks all conditions and activates if thresholds are breached.
    Returns the updated status.
    """
    service = CircuitBreakerService(db)
    triggered = await service.evaluate_and_update()

    if triggered:
        log.warning("circuit_breaker_triggered_via_api")

    # Return updated status
    return await get_circuit_breaker_status(db)


@router.post("/trading/disable")
async def disable_trading(
    db: AsyncSession = Depends(get_session),
) -> dict:
    """
    Disable all trading.

    Different from circuit breaker - this is a manual kill switch.
    """
    stmt = select(SystemState).where(SystemState.id == 1)
    result = await db.execute(stmt)
    system_state = result.scalar_one_or_none()

    if system_state is None:
        system_state = SystemState(id=1, trading_enabled=False)
        db.add(system_state)
    else:
        system_state.trading_enabled = False

    await db.commit()

    log.warning("trading_disabled_via_api")
    return {"success": True, "message": "Trading has been disabled."}


@router.post("/trading/enable")
async def enable_trading(
    db: AsyncSession = Depends(get_session),
) -> dict:
    """
    Enable trading.

    Note: If circuit breaker is active, trading will still be halted.
    """
    stmt = select(SystemState).where(SystemState.id == 1)
    result = await db.execute(stmt)
    system_state = result.scalar_one_or_none()

    if system_state is None:
        system_state = SystemState(id=1, trading_enabled=True)
        db.add(system_state)
    else:
        system_state.trading_enabled = True

    await db.commit()

    log.info("trading_enabled_via_api")

    message = "Trading has been enabled."
    if system_state.circuit_breaker_active:
        message += " Note: Circuit breaker is still active."

    return {"success": True, "message": message}
