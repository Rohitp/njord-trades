"""
Trading cycle API endpoints.

Provides endpoints to trigger and monitor trading cycles.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_session
from src.services.execution import ExecutionService
from src.utils.logging import get_logger
from src.workflows.runner import TradingCycleRunner
from src.workflows.state import Decision, SignalAction

log = get_logger(__name__)


class CycleType(str, Enum):
    """Type of trading cycle."""
    SCHEDULED = "scheduled"
    EVENT = "event"


# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================


class RunCycleRequest(BaseModel):
    """Request to trigger a trading cycle."""

    symbols: list[str] | None = Field(
        default=None,
        description="Symbols to analyze. If not provided, uses watchlist from config.",
    )
    cycle_type: CycleType = Field(
        default=CycleType.SCHEDULED,
        description="Type of cycle: 'scheduled' or 'event'",
    )
    trigger_symbol: str | None = Field(
        default=None,
        description="For event cycles, the symbol that triggered the event.",
    )
    execute: bool = Field(
        default=False,
        description="If true, execute EXECUTE decisions via broker. Defaults to false (dry run).",
    )


class SignalResponse(BaseModel):
    """A trading signal from the Data Agent."""

    id: str
    symbol: str
    action: str
    confidence: float
    proposed_quantity: int
    reasoning: str
    price: float
    rsi_14: float | None = None


class RiskAssessmentResponse(BaseModel):
    """Risk assessment from the Risk Manager."""

    signal_id: str
    approved: bool
    adjusted_quantity: int
    risk_score: float
    hard_constraint_violated: bool = False
    hard_constraint_reason: str | None = None
    concerns: list[str] = []
    reasoning: str = ""


class ValidationResponse(BaseModel):
    """Validation from the Validator."""

    signal_id: str
    approved: bool
    concerns: list[str] = []
    suggestions: list[str] = []
    repetition_detected: bool = False
    sector_clustering_detected: bool = False
    similar_setup_failures: int = 0
    reasoning: str = ""


class FinalDecisionResponse(BaseModel):
    """Final decision from the Meta Agent."""

    signal_id: str
    decision: str
    final_quantity: int
    confidence: float
    reasoning: str


class ExecutionResultResponse(BaseModel):
    """Result of trade execution."""

    signal_id: str
    success: bool
    trade_id: str | None = None
    symbol: str
    action: str
    quantity: int
    requested_price: float
    fill_price: float | None = None
    slippage: float | None = None
    broker_order_id: str | None = None
    error: str | None = None


class CycleResultResponse(BaseModel):
    """Complete result of a trading cycle."""

    cycle_id: str
    trace_id: str | None = None  # For distributed tracing
    cycle_type: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    symbols: list[str]

    # Agent outputs
    signals: list[SignalResponse]
    risk_assessments: list[RiskAssessmentResponse]
    validations: list[ValidationResponse]
    final_decisions: list[FinalDecisionResponse]
    execution_results: list[ExecutionResultResponse] = []

    # Summary
    total_signals: int
    approved_by_risk: int
    approved_by_validator: int
    execute_decisions: int
    executed_trades: int = 0

    # Errors
    errors: list[dict[str, Any]]


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/cycles", tags=["Trading Cycles"])


@router.post("/run", response_model=CycleResultResponse)
async def run_trading_cycle(
    request: RunCycleRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_session),
) -> CycleResultResponse:
    """
    Trigger a trading cycle.

    Runs the full agent pipeline:
    1. DataAgent analyzes market data and generates signals
    2. RiskManager evaluates signals against portfolio constraints
    3. Validator checks for patterns and provides second opinion
    4. MetaAgent synthesizes perspectives and makes final decisions

    Returns the complete cycle results including all agent outputs.

    Pass X-Request-ID header for distributed tracing correlation.
    """
    start_time = datetime.now()

    # Get trace_id from middleware (set from X-Request-ID header or generated)
    trace_id = getattr(http_request.state, "trace_id", None)

    runner = TradingCycleRunner(db_session=db)

    try:
        if request.cycle_type == CycleType.EVENT and request.trigger_symbol:
            result = await runner.run_event_cycle(request.trigger_symbol, trace_id=trace_id)
        else:
            result = await runner.run_scheduled_cycle(request.symbols, trace_id=trace_id)

        # Execute trades if requested
        if request.execute and result.get_execute_decisions():
            log.info("executing_trades", cycle_id=str(result.cycle_id))
            executor = ExecutionService(db_session=db)
            await executor.execute_decisions(result)

    except ValueError as e:
        # Circuit breaker or validation errors
        log.warning("cycle_rejected", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("cycle_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cycle failed: {str(e)}")

    completed_at = datetime.now()
    duration = (completed_at - start_time).total_seconds()

    # Convert state to response
    return CycleResultResponse(
        cycle_id=str(result.cycle_id),
        trace_id=result.trace_id,
        cycle_type=result.cycle_type,
        started_at=result.started_at,
        completed_at=completed_at,
        duration_seconds=duration,
        symbols=result.symbols,
        signals=[
            SignalResponse(
                id=str(s.id),
                symbol=s.symbol,
                action=s.action.value,
                confidence=s.confidence,
                proposed_quantity=s.proposed_quantity,
                reasoning=s.reasoning,
                price=s.price,
                rsi_14=s.rsi_14,
            )
            for s in result.signals
        ],
        risk_assessments=[
            RiskAssessmentResponse(
                signal_id=str(ra.signal_id),
                approved=ra.approved,
                adjusted_quantity=ra.adjusted_quantity,
                risk_score=ra.risk_score,
                hard_constraint_violated=ra.hard_constraint_violated,
                hard_constraint_reason=ra.hard_constraint_reason,
                concerns=ra.concerns,
                reasoning=ra.reasoning,
            )
            for ra in result.risk_assessments
        ],
        validations=[
            ValidationResponse(
                signal_id=str(v.signal_id),
                approved=v.approved,
                concerns=v.concerns,
                suggestions=v.suggestions,
                repetition_detected=v.repetition_detected,
                sector_clustering_detected=v.sector_clustering_detected,
                similar_setup_failures=v.similar_setup_failures,
                reasoning=v.reasoning,
            )
            for v in result.validations
        ],
        final_decisions=[
            FinalDecisionResponse(
                signal_id=str(fd.signal_id),
                decision=fd.decision.value,
                final_quantity=fd.final_quantity,
                confidence=fd.confidence,
                reasoning=fd.reasoning,
            )
            for fd in result.final_decisions
        ],
        execution_results=[
            ExecutionResultResponse(
                signal_id=str(er.signal_id),
                success=er.success,
                trade_id=str(er.trade_id) if er.trade_id else None,
                symbol=er.symbol,
                action=er.action,
                quantity=er.quantity,
                requested_price=er.requested_price,
                fill_price=er.fill_price,
                slippage=er.slippage,
                broker_order_id=er.broker_order_id,
                error=er.error,
            )
            for er in result.execution_results
        ],
        total_signals=len(result.signals),
        approved_by_risk=len([ra for ra in result.risk_assessments if ra.approved]),
        approved_by_validator=len([v for v in result.validations if v.approved]),
        execute_decisions=len(result.get_execute_decisions()),
        executed_trades=len([er for er in result.execution_results if er.success]),
        errors=result.errors,
    )
