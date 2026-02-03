"""
Trading cycle API endpoints.

Provides endpoints to trigger and monitor trading cycles.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_session
from src.workflows.runner import TradingCycleRunner
from src.workflows.state import Decision, SignalAction


# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================


class RunCycleRequest(BaseModel):
    """Request to trigger a trading cycle."""

    symbols: list[str] | None = Field(
        default=None,
        description="Symbols to analyze. If not provided, uses watchlist from config.",
    )
    cycle_type: str = Field(
        default="scheduled",
        description="Type of cycle: 'scheduled' or 'event'",
    )
    trigger_symbol: str | None = Field(
        default=None,
        description="For event cycles, the symbol that triggered the event.",
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


class CycleResultResponse(BaseModel):
    """Complete result of a trading cycle."""

    cycle_id: str
    cycle_type: str
    started_at: datetime
    symbols: list[str]

    # Agent outputs
    signals: list[SignalResponse]
    risk_assessments: list[RiskAssessmentResponse]
    validations: list[ValidationResponse]
    final_decisions: list[FinalDecisionResponse]

    # Summary
    total_signals: int
    approved_by_risk: int
    approved_by_validator: int
    execute_decisions: int

    # Errors
    errors: list[dict[str, Any]]


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/cycles", tags=["Trading Cycles"])


@router.post("/run", response_model=CycleResultResponse)
async def run_trading_cycle(
    request: RunCycleRequest,
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
    """
    runner = TradingCycleRunner(db_session=db)

    try:
        if request.cycle_type == "event" and request.trigger_symbol:
            result = await runner.run_event_cycle(request.trigger_symbol)
        else:
            result = await runner.run_scheduled_cycle(request.symbols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cycle failed: {str(e)}")

    # Convert state to response
    return CycleResultResponse(
        cycle_id=str(result.cycle_id),
        cycle_type=result.cycle_type,
        started_at=result.started_at,
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
        total_signals=len(result.signals),
        approved_by_risk=len([ra for ra in result.risk_assessments if ra.approved]),
        approved_by_validator=len([v for v in result.validations if v.approved]),
        execute_decisions=len(result.get_execute_decisions()),
        errors=result.errors,
    )
