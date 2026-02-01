"""Capital events endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import (
    CapitalEventCreate,
    CapitalEventListResponse,
    CapitalEventResponse,
)
from src.database.models import CapitalEvent
from src.utils.logging import get_logger

router = APIRouter(prefix="/api/capital", tags=["Capital"])
log = get_logger(__name__)


@router.post("", response_model=CapitalEventResponse, status_code=201)
async def create_capital_event(
    event: CapitalEventCreate,
    session: AsyncSession = Depends(get_db),
) -> CapitalEvent:
    """
    Record a capital event (deposit, withdrawal, realized P&L).

    Used to track all money movements for accurate performance calculation.
    """
    db_event = CapitalEvent(
        event_type=event.event_type,
        amount=event.amount,
        balance_after=event.balance_after,
        description=event.description,
        trade_id=event.trade_id,
    )
    session.add(db_event)
    log.info(
        "capital_event_created",
        event_type=event.event_type,
        amount=event.amount,
        balance_after=event.balance_after,
    )
    return db_event


@router.get("", response_model=CapitalEventListResponse)
async def list_capital_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db),
) -> CapitalEventListResponse:
    """List capital events with pagination."""
    query = select(CapitalEvent).order_by(CapitalEvent.created_at.desc())
    count_query = select(func.count()).select_from(CapitalEvent)

    if event_type:
        query = query.where(CapitalEvent.event_type == event_type.upper())
        count_query = count_query.where(CapitalEvent.event_type == event_type.upper())

    total_result = await session.execute(count_query)
    total = total_result.scalar_one()

    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    events = list(result.scalars().all())

    return CapitalEventListResponse(events=events, total=total, limit=limit, offset=offset)


@router.get("/summary")
async def get_capital_summary(
    session: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get summary of capital events.

    Returns total deposits, withdrawals, realized P&L.
    """
    from src.database.models import CapitalEventType

    result = await session.execute(
        select(
            CapitalEvent.event_type,
            func.sum(CapitalEvent.amount).label("total"),
            func.count().label("count"),
        ).group_by(CapitalEvent.event_type)
    )

    summary = {
        "deposits": 0,
        "withdrawals": 0,
        "realized_profit": 0,
        "realized_loss": 0,
        "net_deposits": 0,
        "net_pnl": 0,
    }

    for row in result:
        event_type = row.event_type
        total = float(row.total or 0)

        if event_type == CapitalEventType.DEPOSIT.value:
            summary["deposits"] = total
        elif event_type == CapitalEventType.WITHDRAWAL.value:
            summary["withdrawals"] = total
        elif event_type == CapitalEventType.REALIZED_PROFIT.value:
            summary["realized_profit"] = total
        elif event_type == CapitalEventType.REALIZED_LOSS.value:
            summary["realized_loss"] = total

    summary["net_deposits"] = summary["deposits"] - abs(summary["withdrawals"])
    summary["net_pnl"] = summary["realized_profit"] - abs(summary["realized_loss"])

    return summary
