"""Trade endpoints."""

from uuid import UUID as UUIDType

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import TradeCreate, TradeListResponse, TradeResponse
from src.database.models import Trade
from src.utils.logging import get_logger

router = APIRouter(prefix="/api/trades", tags=["Trades"])
log = get_logger(__name__)


@router.post("", response_model=TradeResponse, status_code=201)
async def create_trade(
    trade: TradeCreate,
    session: AsyncSession = Depends(get_db),
) -> Trade:
    """
    Record a new trade.

    This endpoint records trade intent. Actual execution happens
    through the trading workflow which updates status and fill details.
    """
    db_trade = Trade(
        symbol=trade.symbol,
        action=trade.action,
        quantity=trade.quantity,
        price=trade.price,
        total_value=trade.quantity * trade.price,
        signal_confidence=trade.signal_confidence,
        risk_score=trade.risk_score,
    )
    session.add(db_trade)
    # get_session() will commit automatically
    log.info("trade_created", trade_id=str(db_trade.id), symbol=trade.symbol, action=trade.action)
    return db_trade


@router.get("", response_model=TradeListResponse)
async def list_trades(
    symbol: str | None = Query(None, description="Filter by symbol"),
    outcome: str | None = Query(None, description="Filter by outcome (WIN/LOSS/BREAKEVEN/OPEN)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades"),
    offset: int = Query(0, ge=0, description="Number of trades to skip"),
    session: AsyncSession = Depends(get_db),
) -> TradeListResponse:
    """List trades with pagination and optional filters."""
    query = select(Trade).order_by(Trade.created_at.desc())
    count_query = select(func.count()).select_from(Trade)

    if symbol:
        query = query.where(Trade.symbol == symbol.upper())
        count_query = count_query.where(Trade.symbol == symbol.upper())
    if outcome:
        query = query.where(Trade.outcome == outcome.upper())
        count_query = count_query.where(Trade.outcome == outcome.upper())

    total_result = await session.execute(count_query)
    total = total_result.scalar_one()

    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    trades = list(result.scalars().all())

    return TradeListResponse(trades=trades, total=total, limit=limit, offset=offset)


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(
    trade_id: str,
    session: AsyncSession = Depends(get_db),
) -> Trade:
    """Get a specific trade by ID."""
    try:
        trade_uuid = UUIDType(trade_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid trade ID format: {trade_id}")
    
    result = await session.execute(select(Trade).where(Trade.id == trade_uuid))
    trade = result.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    return trade

