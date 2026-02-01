from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    EventCreate,
    EventListResponse,
    EventResponse,
    PortfolioStateResponse,
    PositionListResponse,
    PositionResponse,
    TradeCreate,
    TradeListResponse,
    TradeResponse,
)
from src.database.connection import get_session
from src.database.models import Event, PortfolioState, Position, Trade

tags_metadata = [
    {"name": "Health", "description": "System health checks"},
    {"name": "Events", "description": "Append-only event log for audit trail"},
    {"name": "Portfolio", "description": "Portfolio state and positions"},
    {"name": "Trades", "description": "Trade history and execution"},
]

app = FastAPI(
    title="Trading Engine",
    description="Multi-agent LLM-powered trading system API",
    version="0.1.0",
    openapi_tags=tags_metadata,
)


@app.get("/health", tags=["Health"])
async def health_check(session: AsyncSession = Depends(get_session)) -> dict:
    """
    Health check endpoint that verifies database connectivity.
    """
    try:
        await session.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "database": db_status,
    }


@app.post("/api/events", response_model=EventResponse, status_code=201, tags=["Events"])
async def create_event(
    event: EventCreate,
    session: AsyncSession = Depends(get_session),
) -> EventResponse:
    """
    Append an event to the event log.
    
    Events are append-only and immutable. This endpoint creates a new event
    with full agent reasoning stored in the data field.
    """
    db_event = Event(
        event_type=event.event_type,
        aggregate_id=event.aggregate_id,
        data=event.data,
        event_metadata=event.event_metadata,
    )
    session.add(db_event)
    # get_session() will commit automatically
    # Return the event (will be converted to EventResponse via response_model)
    return db_event


@app.get("/api/events", response_model=EventListResponse, tags=["Events"])
async def list_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    offset: int = Query(0, ge=0, description="Number of events to skip"),
    session: AsyncSession = Depends(get_session),
) -> EventListResponse:
    """
    List events with pagination, optionally filtered by type.
    
    Returns paginated list of events ordered by timestamp (newest first).
    """
    # Build query
    query = select(Event).order_by(Event.timestamp.desc())
    count_query = select(func.count()).select_from(Event)
    
    # Apply filters
    if event_type:
        query = query.where(Event.event_type == event_type)
        count_query = count_query.where(Event.event_type == event_type)
    
    # Get total count
    total_result = await session.execute(count_query)
    total = total_result.scalar_one()
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    # Execute query
    result = await session.execute(query)
    events = list(result.scalars().all())
    
    return EventListResponse(
        events=events,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/portfolio", response_model=PortfolioStateResponse, tags=["Portfolio"])
async def get_portfolio(
    session: AsyncSession = Depends(get_session),
) -> PortfolioState:
    """Get current portfolio state."""
    result = await session.execute(select(PortfolioState).where(PortfolioState.id == 1))
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not initialized")
    return portfolio


@app.get("/api/positions", response_model=PositionListResponse, tags=["Portfolio"])
async def list_positions(
    sector: str | None = Query(None, description="Filter by sector"),
    session: AsyncSession = Depends(get_session),
) -> PositionListResponse:
    """List all current positions."""
    query = select(Position).order_by(Position.symbol)
    count_query = select(func.count()).select_from(Position)

    if sector:
        query = query.where(Position.sector == sector)
        count_query = count_query.where(Position.sector == sector)

    result = await session.execute(query)
    positions = list(result.scalars().all())

    count_result = await session.execute(count_query)
    total = count_result.scalar_one()

    return PositionListResponse(positions=positions, total_count=total)


@app.get("/api/positions/{symbol}", response_model=PositionResponse, tags=["Portfolio"])
async def get_position(
    symbol: str,
    session: AsyncSession = Depends(get_session),
) -> Position:
    """Get position for a specific symbol."""
    result = await session.execute(
        select(Position).where(Position.symbol == symbol.upper())
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=404, detail=f"No position for {symbol}")
    return position


@app.post("/api/trades", response_model=TradeResponse, status_code=201, tags=["Trades"])
async def create_trade(
    trade: TradeCreate,
    session: AsyncSession = Depends(get_session),
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
    await session.flush()
    await session.refresh(db_trade)
    return db_trade


@app.get("/api/trades", response_model=TradeListResponse, tags=["Trades"])
async def list_trades(
    symbol: str | None = Query(None, description="Filter by symbol"),
    outcome: str | None = Query(None, description="Filter by outcome (WIN/LOSS/BREAKEVEN/OPEN)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades"),
    offset: int = Query(0, ge=0, description="Number of trades to skip"),
    session: AsyncSession = Depends(get_session),
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


@app.get("/api/trades/{trade_id}", response_model=TradeResponse, tags=["Trades"])
async def get_trade(
    trade_id: str,
    session: AsyncSession = Depends(get_session),
) -> Trade:
    """Get a specific trade by ID."""
    result = await session.execute(select(Trade).where(Trade.id == trade_id))
    trade = result.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    return trade
