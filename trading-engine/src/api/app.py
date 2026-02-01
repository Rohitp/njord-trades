from fastapi import Depends, FastAPI, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import EventCreate, EventListResponse, EventResponse
from src.database.connection import get_session
from src.database.models import Event

app = FastAPI(title="Trading Engine", version="0.1.0")


@app.get("/health")
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


@app.post("/api/events", response_model=EventResponse, status_code=201)
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


@app.get("/api/events", response_model=EventListResponse)
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
