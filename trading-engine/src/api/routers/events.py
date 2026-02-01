"""Event log endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import EventCreate, EventListResponse, EventResponse
from src.database.models import Event
from src.utils.logging import get_logger

router = APIRouter(prefix="/api/events", tags=["Events"])
log = get_logger(__name__)


@router.post("", response_model=EventResponse, status_code=201)
async def create_event(
    event: EventCreate,
    session: AsyncSession = Depends(get_db),
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
    log.info("event_created", event_type=event.event_type, aggregate_id=event.aggregate_id)
    return db_event


@router.get("", response_model=EventListResponse)
async def list_events(
    event_type: str | None = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    offset: int = Query(0, ge=0, description="Number of events to skip"),
    session: AsyncSession = Depends(get_db),
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

