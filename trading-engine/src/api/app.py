from datetime import datetime
from uuid import UUID

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_session
from src.database.models import Event

app = FastAPI(title="Trading Engine", version="0.1.0")


class EventCreate(BaseModel):
    event_type: str
    aggregate_id: str | None = None
    data: dict = {}
    event_metadata: dict = {}


class EventResponse(BaseModel):
    id: UUID
    timestamp: datetime
    event_type: str
    aggregate_id: str | None
    data: dict
    event_metadata: dict

    model_config = {"from_attributes": True}


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


@app.post("/api/events", response_model=EventResponse)
async def create_event(
    event: EventCreate,
    session: AsyncSession = Depends(get_session),
) -> Event:
    """Append an event to the event log."""
    db_event = Event(
        event_type=event.event_type,
        aggregate_id=event.aggregate_id,
        data=event.data,
        event_metadata=event.event_metadata,
    )
    session.add(db_event)
    await session.flush()
    await session.refresh(db_event)
    return db_event


@app.get("/api/events", response_model=list[EventResponse])
async def list_events(
    event_type: str | None = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
) -> list[Event]:
    """List events, optionally filtered by type."""
    query = select(Event).order_by(Event.timestamp.desc()).limit(limit)
    if event_type:
        query = query.where(Event.event_type == event_type)
    result = await session.execute(query)
    return list(result.scalars().all())
