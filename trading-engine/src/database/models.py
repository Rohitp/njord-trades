import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class EventType(str, Enum):
    """Event types for the event log."""
    SIGNAL_GENERATED = "SignalGenerated"
    RISK_ASSESSED = "RiskAssessed"
    VALIDATION_COMPLETE = "ValidationComplete"
    META_AGENT_DECISION = "MetaAgentDecision"
    TRADE_EXECUTED = "TradeExecuted"


class SystemState(Base):
    """
    System-wide trading state. Single row table.

    Controls whether trading is enabled and tracks circuit breaker status.
    Enforced to be a single-row table via CHECK constraint.
    """
    __tablename__ = "system_state"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    trading_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    circuit_breaker_active: Mapped[bool] = mapped_column(Boolean, default=False)
    circuit_breaker_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    __table_args__ = (
        CheckConstraint('id = 1', name='single_row_check'),
    )


class Event(Base):
    """
    Append-only event log for complete audit trail.

    Stores all agent decisions, trade executions, and system events.
    Data and metadata are JSONB for flexible schema.
    """
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        index=True
    )
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    aggregate_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    data: Mapped[dict] = mapped_column(JSONB, default=dict)
    event_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index('ix_events_data_gin', 'data', postgresql_using='gin'),
    )
