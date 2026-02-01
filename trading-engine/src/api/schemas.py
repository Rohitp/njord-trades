"""Pydantic schemas for API request/response models."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.database.models import EventType, TradeAction, TradeStatus


class EventCreate(BaseModel):
    """Schema for creating a new event."""
    event_type: str = Field(..., description="Type of event")
    aggregate_id: str | None = Field(None, description="Identifier for the aggregate")
    data: dict = Field(default_factory=dict, description="Event data (JSONB)")
    event_metadata: dict = Field(default_factory=dict, description="Event metadata (JSONB)")

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event_type matches allowed values."""
        allowed_types = [e.value for e in EventType]
        if v not in allowed_types:
            raise ValueError(f"event_type must be one of: {', '.join(allowed_types)}")
        return v


class EventResponse(BaseModel):
    """Schema for event response."""
    id: UUID
    timestamp: datetime
    event_type: str
    aggregate_id: str | None
    data: dict
    event_metadata: dict

    model_config = {"from_attributes": True}


class EventListResponse(BaseModel):
    """Schema for paginated event list response."""
    events: list[EventResponse]
    total: int
    limit: int
    offset: int


class PortfolioStateResponse(BaseModel):
    """Schema for portfolio state response."""
    cash: float
    total_value: float
    deployed_capital: float
    peak_value: float
    updated_at: datetime

    model_config = {"from_attributes": True}


class PositionResponse(BaseModel):
    """Schema for position response."""
    id: UUID
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    current_value: float
    sector: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PositionListResponse(BaseModel):
    """Schema for positions list response."""
    positions: list[PositionResponse]
    total_count: int


class TradeCreate(BaseModel):
    """Schema for creating a new trade."""
    symbol: str = Field(..., description="Stock symbol")
    action: str = Field(..., description="BUY or SELL")
    quantity: int = Field(..., gt=0, description="Number of shares")
    price: float = Field(..., gt=0, description="Price per share")
    signal_confidence: float = Field(0, ge=0, le=1, description="Agent signal confidence")
    risk_score: float = Field(0, ge=0, le=1, description="Risk assessment score")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = [e.value for e in TradeAction]
        if v.upper() not in allowed:
            raise ValueError(f"action must be one of: {', '.join(allowed)}")
        return v.upper()

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper()


class TradeResponse(BaseModel):
    """Schema for trade response."""
    id: UUID
    symbol: str
    action: str
    quantity: int
    price: float
    total_value: float
    status: str
    signal_confidence: float
    risk_score: float
    outcome: str | None
    pnl: float | None
    pnl_pct: float | None
    broker_order_id: str | None
    fill_price: float | None
    slippage: float | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TradeListResponse(BaseModel):
    """Schema for paginated trades list response."""
    trades: list[TradeResponse]
    total: int
    limit: int
    offset: int

