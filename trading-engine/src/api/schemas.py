"""Pydantic schemas for API request/response models."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.database.models import CapitalEventType, EventType, StrategyStatus, TradeAction, TradeStatus


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


# =============================================================================
# CAPITAL EVENTS
# =============================================================================


class CapitalEventCreate(BaseModel):
    """Schema for creating a capital event."""
    event_type: str = Field(..., description="Type of capital event")
    amount: float = Field(..., description="Amount (positive for inflow, negative for outflow)")
    balance_after: float = Field(..., description="Portfolio balance after this event")
    description: str | None = Field(None, description="Optional description")
    trade_id: UUID | None = Field(None, description="Associated trade ID if applicable")

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        allowed = [e.value for e in CapitalEventType]
        if v.upper() not in allowed:
            raise ValueError(f"event_type must be one of: {', '.join(allowed)}")
        return v.upper()


class CapitalEventResponse(BaseModel):
    """Schema for capital event response."""
    id: UUID
    event_type: str
    amount: float
    balance_after: float
    description: str | None
    trade_id: UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CapitalEventListResponse(BaseModel):
    """Schema for paginated capital events list."""
    events: list[CapitalEventResponse]
    total: int
    limit: int
    offset: int


# =============================================================================
# STRATEGIES
# =============================================================================


class StrategyCreate(BaseModel):
    """Schema for creating a strategy."""
    name: str = Field(..., min_length=1, max_length=50, description="Strategy name")
    description: str | None = Field(None, description="Strategy description")
    allocation: float = Field(1.0, ge=0, le=1, description="Allocation (0.0-1.0)")
    capital: float = Field(0, ge=0, description="Initial capital")


class StrategyUpdate(BaseModel):
    """Schema for updating a strategy."""
    status: str | None = Field(None, description="Strategy status")
    description: str | None = Field(None, description="Strategy description")
    allocation: float | None = Field(None, ge=0, le=1, description="Allocation (0.0-1.0)")
    capital: float | None = Field(None, ge=0, description="Capital allocated")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = [e.value for e in StrategyStatus]
        if v.upper() not in allowed:
            raise ValueError(f"status must be one of: {', '.join(allowed)}")
        return v.upper()


class StrategyResponse(BaseModel):
    """Schema for strategy response."""
    id: UUID
    name: str
    status: str
    description: str | None
    allocation: float
    capital: float
    sharpe_30d: float | None
    win_rate_30d: float | None
    total_trades: int
    total_pnl: float
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class StrategyListResponse(BaseModel):
    """Schema for strategies list."""
    strategies: list[StrategyResponse]
    total: int


# =============================================================================
# DISCOVERY SCHEMAS
# =============================================================================


class PickerPerformanceResponse(BaseModel):
    """Schema for picker performance metrics."""
    picker_name: str
    total_suggestions: int
    suggestions_with_returns: int
    pending_suggestions: int  # Suggestions waiting for forward return calculation

    # Win rates
    win_rate_1d: float | None = None
    win_rate_5d: float | None = None
    win_rate_20d: float | None = None

    # Average returns
    avg_return_1d: float | None = None
    avg_return_5d: float | None = None
    avg_return_20d: float | None = None

    # Median returns
    median_return_1d: float | None = None
    median_return_5d: float | None = None
    median_return_20d: float | None = None

    # Best/worst returns
    best_return_1d: float | None = None
    best_return_5d: float | None = None
    best_return_20d: float | None = None

    worst_return_1d: float | None = None
    worst_return_5d: float | None = None
    worst_return_20d: float | None = None

    model_config = {"from_attributes": True}


class DiscoveryPerformanceResponse(BaseModel):
    """Schema for discovery performance API response."""
    pickers: list[PickerPerformanceResponse]
    total_pickers: int
    min_suggestions: int
    limit: int | None = None  # Pagination limit
    offset: int | None = None  # Pagination offset


class ABTestMetricsResponse(BaseModel):
    """Schema for A/B test metrics comparing hypothetical vs actual trades."""
    total_hypothetical_trades: int
    total_actual_trades: int
    matched_trades: int

    # Win rates
    hypothetical_win_rate: float | None = None
    actual_win_rate: float | None = None

    # Average returns
    hypothetical_avg_return: float | None = None
    actual_avg_return: float | None = None

    # Total P&L
    hypothetical_total_pnl: float | None = None
    actual_total_pnl: float | None = None

    # Divergence metrics
    convergence_rate: float | None = None
    hypothetical_outperformed: int = 0
    actual_outperformed: int = 0

