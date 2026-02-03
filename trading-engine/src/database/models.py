import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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


class TradeAction(str, Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(str, Enum):
    """Trade execution status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TradeOutcome(str, Enum):
    """Trade outcome for completed trades."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


class CapitalEventType(str, Enum):
    """Types of capital events."""
    DEPOSIT = "DEPOSIT"           # Money added to account
    WITHDRAWAL = "WITHDRAWAL"     # Money removed from account
    REALIZED_PROFIT = "REALIZED_PROFIT"   # Profit from closed trade
    REALIZED_LOSS = "REALIZED_LOSS"       # Loss from closed trade
    DIVIDEND = "DIVIDEND"         # Dividend received
    FEE = "FEE"                   # Trading fees, commissions


class StrategyStatus(str, Enum):
    """Strategy status."""
    ACTIVE = "ACTIVE"             # Currently trading
    PAUSED = "PAUSED"             # Temporarily stopped
    DISABLED = "DISABLED"         # Permanently disabled


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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
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


class PortfolioState(Base):
    """
    Current portfolio state. Single row table.

    Tracks cash, total value, and deployed capital.
    Updated on every trade.
    """
    __tablename__ = "portfolio_state"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    cash: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    total_value: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    deployed_capital: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    peak_value: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint('id = 1', name='portfolio_single_row_check'),
    )


class Position(Base):
    """
    Current holdings. One row per symbol.

    Tracks quantity, cost basis, current value, and sector.
    """
    __tablename__ = "positions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    avg_cost: Mapped[float] = mapped_column(Numeric(12, 4), default=0)
    current_price: Mapped[float] = mapped_column(Numeric(12, 4), default=0)
    current_value: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    sector: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )


class Trade(Base):
    """
    Trade history. One row per executed trade.

    Records all trade details including agent confidence scores,
    execution details, and outcome for performance analysis.
    """
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(20), index=True)  # Indexed per requirements
    action: Mapped[str] = mapped_column(String(10))  # BUY/SELL
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Numeric(12, 4))
    total_value: Mapped[float] = mapped_column(Numeric(12, 2))
    status: Mapped[str] = mapped_column(String(20), default=TradeStatus.PENDING.value)

    # Agent scores
    signal_confidence: Mapped[float] = mapped_column(Numeric(5, 4), default=0)
    risk_score: Mapped[float] = mapped_column(Numeric(5, 4), default=0)

    # Outcome tracking
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)  # Indexed per requirements
    pnl: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)

    # Broker details
    broker_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    slippage: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    capital_events: Mapped[list["CapitalEvent"]] = relationship(
        "CapitalEvent",
        back_populates="trade",
        lazy="selectin",  # Eager load to avoid N+1 queries
    )


class CapitalEvent(Base):
    """
    Track all capital movements in the account.

    WHAT IT IS:
        A record of every deposit, withdrawal, profit, loss, dividend, or fee.
        Critical for calculating true performance (alpha) vs just deposits.

    WHY WE NEED IT:
        - Without this, can't tell if portfolio growth is from trading or deposits
        - Needed for accurate Sharpe ratio, drawdown calculations
        - Audit trail for all money in/out

    EXAMPLE:
        User deposits £500 -> CapitalEvent(type=DEPOSIT, amount=500)
        Trade closes +£50  -> CapitalEvent(type=REALIZED_PROFIT, amount=50)
    """
    __tablename__ = "capital_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String(30), index=True)
    amount: Mapped[float] = mapped_column(Numeric(12, 2))
    balance_after: Mapped[float] = mapped_column(Numeric(12, 2))  # Portfolio value after event
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    trade_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trades.id", ondelete="SET NULL"),  # Keep capital event even if trade deleted
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    trade: Mapped["Trade | None"] = relationship(
        "Trade",
        back_populates="capital_events",
        lazy="joined",  # Eager load since we usually want trade details
    )


class Strategy(Base):
    """
    Track multiple trading strategies.

    WHAT IT IS:
        Configuration and performance tracking for different strategies.
        Allows running momentum, mean reversion, etc. with separate allocations.

    WHY WE NEED IT:
        - Different strategies have different risk profiles
        - Track which strategies are working
        - Allocate capital per strategy
        - Disable underperforming strategies independently

    EXAMPLE:
        Strategy(name="momentum", allocation=0.6, capital=300, sharpe_30d=1.2)
        Strategy(name="mean_reversion", allocation=0.4, capital=200, sharpe_30d=0.8)
    """
    __tablename__ = "strategies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default=StrategyStatus.ACTIVE.value)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Allocation
    allocation: Mapped[float] = mapped_column(Numeric(5, 4), default=1.0)  # % of portfolio (0.0-1.0)
    capital: Mapped[float] = mapped_column(Numeric(12, 2), default=0)      # Current capital allocated

    # Performance metrics (updated periodically)
    sharpe_30d: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    win_rate_30d: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)  # 0.0-1.0
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl: Mapped[float] = mapped_column(Numeric(12, 2), default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
