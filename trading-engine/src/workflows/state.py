"""
Trading workflow state and message types.

Defines the data structures that flow through the trading pipeline:
    Market Data → Signal → Risk Assessment → Validation → Decision → Execution

Each agent receives input and produces output in a standardized format.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


# =============================================================================
# ENUMS
# =============================================================================


class SignalAction(str, Enum):
    """Possible actions from the Data Agent."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Decision(str, Enum):
    """Final decision from Meta Agent."""
    EXECUTE = "EXECUTE"
    DO_NOT_EXECUTE = "DO_NOT_EXECUTE"


# =============================================================================
# AGENT INPUT/OUTPUT TYPES
# =============================================================================


@dataclass
class Signal:
    """
    Output from Data Agent.

    Represents a trading signal with confidence and reasoning.
    """
    id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    action: SignalAction = SignalAction.HOLD
    confidence: float = 0.0  # 0.0 - 1.0
    proposed_quantity: int = 0
    reasoning: str = ""

    # Technical indicators that informed the signal
    price: float = 0.0
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    volume_ratio: float | None = None

    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if isinstance(self.action, str):
            self.action = SignalAction(self.action)


@dataclass
class RiskAssessment:
    """
    Output from Risk Manager.

    Evaluates a signal against portfolio constraints.
    """
    signal_id: UUID = field(default_factory=uuid4)
    approved: bool = False
    adjusted_quantity: int = 0  # May be reduced from signal's proposal
    risk_score: float = 0.0  # 0.0 - 1.0 (higher = riskier)

    # Hard constraint violations (cannot be overridden)
    hard_constraint_violated: bool = False
    hard_constraint_reason: str | None = None

    # Soft concerns (can be debated)
    concerns: list[str] = field(default_factory=list)
    reasoning: str = ""

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Validation:
    """
    Output from Validator.

    Pattern recognition and quality control checks.
    """
    signal_id: UUID = field(default_factory=uuid4)
    approved: bool = False
    concerns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    reasoning: str = ""

    # Pattern detection results
    repetition_detected: bool = False
    sector_clustering_detected: bool = False
    similar_setup_failures: int = 0

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FinalDecision:
    """
    Output from Meta Agent.

    Final synthesis of all agent perspectives.
    """
    signal_id: UUID = field(default_factory=uuid4)
    decision: Decision = Decision.DO_NOT_EXECUTE
    final_quantity: int = 0
    confidence: float = 0.0  # 0.0 - 1.0
    reasoning: str = ""

    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if isinstance(self.decision, str):
            self.decision = Decision(self.decision)


@dataclass
class ExecutionResult:
    """
    Result of trade execution.

    Contains broker response and actual fill details.
    """
    signal_id: UUID = field(default_factory=uuid4)
    success: bool = False
    trade_id: UUID | None = None

    # Execution details
    symbol: str = ""
    action: str = ""
    quantity: int = 0
    requested_price: float = 0.0
    fill_price: float | None = None
    slippage: float | None = None

    # Broker info
    broker_order_id: str | None = None
    error: str | None = None

    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# WORKFLOW STATE
# =============================================================================


@dataclass
class PortfolioSnapshot:
    """
    Point-in-time snapshot of portfolio state.

    Captured at the start of each trading cycle.
    """
    cash: float = 0.0
    total_value: float = 0.0
    deployed_capital: float = 0.0
    peak_value: float = 0.0

    # Position details: symbol -> {quantity, value, sector}
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Sector exposure: sector -> total value
    sector_exposure: dict[str, float] = field(default_factory=dict)


@dataclass
class TradingState:
    """
    Complete state for a trading cycle.

    This is the state object that flows through the LangGraph workflow.
    Each node reads from and writes to this state.

    WORKFLOW FLOW:
        1. Initialize with market_data and portfolio_snapshot
        2. Data Agent adds to signals[]
        3. Risk Manager adds to risk_assessments[]
        4. Validator adds to validations[]
        5. Meta Agent adds to final_decisions[]
        6. Executor adds to execution_results[]
    """
    # Cycle metadata
    cycle_id: UUID = field(default_factory=uuid4)
    cycle_type: str = "scheduled"  # "scheduled" or "event"
    trigger_symbol: str | None = None  # For event-driven cycles
    started_at: datetime = field(default_factory=datetime.now)

    # Tracing - trace_id flows from HTTP request for distributed tracing
    trace_id: str | None = None

    # Inputs
    symbols: list[str] = field(default_factory=list)  # Symbols to analyze
    portfolio_snapshot: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)

    # Agent outputs (accumulated through workflow)
    signals: list[Signal] = field(default_factory=list)
    risk_assessments: list[RiskAssessment] = field(default_factory=list)
    validations: list[Validation] = field(default_factory=list)
    final_decisions: list[FinalDecision] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)

    # Errors encountered during workflow
    errors: list[dict[str, Any]] = field(default_factory=list)

    def get_signal(self, signal_id: UUID) -> Signal | None:
        """Get a signal by ID."""
        for signal in self.signals:
            if signal.id == signal_id:
                return signal
        return None

    def get_approved_signals(self) -> list[Signal]:
        """Get signals that passed risk assessment."""
        approved_ids = {ra.signal_id for ra in self.risk_assessments if ra.approved}
        return [s for s in self.signals if s.id in approved_ids]

    def get_validated_signals(self) -> list[Signal]:
        """Get signals that passed validation."""
        validated_ids = {v.signal_id for v in self.validations if v.approved}
        return [s for s in self.signals if s.id in validated_ids]

    def get_execute_decisions(self) -> list[FinalDecision]:
        """Get decisions marked for execution."""
        return [d for d in self.final_decisions if d.decision == Decision.EXECUTE]

    def add_error(self, agent: str, error: str, details: dict | None = None) -> None:
        """Record an error that occurred during workflow."""
        self.errors.append({
            "agent": agent,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })
