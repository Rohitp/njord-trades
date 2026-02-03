"""
Trading workflow state and message types.

Defines the data structures that flow through the trading pipeline:
    Market Data → Signal → Risk Assessment → Validation → Decision → Execution

Each agent receives input and produces output in a standardized format.

All dataclasses include from_dict() classmethods for deserialization from
LangGraph's dict output, keeping serialization logic co-located with schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Self
from uuid import UUID, uuid4


# =============================================================================
# HELPERS
# =============================================================================


def _parse_uuid(value: str | UUID | None) -> UUID | None:
    """Convert string to UUID if needed."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    return UUID(value)


def _parse_datetime(value: str | datetime | None) -> datetime:
    """Convert ISO string to datetime if needed."""
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from dict (e.g., LangGraph output)."""
        return cls(
            id=_parse_uuid(data.get("id")) or uuid4(),
            symbol=data.get("symbol", ""),
            action=SignalAction(data["action"]) if isinstance(data.get("action"), str) else data.get("action", SignalAction.HOLD),
            confidence=data.get("confidence", 0.0),
            proposed_quantity=data.get("proposed_quantity", 0),
            reasoning=data.get("reasoning", ""),
            price=data.get("price", 0.0),
            sma_20=data.get("sma_20"),
            sma_50=data.get("sma_50"),
            sma_200=data.get("sma_200"),
            rsi_14=data.get("rsi_14"),
            volume_ratio=data.get("volume_ratio"),
            timestamp=_parse_datetime(data.get("timestamp")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from dict (e.g., LangGraph output)."""
        return cls(
            signal_id=_parse_uuid(data.get("signal_id")) or uuid4(),
            approved=data.get("approved", False),
            adjusted_quantity=data.get("adjusted_quantity", 0),
            risk_score=data.get("risk_score", 0.0),
            hard_constraint_violated=data.get("hard_constraint_violated", False),
            hard_constraint_reason=data.get("hard_constraint_reason"),
            concerns=data.get("concerns", []),
            reasoning=data.get("reasoning", ""),
            timestamp=_parse_datetime(data.get("timestamp")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from dict (e.g., LangGraph output)."""
        return cls(
            signal_id=_parse_uuid(data.get("signal_id")) or uuid4(),
            approved=data.get("approved", False),
            concerns=data.get("concerns", []),
            suggestions=data.get("suggestions", []),
            reasoning=data.get("reasoning", ""),
            repetition_detected=data.get("repetition_detected", False),
            sector_clustering_detected=data.get("sector_clustering_detected", False),
            similar_setup_failures=data.get("similar_setup_failures", 0),
            timestamp=_parse_datetime(data.get("timestamp")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from dict (e.g., LangGraph output)."""
        decision = data.get("decision", Decision.DO_NOT_EXECUTE)
        if isinstance(decision, str):
            decision = Decision(decision)
        return cls(
            signal_id=_parse_uuid(data.get("signal_id")) or uuid4(),
            decision=decision,
            final_quantity=data.get("final_quantity", 0),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            timestamp=_parse_datetime(data.get("timestamp")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from dict (e.g., LangGraph output)."""
        return cls(
            signal_id=_parse_uuid(data.get("signal_id")) or uuid4(),
            success=data.get("success", False),
            trade_id=_parse_uuid(data.get("trade_id")),
            symbol=data.get("symbol", ""),
            action=data.get("action", ""),
            quantity=data.get("quantity", 0),
            requested_price=data.get("requested_price", 0.0),
            fill_price=data.get("fill_price"),
            slippage=data.get("slippage"),
            broker_order_id=data.get("broker_order_id"),
            error=data.get("error"),
            timestamp=_parse_datetime(data.get("timestamp")),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from dict (e.g., LangGraph output)."""
        return cls(
            cash=data.get("cash", 0.0),
            total_value=data.get("total_value", 0.0),
            deployed_capital=data.get("deployed_capital", 0.0),
            peak_value=data.get("peak_value", 0.0),
            positions=data.get("positions", {}),
            sector_exposure=data.get("sector_exposure", {}),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Deserialize from dict (e.g., LangGraph output).

        LangGraph serializes dataclasses to dicts, so this method reconstructs
        the full TradingState with all nested objects properly typed.

        Args:
            data: Dictionary from LangGraph output

        Returns:
            Fully reconstructed TradingState

        Raises:
            ValueError: If required fields are missing
        """
        # Validate required fields
        required_fields = ["cycle_id", "cycle_type", "symbols"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Reconstruct nested objects using their from_dict methods
        signals = [
            Signal.from_dict(s) if isinstance(s, dict) else s
            for s in data.get("signals", [])
        ]

        risk_assessments = [
            RiskAssessment.from_dict(ra) if isinstance(ra, dict) else ra
            for ra in data.get("risk_assessments", [])
        ]

        validations = [
            Validation.from_dict(v) if isinstance(v, dict) else v
            for v in data.get("validations", [])
        ]

        final_decisions = [
            FinalDecision.from_dict(fd) if isinstance(fd, dict) else fd
            for fd in data.get("final_decisions", [])
        ]

        execution_results = [
            ExecutionResult.from_dict(er) if isinstance(er, dict) else er
            for er in data.get("execution_results", [])
        ]

        portfolio_data = data.get("portfolio_snapshot", {})
        if isinstance(portfolio_data, dict):
            portfolio = PortfolioSnapshot.from_dict(portfolio_data)
        elif isinstance(portfolio_data, PortfolioSnapshot):
            portfolio = portfolio_data
        else:
            portfolio = PortfolioSnapshot()

        return cls(
            cycle_id=_parse_uuid(data.get("cycle_id")) or uuid4(),
            cycle_type=data.get("cycle_type", "scheduled"),
            trigger_symbol=data.get("trigger_symbol"),
            started_at=_parse_datetime(data.get("started_at")),
            trace_id=data.get("trace_id"),
            symbols=data.get("symbols", []),
            portfolio_snapshot=portfolio,
            signals=signals,
            risk_assessments=risk_assessments,
            validations=validations,
            final_decisions=final_decisions,
            execution_results=execution_results,
            errors=data.get("errors", []),
        )
