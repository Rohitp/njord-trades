"""Tests for workflow state and message types."""

import pytest
from uuid import uuid4

from src.workflows.state import (
    Decision,
    ExecutionResult,
    FinalDecision,
    PortfolioSnapshot,
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
    Validation,
)


class TestSignal:
    """Tests for Signal dataclass."""

    def test_default_signal(self):
        """Signal should have sensible defaults."""
        signal = Signal()
        assert signal.action == SignalAction.HOLD
        assert signal.confidence == 0.0
        assert signal.proposed_quantity == 0
        assert signal.id is not None

    def test_signal_with_values(self):
        """Signal should accept all fields."""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.85,
            proposed_quantity=10,
            reasoning="Strong momentum",
            price=150.0,
            rsi_14=65.0,
        )
        assert signal.symbol == "AAPL"
        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.85
        assert signal.proposed_quantity == 10
        assert signal.rsi_14 == 65.0

    def test_signal_action_from_string(self):
        """Signal should convert string action to enum."""
        signal = Signal(action="BUY")
        assert signal.action == SignalAction.BUY


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""

    def test_default_risk_assessment(self):
        """RiskAssessment should default to not approved."""
        ra = RiskAssessment()
        assert ra.approved is False
        assert ra.adjusted_quantity == 0
        assert ra.concerns == []

    def test_approved_with_adjustment(self):
        """RiskAssessment can approve with reduced quantity."""
        signal_id = uuid4()
        ra = RiskAssessment(
            signal_id=signal_id,
            approved=True,
            adjusted_quantity=5,  # Reduced from proposed
            risk_score=0.3,
            concerns=["High volatility"],
        )
        assert ra.approved is True
        assert ra.adjusted_quantity == 5
        assert len(ra.concerns) == 1

    def test_hard_constraint_violation(self):
        """RiskAssessment should flag hard constraint violations."""
        ra = RiskAssessment(
            approved=False,
            hard_constraint_violated=True,
            hard_constraint_reason="Insufficient cash",
        )
        assert ra.approved is False
        assert ra.hard_constraint_violated is True


class TestValidation:
    """Tests for Validation dataclass."""

    def test_default_validation(self):
        """Validation should default to not approved."""
        v = Validation()
        assert v.approved is False
        assert v.concerns == []
        assert v.suggestions == []

    def test_validation_with_patterns(self):
        """Validation should track detected patterns."""
        v = Validation(
            approved=False,
            concerns=["Too many tech trades"],
            repetition_detected=True,
            sector_clustering_detected=True,
            similar_setup_failures=3,
        )
        assert v.repetition_detected is True
        assert v.sector_clustering_detected is True
        assert v.similar_setup_failures == 3


class TestFinalDecision:
    """Tests for FinalDecision dataclass."""

    def test_default_decision(self):
        """FinalDecision should default to DO_NOT_EXECUTE."""
        fd = FinalDecision()
        assert fd.decision == Decision.DO_NOT_EXECUTE
        assert fd.final_quantity == 0

    def test_execute_decision(self):
        """FinalDecision can approve execution."""
        fd = FinalDecision(
            decision=Decision.EXECUTE,
            final_quantity=10,
            confidence=0.9,
            reasoning="All agents agree",
        )
        assert fd.decision == Decision.EXECUTE
        assert fd.final_quantity == 10

    def test_decision_from_string(self):
        """FinalDecision should convert string to enum."""
        fd = FinalDecision(decision="EXECUTE")
        assert fd.decision == Decision.EXECUTE


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_default_execution_result(self):
        """ExecutionResult should default to not successful."""
        er = ExecutionResult()
        assert er.success is False
        assert er.trade_id is None

    def test_successful_execution(self):
        """ExecutionResult can record successful trade."""
        trade_id = uuid4()
        er = ExecutionResult(
            success=True,
            trade_id=trade_id,
            symbol="AAPL",
            action="BUY",
            quantity=10,
            requested_price=150.0,
            fill_price=150.05,
            slippage=0.05,
            broker_order_id="ABC123",
        )
        assert er.success is True
        assert er.fill_price == 150.05
        assert er.slippage == 0.05


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot dataclass."""

    def test_default_snapshot(self):
        """PortfolioSnapshot should have zero values by default."""
        ps = PortfolioSnapshot()
        assert ps.cash == 0.0
        assert ps.total_value == 0.0
        assert ps.positions == {}

    def test_snapshot_with_positions(self):
        """PortfolioSnapshot should store position details."""
        ps = PortfolioSnapshot(
            cash=1000.0,
            total_value=5000.0,
            positions={
                "AAPL": {"quantity": 10, "value": 1500.0, "sector": "Technology"},
                "MSFT": {"quantity": 20, "value": 2500.0, "sector": "Technology"},
            },
            sector_exposure={"Technology": 4000.0},
        )
        assert ps.cash == 1000.0
        assert len(ps.positions) == 2
        assert ps.sector_exposure["Technology"] == 4000.0


class TestTradingState:
    """Tests for TradingState dataclass."""

    def test_default_state(self):
        """TradingState should initialize with empty lists."""
        state = TradingState()
        assert state.signals == []
        assert state.risk_assessments == []
        assert state.validations == []
        assert state.final_decisions == []
        assert state.execution_results == []
        assert state.errors == []
        assert state.cycle_type == "scheduled"

    def test_event_triggered_state(self):
        """TradingState can be event-triggered."""
        state = TradingState(
            cycle_type="event",
            trigger_symbol="TSLA",
            symbols=["TSLA"],
        )
        assert state.cycle_type == "event"
        assert state.trigger_symbol == "TSLA"

    def test_get_signal(self):
        """TradingState.get_signal should find signal by ID."""
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        state = TradingState(signals=[signal])

        found = state.get_signal(signal.id)
        assert found is not None
        assert found.symbol == "AAPL"

        not_found = state.get_signal(uuid4())
        assert not_found is None

    def test_get_approved_signals(self):
        """TradingState.get_approved_signals filters by risk approval."""
        signal1 = Signal(symbol="AAPL", action=SignalAction.BUY)
        signal2 = Signal(symbol="MSFT", action=SignalAction.BUY)
        signal3 = Signal(symbol="GOOGL", action=SignalAction.BUY)

        state = TradingState(
            signals=[signal1, signal2, signal3],
            risk_assessments=[
                RiskAssessment(signal_id=signal1.id, approved=True),
                RiskAssessment(signal_id=signal2.id, approved=False),
                RiskAssessment(signal_id=signal3.id, approved=True),
            ],
        )

        approved = state.get_approved_signals()
        assert len(approved) == 2
        symbols = {s.symbol for s in approved}
        assert symbols == {"AAPL", "GOOGL"}

    def test_get_validated_signals(self):
        """TradingState.get_validated_signals filters by validation."""
        signal1 = Signal(symbol="AAPL", action=SignalAction.BUY)
        signal2 = Signal(symbol="MSFT", action=SignalAction.BUY)

        state = TradingState(
            signals=[signal1, signal2],
            validations=[
                Validation(signal_id=signal1.id, approved=True),
                Validation(signal_id=signal2.id, approved=False),
            ],
        )

        validated = state.get_validated_signals()
        assert len(validated) == 1
        assert validated[0].symbol == "AAPL"

    def test_get_execute_decisions(self):
        """TradingState.get_execute_decisions filters by decision."""
        signal1 = Signal(symbol="AAPL")
        signal2 = Signal(symbol="MSFT")

        state = TradingState(
            signals=[signal1, signal2],
            final_decisions=[
                FinalDecision(signal_id=signal1.id, decision=Decision.EXECUTE, final_quantity=10),
                FinalDecision(signal_id=signal2.id, decision=Decision.DO_NOT_EXECUTE),
            ],
        )

        execute = state.get_execute_decisions()
        assert len(execute) == 1
        assert execute[0].final_quantity == 10

    def test_add_error(self):
        """TradingState.add_error should record errors."""
        state = TradingState()
        state.add_error("DataAgent", "API timeout", {"symbol": "AAPL"})

        assert len(state.errors) == 1
        assert state.errors[0]["agent"] == "DataAgent"
        assert state.errors[0]["error"] == "API timeout"
        assert state.errors[0]["details"]["symbol"] == "AAPL"
        assert "timestamp" in state.errors[0]
