"""
Tests for Risk Manager agent.

These tests focus on the hard constraint logic which runs in Python code
(not LLM), ensuring that safety limits cannot be bypassed.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.risk_manager import RiskManager
from src.workflows.state import (
    PortfolioSnapshot,
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
)


class TestHardConstraints:
    """Tests for hard constraint checks (no LLM calls)."""

    def setup_method(self):
        """Create a RiskManager instance for each test."""
        self.risk_manager = RiskManager()

    def test_hold_signal_auto_approved(self):
        """HOLD signals should be auto-approved without checks."""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.HOLD,
            price=150.0,
            proposed_quantity=0,
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(cash=1000.0, total_value=5000.0)
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        assert result is not None
        assert result.approved is True
        assert "HOLD" in result.reasoning

    def test_insufficient_cash_rejected(self):
        """BUY signal should be rejected if insufficient cash."""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=150.0,
            proposed_quantity=10,  # $1500 required
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=1000.0,  # Only $1000 available
                total_value=5000.0,
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        assert result is not None
        assert result.approved is False
        assert result.hard_constraint_violated is True
        assert "Insufficient cash" in result.hard_constraint_reason

    def test_sufficient_cash_passes(self):
        """BUY signal should pass if sufficient cash."""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=100.0,
            proposed_quantity=5,  # $500 required
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=1000.0,  # $1000 available
                total_value=5000.0,
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        # None means passed hard checks
        assert result is None

    def test_max_position_size_rejected(self):
        """BUY should be rejected if position would exceed 20% of portfolio."""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=100.0,
            proposed_quantity=15,  # $1500, which is 30% of $5000
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=2000.0,
                total_value=5000.0,
                positions={},  # No existing position
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        assert result is not None
        assert result.approved is False
        assert result.hard_constraint_violated is True
        assert "20%" in result.hard_constraint_reason

    def test_max_position_includes_existing(self):
        """Position size check should include existing position value."""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=100.0,
            proposed_quantity=5,  # $500 new
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=2000.0,
                total_value=5000.0,
                positions={
                    "AAPL": {"quantity": 6, "value": 600.0}  # Existing $600
                },
                # Total would be $1100 = 22% > 20%
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        assert result is not None
        assert result.approved is False
        assert result.hard_constraint_violated is True

    def test_max_positions_count_rejected(self):
        """BUY should be rejected if already at max positions."""
        signal = Signal(
            symbol="NEW_SYMBOL",
            action=SignalAction.BUY,
            price=10.0,
            proposed_quantity=1,
        )
        # Create 10 existing positions (the max)
        positions = {f"SYM{i}": {"quantity": 1, "value": 100.0} for i in range(10)}

        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=1000.0,
                total_value=5000.0,
                positions=positions,
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        assert result is not None
        assert result.approved is False
        assert result.hard_constraint_violated is True
        assert "max positions" in result.hard_constraint_reason.lower()

    def test_adding_to_existing_position_allowed(self):
        """Adding to existing position doesn't count against max positions."""
        signal = Signal(
            symbol="SYM0",  # Already in portfolio
            action=SignalAction.BUY,
            price=10.0,
            proposed_quantity=1,
        )
        positions = {f"SYM{i}": {"quantity": 1, "value": 100.0} for i in range(10)}

        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=1000.0,
                total_value=5000.0,
                positions=positions,
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        # Should pass (None) because we're not adding a new position
        assert result is None

    def test_sector_exposure_rejected(self):
        """BUY should be rejected if sector would exceed 30%."""
        # Use values where position size is OK but sector exposure fails
        # Portfolio: $10,000
        # Position limit (20%): $2,000
        # Sector limit (30%): $3,000
        # New position: $500 (within position limit)
        # Existing sector: $2,600 + $500 = $3,100 > $3,000 (exceeds sector limit)
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=100.0,
            proposed_quantity=5,  # $500 new
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=3000.0,
                total_value=10000.0,
                positions={
                    "AAPL": {"quantity": 10, "value": 1000.0, "sector": "Technology"},
                    "MSFT": {"quantity": 16, "value": 1600.0, "sector": "Technology"},
                },
                sector_exposure={
                    "Technology": 2600.0  # 26%, adding $500 = 31% > 30%
                },
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        assert result is not None
        assert result.approved is False
        assert result.hard_constraint_violated is True
        assert "30%" in result.hard_constraint_reason

    def test_sell_signal_passes_cash_check(self):
        """SELL signals don't require cash."""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.SELL,
            price=150.0,
            proposed_quantity=10,
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(
                cash=0.0,  # No cash
                total_value=5000.0,
                positions={"AAPL": {"quantity": 10, "value": 1500.0}},
            )
        )

        result = self.risk_manager._check_hard_constraints(signal, state)

        # SELL doesn't need cash check, should pass
        assert result is None


class TestAssessmentParsing:
    """Tests for parsing LLM responses into RiskAssessment objects."""

    def setup_method(self):
        self.risk_manager = RiskManager()

    def test_parse_valid_json(self):
        """Should parse valid JSON response."""
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        response = f'''```json
[
    {{
        "signal_id": "{signal.id}",
        "approved": true,
        "adjusted_quantity": 5,
        "risk_score": 0.3,
        "concerns": ["Minor volatility"],
        "reasoning": "Low risk trade"
    }}
]
```'''

        assessments = self.risk_manager._parse_assessments(response, [signal])

        assert len(assessments) == 1
        assert assessments[0].approved is True
        assert assessments[0].adjusted_quantity == 5
        assert assessments[0].risk_score == 0.3
        assert "Minor volatility" in assessments[0].concerns

    def test_parse_json_without_markdown(self):
        """Should parse JSON even without markdown code blocks."""
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        response = f'''[{{"signal_id": "{signal.id}", "approved": false, "adjusted_quantity": 0, "risk_score": 0.9, "concerns": [], "reasoning": "Too risky"}}]'''

        assessments = self.risk_manager._parse_assessments(response, [signal])

        assert len(assessments) == 1
        assert assessments[0].approved is False

    def test_parse_ignores_unknown_signal_ids(self):
        """Should skip assessments for unknown signal IDs."""
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        response = f'''```json
[
    {{"signal_id": "unknown-id-123", "approved": true, "adjusted_quantity": 5, "risk_score": 0.1, "concerns": [], "reasoning": "OK"}}
]
```'''

        assessments = self.risk_manager._parse_assessments(response, [signal])

        assert len(assessments) == 0

    def test_parse_invalid_json_raises(self):
        """Should raise ValueError for invalid JSON."""
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        response = "This is not JSON at all"

        with pytest.raises(ValueError) as exc_info:
            self.risk_manager._parse_assessments(response, [signal])

        assert "Failed to parse" in str(exc_info.value)


class TestFullFlow:
    """Tests for the full run() method with mocked LLM."""

    @pytest.mark.asyncio
    async def test_hard_constraint_rejection_skips_llm(self):
        """Signals failing hard constraints should not trigger LLM call."""
        risk_manager = RiskManager()

        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=100.0,
            proposed_quantity=100,  # Way too much
        )
        state = TradingState(
            signals=[signal],
            portfolio_snapshot=PortfolioSnapshot(
                cash=100.0,  # Only $100
                total_value=1000.0,
            ),
        )

        # Mock _call_llm to verify it's not called
        risk_manager._call_llm = AsyncMock()

        result = await risk_manager.run(state)

        # LLM should not be called for hard constraint failures
        risk_manager._call_llm.assert_not_called()

        # Should have one rejection
        assert len(result.risk_assessments) == 1
        assert result.risk_assessments[0].approved is False
        assert result.risk_assessments[0].hard_constraint_violated is True

    @pytest.mark.asyncio
    async def test_passing_signals_sent_to_llm(self):
        """Signals passing hard constraints should be sent to LLM."""
        risk_manager = RiskManager()

        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=10.0,
            proposed_quantity=5,  # $50, well within limits
        )
        state = TradingState(
            signals=[signal],
            portfolio_snapshot=PortfolioSnapshot(
                cash=1000.0,
                total_value=5000.0,
            ),
        )

        # Mock LLM response
        mock_response = f'''```json
[{{"signal_id": "{signal.id}", "approved": true, "adjusted_quantity": 5, "risk_score": 0.2, "concerns": [], "reasoning": "Looks good"}}]
```'''
        risk_manager._call_llm = AsyncMock(return_value=mock_response)

        result = await risk_manager.run(state)

        # LLM should be called
        risk_manager._call_llm.assert_called_once()

        # Should have one approval
        assert len(result.risk_assessments) == 1
        assert result.risk_assessments[0].approved is True

    @pytest.mark.asyncio
    async def test_llm_error_rejects_signals(self):
        """LLM errors should result in rejections with error logged."""
        risk_manager = RiskManager()

        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=10.0,
            proposed_quantity=5,
        )
        state = TradingState(
            signals=[signal],
            portfolio_snapshot=PortfolioSnapshot(
                cash=1000.0,
                total_value=5000.0,
            ),
        )

        # Mock LLM to raise error
        risk_manager._call_llm = AsyncMock(side_effect=Exception("API timeout"))

        result = await risk_manager.run(state)

        # Should have rejection and error logged
        assert len(result.risk_assessments) == 1
        assert result.risk_assessments[0].approved is False
        assert len(result.errors) == 1
        assert "timeout" in result.errors[0]["error"]
