"""
Tests for Meta Agent.

Tests the final decision synthesis logic.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.meta_agent import MetaAgent
from src.workflows.state import (
    Decision,
    FinalDecision,
    PortfolioSnapshot,
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
    Validation,
)


class TestMetaAgentBasic:
    """Tests for basic MetaAgent functionality."""

    def test_meta_agent_name(self):
        """Test that MetaAgent has correct name."""
        agent = MetaAgent()
        assert agent.name == "MetaAgent"

    @pytest.mark.asyncio
    async def test_no_validated_signals_returns_unchanged(self):
        """Test that empty validated signals returns state unchanged."""
        agent = MetaAgent()
        state = TradingState(symbols=["AAPL"])
        # No validated signals

        result = await agent.run(state)

        assert result.final_decisions == []
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_signals_without_validation_skipped(self):
        """Test that signals without validation are not processed."""
        agent = MetaAgent()
        signal = Signal(symbol="AAPL", action=SignalAction.BUY, confidence=0.8)
        # Signal has risk approval but no validation
        assessment = RiskAssessment(signal_id=signal.id, approved=True, adjusted_quantity=5)

        state = TradingState(
            symbols=["AAPL"],
            signals=[signal],
            risk_assessments=[assessment],
            validations=[],  # No validation
        )

        result = await agent.run(state)

        assert result.final_decisions == []


class TestMetaAgentPromptBuilding:
    """Tests for prompt building."""

    def test_build_user_prompt_basic(self):
        """Test basic prompt building with all agent perspectives."""
        agent = MetaAgent()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.8,
            proposed_quantity=10,
            reasoning="RSI oversold",
            price=150.0,
            rsi_14=28.0,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
            risk_score=0.3,
            concerns=["Market volatility"],
            reasoning="Acceptable risk",
        )
        validation = Validation(
            signal_id=signal.id,
            approved=True,
            concerns=[],
            suggestions=["Consider stop loss"],
            repetition_detected=False,
            sector_clustering_detected=False,
            similar_setup_failures=0,
            reasoning="No patterns detected",
        )
        state = TradingState(
            portfolio_snapshot=PortfolioSnapshot(cash=10000.0, total_value=50000.0)
        )

        prompt = agent._build_user_prompt(
            [signal],
            {signal.id: assessment},
            {signal.id: validation},
            state,
        )

        assert "AAPL" in prompt
        assert "BUY" in prompt
        assert "RSI oversold" in prompt
        assert "0.8" in prompt  # Signal confidence
        assert "DATA AGENT" in prompt
        assert "RISK MANAGER" in prompt
        assert "VALIDATOR" in prompt
        assert "Market volatility" in prompt
        assert "Consider stop loss" in prompt

    def test_build_user_prompt_with_concerns(self):
        """Test prompt building with multiple concerns."""
        agent = MetaAgent()
        signal = Signal(symbol="AAPL", action=SignalAction.BUY, price=150.0)
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=3,
            risk_score=0.6,
            concerns=["High volatility", "Market uncertainty"],
        )
        validation = Validation(
            signal_id=signal.id,
            approved=True,
            concerns=["Third tech trade this week"],
            sector_clustering_detected=True,
            similar_setup_failures=1,
        )
        state = TradingState()

        prompt = agent._build_user_prompt(
            [signal],
            {signal.id: assessment},
            {signal.id: validation},
            state,
        )

        assert "High volatility" in prompt
        assert "Market uncertainty" in prompt
        assert "Third tech trade this week" in prompt
        assert "Sector Clustering: True" in prompt
        assert "Similar Failures: 1" in prompt


class TestMetaAgentDecisionParsing:
    """Tests for decision parsing."""

    def test_parse_decisions_execute(self):
        """Test parsing EXECUTE decision."""
        agent = MetaAgent()
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        response = f'''```json
[
    {{"signal_id": "{signal.id}", "decision": "EXECUTE", "final_quantity": 5, "confidence": 0.85, "reasoning": "All agents agree"}}
]
```'''

        decisions = agent._parse_decisions(response, [signal])

        assert len(decisions) == 1
        assert decisions[0].signal_id == signal.id
        assert decisions[0].decision == Decision.EXECUTE
        assert decisions[0].final_quantity == 5
        assert decisions[0].confidence == 0.85
        assert "All agents agree" in decisions[0].reasoning

    def test_parse_decisions_do_not_execute(self):
        """Test parsing DO_NOT_EXECUTE decision."""
        agent = MetaAgent()
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        response = f'[{{"signal_id": "{signal.id}", "decision": "DO_NOT_EXECUTE", "final_quantity": 0, "confidence": 0.1, "reasoning": "Too risky"}}]'

        decisions = agent._parse_decisions(response, [signal])

        assert len(decisions) == 1
        assert decisions[0].decision == Decision.DO_NOT_EXECUTE
        assert decisions[0].final_quantity == 0

    def test_parse_decisions_multiple(self):
        """Test parsing multiple decisions."""
        agent = MetaAgent()
        signal1 = Signal(symbol="AAPL", action=SignalAction.BUY)
        signal2 = Signal(symbol="MSFT", action=SignalAction.SELL)
        response = f'''[
            {{"signal_id": "{signal1.id}", "decision": "EXECUTE", "final_quantity": 5, "confidence": 0.8, "reasoning": "Good"}},
            {{"signal_id": "{signal2.id}", "decision": "DO_NOT_EXECUTE", "final_quantity": 0, "confidence": 0.3, "reasoning": "Rejected"}}
        ]'''

        decisions = agent._parse_decisions(response, [signal1, signal2])

        assert len(decisions) == 2
        assert decisions[0].signal_id == signal1.id
        assert decisions[0].decision == Decision.EXECUTE
        assert decisions[1].signal_id == signal2.id
        assert decisions[1].decision == Decision.DO_NOT_EXECUTE

    def test_parse_decisions_ignores_unknown_ids(self):
        """Test that unknown signal IDs are ignored."""
        agent = MetaAgent()
        signal = Signal(symbol="AAPL", action=SignalAction.BUY)
        response = f'''[
            {{"signal_id": "{signal.id}", "decision": "EXECUTE", "final_quantity": 5, "confidence": 0.8, "reasoning": "OK"}},
            {{"signal_id": "unknown-id-123", "decision": "EXECUTE", "final_quantity": 10, "confidence": 0.9, "reasoning": "Unknown"}}
        ]'''

        decisions = agent._parse_decisions(response, [signal])

        assert len(decisions) == 1
        assert decisions[0].signal_id == signal.id


class TestMetaAgentFullFlow:
    """Tests for full run flow."""

    @pytest.mark.asyncio
    async def test_run_execute_decision(self):
        """Test full run with EXECUTE decision."""
        agent = MetaAgent()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.8,
            proposed_quantity=10,
            price=150.0,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
            risk_score=0.2,
        )
        validation = Validation(
            signal_id=signal.id,
            approved=True,
        )
        state = TradingState(
            symbols=["AAPL"],
            signals=[signal],
            risk_assessments=[assessment],
            validations=[validation],
            portfolio_snapshot=PortfolioSnapshot(cash=10000.0, total_value=50000.0),
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = f'[{{"signal_id": "{signal.id}", "decision": "EXECUTE", "final_quantity": 5, "confidence": 0.85, "reasoning": "Strong signal"}}]'

            result = await agent.run(state)

            assert len(result.final_decisions) == 1
            assert result.final_decisions[0].decision == Decision.EXECUTE
            assert result.final_decisions[0].final_quantity == 5

    @pytest.mark.asyncio
    async def test_run_reject_decision(self):
        """Test full run with DO_NOT_EXECUTE decision."""
        agent = MetaAgent()
        signal = Signal(symbol="AAPL", action=SignalAction.BUY, price=150.0)
        assessment = RiskAssessment(signal_id=signal.id, approved=True, adjusted_quantity=5)
        validation = Validation(signal_id=signal.id, approved=True, similar_setup_failures=3)
        state = TradingState(
            signals=[signal],
            risk_assessments=[assessment],
            validations=[validation],
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = f'[{{"signal_id": "{signal.id}", "decision": "DO_NOT_EXECUTE", "final_quantity": 0, "confidence": 0.2, "reasoning": "Too many similar failures"}}]'

            result = await agent.run(state)

            assert len(result.final_decisions) == 1
            assert result.final_decisions[0].decision == Decision.DO_NOT_EXECUTE

    @pytest.mark.asyncio
    async def test_run_handles_llm_error(self):
        """Test that LLM errors result in rejection (fail-closed)."""
        agent = MetaAgent()
        signal = Signal(symbol="AAPL", action=SignalAction.BUY, price=150.0)
        assessment = RiskAssessment(signal_id=signal.id, approved=True, adjusted_quantity=5)
        validation = Validation(signal_id=signal.id, approved=True)
        state = TradingState(
            signals=[signal],
            risk_assessments=[assessment],
            validations=[validation],
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM timeout")

            result = await agent.run(state)

            # Should fail-closed: reject on error
            assert len(result.final_decisions) == 1
            assert result.final_decisions[0].decision == Decision.DO_NOT_EXECUTE
            assert result.final_decisions[0].final_quantity == 0
            assert "error" in result.final_decisions[0].reasoning.lower()
            assert len(result.errors) == 1
            assert "LLM timeout" in result.errors[0]["error"]

    @pytest.mark.asyncio
    async def test_run_multiple_signals(self):
        """Test running with multiple validated signals."""
        agent = MetaAgent()
        signal1 = Signal(symbol="AAPL", action=SignalAction.BUY, price=150.0)
        signal2 = Signal(symbol="MSFT", action=SignalAction.SELL, price=300.0)

        state = TradingState(
            signals=[signal1, signal2],
            risk_assessments=[
                RiskAssessment(signal_id=signal1.id, approved=True, adjusted_quantity=5),
                RiskAssessment(signal_id=signal2.id, approved=True, adjusted_quantity=3),
            ],
            validations=[
                Validation(signal_id=signal1.id, approved=True),
                Validation(signal_id=signal2.id, approved=True),
            ],
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = f'''[
                {{"signal_id": "{signal1.id}", "decision": "EXECUTE", "final_quantity": 5, "confidence": 0.8, "reasoning": "OK"}},
                {{"signal_id": "{signal2.id}", "decision": "EXECUTE", "final_quantity": 3, "confidence": 0.7, "reasoning": "OK"}}
            ]'''

            result = await agent.run(state)

            assert len(result.final_decisions) == 2
            decisions_by_symbol = {
                state.get_signal(d.signal_id).symbol: d for d in result.final_decisions
            }
            assert decisions_by_symbol["AAPL"].decision == Decision.EXECUTE
            assert decisions_by_symbol["MSFT"].decision == Decision.EXECUTE

    @pytest.mark.asyncio
    async def test_run_partial_validation(self):
        """Test that only validated signals are processed."""
        agent = MetaAgent()
        signal1 = Signal(symbol="AAPL", action=SignalAction.BUY, price=150.0)
        signal2 = Signal(symbol="MSFT", action=SignalAction.BUY, price=300.0)

        state = TradingState(
            signals=[signal1, signal2],
            risk_assessments=[
                RiskAssessment(signal_id=signal1.id, approved=True, adjusted_quantity=5),
                RiskAssessment(signal_id=signal2.id, approved=True, adjusted_quantity=3),
            ],
            validations=[
                Validation(signal_id=signal1.id, approved=True),
                Validation(signal_id=signal2.id, approved=False),  # Rejected by validator
            ],
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = f'[{{"signal_id": "{signal1.id}", "decision": "EXECUTE", "final_quantity": 5, "confidence": 0.8, "reasoning": "OK"}}]'

            result = await agent.run(state)

            # Only signal1 should have a decision (signal2 was rejected by validator)
            assert len(result.final_decisions) == 1
            assert result.final_decisions[0].signal_id == signal1.id
