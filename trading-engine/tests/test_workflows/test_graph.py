"""
Tests for the LangGraph trading workflow.

These tests verify:
1. Graph compiles correctly
2. Nodes execute in order
3. State flows through the pipeline
4. Error handling works
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.workflows.graph import build_trading_graph, trading_graph
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


class TestGraphCompilation:
    """Tests for graph structure and compilation."""

    def test_trading_graph_is_compiled(self):
        """The pre-built trading_graph should be a compiled graph."""
        assert trading_graph is not None
        # CompiledStateGraph has an invoke method
        assert hasattr(trading_graph, "invoke")
        assert hasattr(trading_graph, "ainvoke")

    def test_build_trading_graph_returns_compiled(self):
        """build_trading_graph() should return a new compiled graph."""
        graph = build_trading_graph()
        assert graph is not None
        assert hasattr(graph, "ainvoke")


class TestGraphExecution:
    """Tests for graph execution with mocked agents."""

    @pytest.mark.asyncio
    async def test_empty_symbols_returns_empty_state(self):
        """Graph with no symbols should return state with empty outputs."""
        # Patch all agents to return state unchanged
        with patch("src.workflows.graph._data_agent") as mock_data, \
             patch("src.workflows.graph._risk_manager") as mock_risk, \
             patch("src.workflows.graph._validator") as mock_validator, \
             patch("src.workflows.graph._meta_agent") as mock_meta:

            # Each agent just returns state unchanged
            mock_data.run = AsyncMock(side_effect=lambda s: s)
            mock_risk.run = AsyncMock(side_effect=lambda s: s)
            mock_validator.run = AsyncMock(side_effect=lambda s: s)
            mock_meta.run = AsyncMock(side_effect=lambda s: s)

            state = TradingState(symbols=[])
            result = await trading_graph.ainvoke(state)

            # All agents should be called even with empty symbols
            mock_data.run.assert_called_once()
            mock_risk.run.assert_called_once()
            mock_validator.run.assert_called_once()
            mock_meta.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_flows_through_pipeline(self):
        """State should accumulate outputs as it flows through agents."""

        # Create signal that DataAgent will "produce"
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.8,
            proposed_quantity=5,
            price=150.0,
        )

        # Create assessment that RiskManager will "produce"
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
            risk_score=0.2,
        )

        # Create validation that Validator will "produce"
        validation = Validation(
            signal_id=signal.id,
            approved=True,
        )

        # Create decision that MetaAgent will "produce"
        decision = FinalDecision(
            signal_id=signal.id,
            decision=Decision.EXECUTE,
            final_quantity=5,
            confidence=0.85,
        )

        # Mock each agent to add its output
        async def mock_data_run(state):
            state.signals.append(signal)
            return state

        async def mock_risk_run(state):
            state.risk_assessments.append(assessment)
            return state

        async def mock_validator_run(state):
            state.validations.append(validation)
            return state

        async def mock_meta_run(state):
            state.final_decisions.append(decision)
            return state

        with patch("src.workflows.graph._data_agent") as mock_data, \
             patch("src.workflows.graph._risk_manager") as mock_risk, \
             patch("src.workflows.graph._validator") as mock_validator, \
             patch("src.workflows.graph._meta_agent") as mock_meta:

            mock_data.run = AsyncMock(side_effect=mock_data_run)
            mock_risk.run = AsyncMock(side_effect=mock_risk_run)
            mock_validator.run = AsyncMock(side_effect=mock_validator_run)
            mock_meta.run = AsyncMock(side_effect=mock_meta_run)

            initial_state = TradingState(
                symbols=["AAPL"],
                portfolio_snapshot=PortfolioSnapshot(
                    cash=1000.0,
                    total_value=1000.0,
                ),
            )

            result = await trading_graph.ainvoke(initial_state)

            # LangGraph returns a dict - verify all outputs accumulated
            assert len(result["signals"]) == 1
            assert result["signals"][0].symbol == "AAPL"

            assert len(result["risk_assessments"]) == 1
            assert result["risk_assessments"][0].approved is True

            assert len(result["validations"]) == 1
            assert result["validations"][0].approved is True

            assert len(result["final_decisions"]) == 1
            assert result["final_decisions"][0].decision == Decision.EXECUTE

    @pytest.mark.asyncio
    async def test_agent_error_adds_to_state_errors(self):
        """Agent errors should be recorded in state.errors."""

        async def mock_data_with_error(state):
            state.add_error("DataAgent", "API timeout", {"symbol": "AAPL"})
            return state

        with patch("src.workflows.graph._data_agent") as mock_data, \
             patch("src.workflows.graph._risk_manager") as mock_risk, \
             patch("src.workflows.graph._validator") as mock_validator, \
             patch("src.workflows.graph._meta_agent") as mock_meta:

            mock_data.run = AsyncMock(side_effect=mock_data_with_error)
            mock_risk.run = AsyncMock(side_effect=lambda s: s)
            mock_validator.run = AsyncMock(side_effect=lambda s: s)
            mock_meta.run = AsyncMock(side_effect=lambda s: s)

            state = TradingState(symbols=["AAPL"])
            result = await trading_graph.ainvoke(state)

            # LangGraph returns a dict - error should be recorded
            assert len(result["errors"]) == 1
            assert result["errors"][0]["agent"] == "DataAgent"
            assert "timeout" in result["errors"][0]["error"]


class TestGraphWithRunner:
    """Tests for TradingCycleRunner integration."""

    @pytest.mark.asyncio
    async def test_runner_scheduled_cycle(self):
        """Runner should create state and invoke graph."""
        from src.workflows.runner import TradingCycleRunner

        with patch("src.workflows.graph._data_agent") as mock_data, \
             patch("src.workflows.graph._risk_manager") as mock_risk, \
             patch("src.workflows.graph._validator") as mock_validator, \
             patch("src.workflows.graph._meta_agent") as mock_meta:

            mock_data.run = AsyncMock(side_effect=lambda s: s)
            mock_risk.run = AsyncMock(side_effect=lambda s: s)
            mock_validator.run = AsyncMock(side_effect=lambda s: s)
            mock_meta.run = AsyncMock(side_effect=lambda s: s)

            runner = TradingCycleRunner(db_session=None)
            result = await runner.run_scheduled_cycle(["AAPL", "MSFT"])

            assert result.cycle_type == "scheduled"
            assert result.symbols == ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    async def test_runner_event_cycle(self):
        """Runner event cycle should set trigger_symbol."""
        from src.workflows.runner import TradingCycleRunner

        with patch("src.workflows.graph._data_agent") as mock_data, \
             patch("src.workflows.graph._risk_manager") as mock_risk, \
             patch("src.workflows.graph._validator") as mock_validator, \
             patch("src.workflows.graph._meta_agent") as mock_meta:

            mock_data.run = AsyncMock(side_effect=lambda s: s)
            mock_risk.run = AsyncMock(side_effect=lambda s: s)
            mock_validator.run = AsyncMock(side_effect=lambda s: s)
            mock_meta.run = AsyncMock(side_effect=lambda s: s)

            runner = TradingCycleRunner(db_session=None)
            result = await runner.run_event_cycle("TSLA")

            assert result.cycle_type == "event"
            assert result.trigger_symbol == "TSLA"
            assert result.symbols == ["TSLA"]
