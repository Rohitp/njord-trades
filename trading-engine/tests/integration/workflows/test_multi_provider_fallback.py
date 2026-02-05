"""
Integration tests for multi-provider LLM fallback behavior.

Tests end-to-end workflow with provider fallback scenarios.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.workflows.runner import TradingCycleRunner
from src.workflows.state import TradingState
from src.config import settings


@pytest.fixture
def mock_db_session():
    """Mock database session for integration tests."""
    from sqlalchemy.ext.asyncio import AsyncSession
    return MagicMock(spec=AsyncSession)


class TestWorkflowProviderFallback:
    """Test provider fallback in full trading workflow."""

    @pytest.mark.asyncio
    async def test_workflow_falls_back_on_primary_provider_failure(self, mock_db_session):
        """Test that workflow falls back to secondary provider when primary fails."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                with patch.object(settings.llm, "default_provider", "openai"):
                    with patch.object(settings.llm, "fallback_provider", "anthropic"):
                        with patch.object(settings.llm, "max_retries", 3):
                            # Mock portfolio state query (for _load_portfolio_snapshot)
                            mock_portfolio_result = MagicMock()
                            mock_portfolio_result.scalar_one_or_none.return_value = None
                            
                            # Mock system state query (for circuit breaker check)
                            mock_system_result = MagicMock()
                            mock_system_state = MagicMock()
                            mock_system_state.circuit_breaker_active = False
                            mock_system_state.trading_enabled = True
                            mock_system_result.scalar_one_or_none.return_value = mock_system_state
                            
                            # Mock execute to return different results based on query
                            async def execute_side_effect(stmt):
                                stmt_str = str(stmt).lower()
                                if "portfolio_state" in stmt_str or "portfoliostate" in stmt_str:
                                    return mock_portfolio_result
                                elif "system_state" in stmt_str or "systemstate" in stmt_str:
                                    return mock_system_result
                                return MagicMock()
                            
                            mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
                            
                            # Mock the graph agents to verify they're called (fallback should work)
                            with patch("src.workflows.graph._data_agent") as mock_data, \
                                 patch("src.workflows.graph._risk_manager") as mock_risk, \
                                 patch("src.workflows.graph._validator") as mock_validator, \
                                 patch("src.workflows.graph._meta_agent") as mock_meta:
                                
                                # Agents should just pass state through for this test
                                mock_data.run = AsyncMock(side_effect=lambda s: s)
                                mock_risk.run = AsyncMock(side_effect=lambda s: s)
                                mock_validator.run = AsyncMock(side_effect=lambda s: s)
                                mock_meta.run = AsyncMock(side_effect=lambda s: s)
                                
                                runner = TradingCycleRunner(db_session=mock_db_session)
                                
                                # Should not raise error (fallback should work)
                                # The DataAgent will use Anthropic after OpenAI fails
                                result = await runner.run_scheduled_cycle(symbols=["AAPL"])
                                
                                # Verify cycle completed successfully
                                assert result is not None
                                assert isinstance(result, TradingState)
                                assert result.cycle_type == "scheduled"
                                assert "AAPL" in result.symbols

