"""
Integration tests for alert service integration points.

Tests that alerts are actually called when circuit breaker activates,
trades execute, etc.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.alerts.service import AlertService
from src.services.circuit_breaker import CircuitBreakerService


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.scalar = AsyncMock()
    return session


@pytest.mark.asyncio
class TestCircuitBreakerAlertIntegration:
    """Test that circuit breaker actually sends alerts."""

    async def test_circuit_breaker_sends_alert_on_activation(
        self, mock_db_session
    ):
        """Test that circuit breaker sends alert when activated."""
        from src.database.models import PortfolioState, SystemState
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=False,
            circuit_breaker_reason=None,
        )

        # Create portfolio with high drawdown
        portfolio = PortfolioState(
            id=1,
            cash=50000.0,
            total_value=79000.0,  # 21% drawdown (above 20% threshold)
            peak_value=100000.0,
            deployed_capital=29000.0,
        )

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = portfolio

        async def execute_side_effect(stmt):
            stmt_str = str(stmt).lower()
            if "system_state" in stmt_str or "systemstate" in stmt_str:
                return system_result
            elif "portfolio_state" in stmt_str or "portfoliostate" in stmt_str:
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
        mock_db_session.commit = AsyncMock()
        mock_db_session.add = MagicMock()

        # Mock AlertService - patch where it's imported (inside the function)
        with patch("src.services.alerts.service.AlertService") as mock_alert_class:
            mock_alert_service = MagicMock()
            mock_alert_service.send_circuit_breaker_alert = AsyncMock(return_value=True)
            mock_alert_class.return_value = mock_alert_service

            service = CircuitBreakerService(mock_db_session)
            triggered = await service.evaluate_and_update()

            assert triggered is True
            # Verify alert was sent
            mock_alert_service.send_circuit_breaker_alert.assert_called_once()
            call_args = mock_alert_service.send_circuit_breaker_alert.call_args
            assert "drawdown" in call_args[1]["reason"].lower()
            assert call_args[1]["drawdown_pct"] is not None

    async def test_auto_resume_sends_alert_when_conditions_met(
        self, mock_db_session
    ):
        """Test that auto-resume check sends alert when conditions are met."""
        from src.database.models import PortfolioState, SystemState
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="Drawdown 21.0% exceeds 20% threshold",
        )

        # Create portfolio with recovered drawdown
        portfolio = PortfolioState(
            id=1,
            cash=50000.0,
            total_value=90000.0,  # 10% drawdown (below 15% threshold)
            peak_value=100000.0,
            deployed_capital=40000.0,
        )

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = portfolio

        async def execute_side_effect(stmt):
            stmt_str = str(stmt).lower()
            if "system_state" in stmt_str or "systemstate" in stmt_str:
                return system_result
            elif "portfolio_state" in stmt_str or "portfoliostate" in stmt_str:
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        # Mock AlertService - patch where it's imported (inside the function)
        with patch("src.services.alerts.service.AlertService") as mock_alert_class:
            mock_alert_service = MagicMock()
            mock_alert_service.send_auto_resume_conditions_met_alert = AsyncMock(return_value=True)
            mock_alert_class.return_value = mock_alert_service

            service = CircuitBreakerService(mock_db_session)
            result = await service.check_auto_resume()

            assert result is True
            # Verify alert was sent
            mock_alert_service.send_auto_resume_conditions_met_alert.assert_called_once()


@pytest.mark.asyncio
class TestExecutionServiceAlertIntegration:
    """Test that execution service sends alerts on trade execution."""

    async def test_execution_service_sends_alert_on_successful_trade(
        self, mock_db_session
    ):
        """Test that execution service sends position change alert."""
        from src.services.execution.service import ExecutionService
        from src.workflows.state import (
            Decision,
            ExecutionResult,
            FinalDecision,
            Signal,
            SignalAction,
            TradingState,
        )
        from uuid import uuid4
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create a successful execution result
        signal = Signal(
            id=uuid4(),
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.8,
            proposed_quantity=10,
            price=150.0,  # Need price for slippage calculation
        )

        decision = FinalDecision(
            signal_id=signal.id,
            decision=Decision.EXECUTE,
            final_quantity=10,
            confidence=0.75,
        )

        execution_result = ExecutionResult(
            signal_id=signal.id,
            success=True,
            symbol="AAPL",
            action="BUY",
            quantity=10,
            fill_price=150.0,
            broker_order_id="order_123",
        )

        state = TradingState(
            cycle_type="scheduled",
            symbols=["AAPL"],
            signals=[signal],
            final_decisions=[decision],
        )

        # Mock broker - use submit_order (not execute_order)
        from src.services.execution.broker import OrderResult, OrderStatus
        
        mock_broker = MagicMock()
        mock_order_result = OrderResult(
            success=True,
            status=OrderStatus.FILLED,  # This makes is_filled property return True
            filled_price=150.0,
            filled_quantity=10,
            broker_order_id="order_123",
            error_message=None,
        )
        mock_broker.submit_order = AsyncMock(return_value=mock_order_result)
        mock_broker.name = "test_broker"

        # Mock database operations
        mock_db_session.execute = AsyncMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.add = MagicMock()
        mock_db_session.scalar = AsyncMock(return_value=None)  # No existing position

        # Mock AlertService - patch where it's imported (inside the function)
        with patch("src.services.alerts.service.AlertService") as mock_alert_class:
            mock_alert_service = MagicMock()
            mock_alert_service.send_position_change_alert = AsyncMock(return_value=True)
            mock_alert_class.return_value = mock_alert_service

            # Create service
            service = ExecutionService(broker=mock_broker, db_session=mock_db_session)
            
            # Mock _persist_trade to return a trade_id (this is called before the alert)
            with patch.object(service, "_persist_trade", new_callable=AsyncMock) as mock_persist:
                mock_persist.return_value = uuid4()
                
                # Mock _update_position and _update_portfolio (called by _persist_trade)
                with patch.object(service, "_update_position", new_callable=AsyncMock) as mock_update_pos:
                    with patch.object(service, "_update_portfolio", new_callable=AsyncMock) as mock_update_portfolio:
                        # Execute the decision
                        result = await service._execute_single(
                            state, decision, cached_positions=None
                        )

                        # Verify execution succeeded
                        assert result.success is True
                        assert result.symbol == "AAPL"
                        assert result.action == "BUY"
                        assert result.quantity == 10
                        
                        # Verify _persist_trade was called (which triggers the alert)
                        mock_persist.assert_called_once()
                        
                        # Verify alert was sent
                        mock_alert_service.send_position_change_alert.assert_called_once()
                        call_args = mock_alert_service.send_position_change_alert.call_args
                        assert call_args[1]["symbol"] == "AAPL"
                        assert call_args[1]["action"] == "BUY"
                        assert call_args[1]["quantity"] == 10


@pytest.mark.asyncio
class TestAlertRetryLogic:
    """Test retry logic for Telegram alerts."""

    async def test_telegram_retries_on_failure(self):
        """Test that Telegram provider retries on HTTP errors."""
        from src.services.alerts.telegram import TelegramAlertProvider
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = TelegramAlertProvider(bot_token="test_token", chat_id="test_chat_id")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            
            # First call fails, second succeeds
            mock_response_success = MagicMock()
            mock_response_success.json.return_value = {"ok": True}
            mock_response_success.raise_for_status = MagicMock()
            
            # Use ConnectionError which is in the retryable exceptions
            mock_client.post = AsyncMock(
                side_effect=[
                    ConnectionError("Connection error"),
                    mock_response_success,
                ]
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Should retry and eventually succeed
            result = await provider.send_message("Test message")
            
            # Should have been called twice (retry)
            assert mock_client.post.call_count == 2
            assert result is True

    async def test_telegram_fails_after_max_retries(self):
        """Test that Telegram provider fails after max retries."""
        from src.services.alerts.telegram import TelegramAlertProvider
        from unittest.mock import AsyncMock, MagicMock, patch

        provider = TelegramAlertProvider(bot_token="test_token", chat_id="test_chat_id")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            
            # All calls fail
            mock_client.post = AsyncMock(
                side_effect=ConnectionError("Connection error")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Should fail after retries
            with pytest.raises(ConnectionError):
                await provider.send_message("Test message")
            
            # Should have been called 3 times (initial + 2 retries)
            assert mock_client.post.call_count == 3

