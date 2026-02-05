"""
Unit tests for alert services.

NOTE: All Telegram tests use mocks to prevent real notifications during test runs.
This is intentional - tests should not send real Telegram messages.

To test with real Telegram notifications (manual testing only):
1. Set environment variables: ALERT_TELEGRAM_BOT_TOKEN and ALERT_TELEGRAM_CHAT_ID
2. Run: pytest tests/unit/services/test_alerts.py::test_send_message_success --no-mock-telegram
   (Note: This requires a custom test marker - not recommended for CI/CD)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.alerts.service import AlertService
from src.services.alerts.telegram import TelegramAlertProvider


class TestTelegramAlertProvider:
    """Test TelegramAlertProvider."""

    @pytest.fixture
    def provider(self):
        """Create TelegramAlertProvider with test credentials."""
        return TelegramAlertProvider(
            bot_token="test_token",
            chat_id="test_chat_id",
        )

    def test_is_configured(self, provider: TelegramAlertProvider):
        """Test configuration check."""
        assert provider.is_configured() is True

    def test_is_not_configured_missing_token(self):
        """Test configuration check with missing token."""
        # Pass empty string explicitly - should use it (not fall back to config)
        provider = TelegramAlertProvider(bot_token="", chat_id="test_chat_id")
        assert provider.is_configured() is False

    def test_is_not_configured_missing_chat_id(self):
        """Test configuration check with missing chat_id."""
        # Pass empty string explicitly - should use it (not fall back to config)
        provider = TelegramAlertProvider(bot_token="test_token", chat_id="")
        assert provider.is_configured() is False

    @pytest.mark.asyncio
    async def test_send_message_success(self, provider: TelegramAlertProvider):
        """Test successful message sending."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {"ok": True}
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await provider.send_message("Test message")
            assert result is True

    @pytest.mark.asyncio
    async def test_send_message_api_error(self, provider: TelegramAlertProvider):
        """Test message sending with API error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {"ok": False, "description": "Invalid chat_id"}
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await provider.send_message("Test message")
            assert result is False

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self):
        """Test message sending when not configured."""
        # Pass empty strings explicitly - should use them (not fall back to config)
        provider = TelegramAlertProvider(bot_token="", chat_id="")
        result = await provider.send_message("Test message")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert(self, provider: TelegramAlertProvider):
        """Test send_alert with formatting."""
        with patch.object(provider, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True

            result = await provider.send_alert(
                title="Test Alert",
                message="Test message",
                severity="error",
            )

            assert result is True
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "🔴" in call_args[1]["text"]  # Error emoji
            assert "Test Alert" in call_args[1]["text"]
            assert call_args[1]["parse_mode"] == "HTML"


class TestAlertService:
    """Test AlertService."""

    @pytest.fixture
    def alert_service(self):
        """Create AlertService with mocked Telegram provider."""
        mock_telegram = MagicMock(spec=TelegramAlertProvider)
        mock_telegram.send_alert = AsyncMock(return_value=True)
        return AlertService(telegram_provider=mock_telegram)

    @pytest.mark.asyncio
    async def test_send_circuit_breaker_alert(self, alert_service: AlertService):
        """Test circuit breaker alert."""
        result = await alert_service.send_circuit_breaker_alert(
            reason="Drawdown 21.0% exceeds 20% threshold",
            drawdown_pct=21.0,
        )

        assert result is True
        alert_service.telegram.send_alert.assert_called_once()
        call_args = alert_service.telegram.send_alert.call_args
        assert call_args[1]["severity"] == "error"
        assert "Circuit Breaker Activated" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_send_auto_resume_conditions_met_alert(self, alert_service: AlertService):
        """Test auto-resume conditions met alert."""
        result = await alert_service.send_auto_resume_conditions_met_alert(
            resume_reason="Drawdown recovered to 10.0%",
            drawdown_pct=10.0,
        )

        assert result is True
        alert_service.telegram.send_alert.assert_called_once()
        call_args = alert_service.telegram.send_alert.call_args
        assert call_args[1]["severity"] == "info"
        assert "Auto-Resume Conditions Met" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_send_position_change_alert(self, alert_service: AlertService):
        """Test position change alert."""
        result = await alert_service.send_position_change_alert(
            symbol="AAPL",
            action="BUY",
            quantity=10,
            price=150.0,
            total_value=1500.0,
        )

        assert result is True
        alert_service.telegram.send_alert.assert_called_once()
        call_args = alert_service.telegram.send_alert.call_args
        assert call_args[1]["severity"] == "info"
        assert call_args[1]["disable_notification"] is True  # Silent for routine trades
        assert "AAPL" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_send_daily_pnl_summary(self, alert_service: AlertService):
        """Test daily P&L summary."""
        from datetime import datetime

        result = await alert_service.send_daily_pnl_summary(
            date=datetime(2026, 2, 4),
            daily_pnl=500.0,
            daily_pnl_pct=1.5,
            total_value=35000.0,
            cash=10000.0,
            position_count=3,
        )

        assert result is True
        alert_service.telegram.send_alert.assert_called_once()
        call_args = alert_service.telegram.send_alert.call_args
        assert call_args[1]["severity"] == "info"
        assert "Daily P&L Summary" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_send_system_error_alert(self, alert_service: AlertService):
        """Test system error alert."""
        result = await alert_service.send_system_error_alert(
            error_type="DatabaseError",
            error_message="Connection timeout",
            context={"host": "localhost", "port": 5432},
        )

        assert result is True
        alert_service.telegram.send_alert.assert_called_once()
        call_args = alert_service.telegram.send_alert.call_args
        assert call_args[1]["severity"] == "error"
        assert "System Error" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_send_test_alert(self, alert_service: AlertService):
        """Test test alert."""
        result = await alert_service.send_test_alert()

        assert result is True
        alert_service.telegram.send_alert.assert_called_once()
        call_args = alert_service.telegram.send_alert.call_args
        assert call_args[1]["title"] == "Test Alert"

