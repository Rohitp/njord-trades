"""
Unit tests for Telegram bot command handler.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.database.models import (
    CapitalEvent,
    CapitalEventType,
    Position,
    PortfolioState,
    SystemState,
    Trade,
    TradeOutcome,
)
from src.services.alerts.telegram_bot import TelegramBot


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def telegram_bot(mock_db_session):
    """Create TelegramBot instance with mocked dependencies."""
    with patch("src.services.alerts.telegram_bot.settings") as mock_settings:
        mock_settings.alerts.telegram_bot_token = "test-token"
        mock_settings.alerts.telegram_chat_id = "12345"
        bot = TelegramBot(db_session=mock_db_session)
        # Mock the telegram.send_message to prevent real notifications
        bot.telegram.send_message = AsyncMock(return_value=True)
        return bot


class TestTelegramBotRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_allows_commands(self, telegram_bot):
        """Test that commands within rate limit are allowed."""
        chat_id = "12345"
        
        # First 10 commands should be allowed
        for i in range(10):
            assert telegram_bot._check_rate_limit(chat_id) is True
        
        # 11th command should be blocked
        assert telegram_bot._check_rate_limit(chat_id) is False

    def test_rate_limit_resets_after_window(self, telegram_bot):
        """Test that rate limit resets after time window."""
        chat_id = "12345"
        
        # Exhaust rate limit
        for _ in range(10):
            telegram_bot._check_rate_limit(chat_id)
        
        # Manually expire old entries
        telegram_bot._rate_limit[chat_id] = [
            datetime.now() - timedelta(minutes=2)
        ]
        
        # Should be allowed again
        assert telegram_bot._check_rate_limit(chat_id) is True


class TestTelegramBotAuthentication:
    """Test authentication and authorization."""

    @pytest.mark.asyncio
    async def test_rejects_unauthorized_chat_id(self, telegram_bot, mock_db_session):
        """Test that messages from unauthorized chat IDs are rejected."""
        message = {
            "chat": {"id": "99999"},  # Different chat ID
            "text": "/status",
        }
        
        result = await telegram_bot.handle_message(message)
        
        assert result is None  # Should not respond

    @pytest.mark.asyncio
    async def test_accepts_authorized_chat_id(self, telegram_bot, mock_db_session):
        """Test that messages from authorized chat ID are processed."""
        with patch("src.services.alerts.telegram_bot.settings") as mock_settings:
            mock_settings.alerts.telegram_chat_id = "12345"
            
            message = {
                "chat": {"id": "12345"},
                "text": "/status",
            }
            
            # Mock database responses
            system_result = MagicMock()
            system_result.scalar_one_or_none.return_value = SystemState(
                id=1,
                trading_enabled=True,
                circuit_breaker_active=False,
            )
            
            portfolio_result = MagicMock()
            portfolio_result.scalar_one_or_none.return_value = PortfolioState(
                id=1,
                cash=1000.0,
                total_value=5000.0,
            )
            
            position_count_result = MagicMock()
            position_count_result.scalar_one.return_value = 3
            
            async def execute_side_effect(stmt):
                stmt_str = str(stmt).lower()
                if "system_state" in stmt_str:
                    return system_result
                elif "portfolio_state" in stmt_str:
                    return portfolio_result
                elif "count" in stmt_str:
                    return position_count_result
                return MagicMock()
            
            mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
            
            result = await telegram_bot.handle_message(message)
            
            assert result is not None
            assert "System Status" in result
            assert "Enabled" in result


class TestTelegramBotCommands:
    """Test individual command handlers."""

    @pytest.mark.asyncio
    async def test_status_command(self, telegram_bot, mock_db_session):
        """Test /status command."""
        # Mock database responses
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=False,
        )
        
        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = PortfolioState(
            id=1,
            cash=1000.0,
            total_value=5000.0,
        )
        
        position_count_result = MagicMock()
        position_count_result.scalar_one.return_value = 2
        
        async def execute_side_effect(stmt):
            stmt_str = str(stmt).lower()
            if "system_state" in stmt_str:
                return system_result
            elif "portfolio_state" in stmt_str:
                return portfolio_result
            elif "count" in stmt_str:
                return position_count_result
            return MagicMock()
        
        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
        
        result = await telegram_bot._handle_status()
        
        assert "System Status" in result
        assert "Enabled" in result
        assert "$5,000.00" in result
        assert "$1,000.00" in result
        assert "2" in result

    @pytest.mark.asyncio
    async def test_portfolio_command(self, telegram_bot, mock_db_session):
        """Test /portfolio command."""
        # Mock database responses
        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = PortfolioState(
            id=1,
            cash=1000.0,
            total_value=5000.0,
            deployed_capital=4000.0,
        )
        
        positions_result = MagicMock()
        positions_result.scalars.return_value.all.return_value = [
            Position(
                symbol="AAPL",
                quantity=10,
                avg_cost=150.0,
                current_price=160.0,
                current_value=1600.0,
                sector="Technology",
            ),
            Position(
                symbol="MSFT",
                quantity=5,
                avg_cost=300.0,
                current_price=310.0,
                current_value=1550.0,
                sector="Technology",
            ),
        ]
        
        async def execute_side_effect(stmt):
            stmt_str = str(stmt).lower()
            if "portfolio_state" in stmt_str:
                return portfolio_result
            elif "position" in stmt_str:
                return positions_result
            return MagicMock()
        
        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
        
        result = await telegram_bot._handle_portfolio()
        
        assert "Portfolio" in result
        assert "$5,000.00" in result
        assert "AAPL" in result
        assert "MSFT" in result
        assert "Technology" in result

    @pytest.mark.asyncio
    async def test_trades_command(self, telegram_bot, mock_db_session):
        """Test /trades command."""
        # Mock database responses
        trades_result = MagicMock()
        trades_result.scalars.return_value.all.return_value = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action="BUY",
                quantity=10,
                price=150.0,
                total_value=1500.0,
                outcome=TradeOutcome.WIN.value,
                pnl=50.0,
                created_at=datetime.now(),
            ),
        ]
        
        async def execute_side_effect(stmt):
            if "trade" in str(stmt).lower():
                return trades_result
            return MagicMock()
        
        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
        
        result = await telegram_bot._handle_trades("10")
        
        assert "Recent Trades" in result
        assert "AAPL" in result
        assert "BUY" in result

    @pytest.mark.asyncio
    async def test_trades_command_with_limit(self, telegram_bot, mock_db_session):
        """Test /trades command with custom limit."""
        # Mock database responses
        trades_result = MagicMock()
        trades_result.scalars.return_value.all.return_value = []
        
        mock_db_session.execute = AsyncMock(return_value=trades_result)
        
        result = await telegram_bot._handle_trades("25")
        
        # Should still work with custom limit
        assert "Recent Trades" in result or "No trades found" in result

    @pytest.mark.asyncio
    async def test_metrics_command(self, telegram_bot, mock_db_session):
        """Test /metrics command."""
        # Mock database responses
        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = PortfolioState(
            id=1,
            cash=1000.0,
            total_value=5000.0,
            peak_value=6000.0,
        )
        
        trades_result = MagicMock()
        trades_result.scalars.return_value.all.return_value = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action="BUY",
                quantity=10,
                price=150.0,
                outcome=TradeOutcome.WIN.value,
                pnl=50.0,
                created_at=datetime.now() - timedelta(days=5),
            ),
            Trade(
                id=uuid4(),
                symbol="MSFT",
                action="BUY",
                quantity=5,
                price=300.0,
                outcome=TradeOutcome.LOSS.value,
                pnl=-20.0,
                created_at=datetime.now() - timedelta(days=10),
            ),
        ]
        
        pnl_sum_result = MagicMock()
        pnl_sum_result.scalar_one.return_value = 30.0
        
        deposits_result = MagicMock()
        deposits_result.scalar_one.return_value = 1000.0
        
        async def execute_side_effect(stmt):
            stmt_str = str(stmt).lower()
            if "portfolio_state" in stmt_str:
                return portfolio_result
            elif "trade" in stmt_str and "sum" not in stmt_str:
                return trades_result
            elif "sum" in stmt_str and "pnl" in stmt_str:
                return pnl_sum_result
            elif "capital_event" in stmt_str:
                return deposits_result
            return MagicMock()
        
        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
        
        result = await telegram_bot._handle_metrics()
        
        assert "Performance Metrics" in result
        assert "Win Rate" in result
        assert "Drawdown" in result
        assert "P&L" in result

    @pytest.mark.asyncio
    async def test_logs_command(self, telegram_bot, mock_db_session):
        """Test /logs command."""
        # Mock database responses
        events_result = MagicMock()
        events_result.scalars.return_value.all.return_value = []
        
        mock_db_session.execute = AsyncMock(return_value=events_result)
        
        result = await telegram_bot._handle_logs("ERROR last_hour")
        
        assert "Logs" in result
        assert "ERROR" in result
        assert "last_hour" in result

    @pytest.mark.asyncio
    async def test_query_command(self, telegram_bot, mock_db_session):
        """Test /query command."""
        result = await telegram_bot._handle_query("What trades did we make on AAPL?")
        
        assert "Natural Language Query" in result
        assert "AAPL" in result

    def test_help_command(self, telegram_bot):
        """Test /help command."""
        result = telegram_bot._handle_help()
        
        assert "Trading Bot Commands" in result
        assert "/status" in result
        assert "/portfolio" in result
        assert "/trades" in result
        assert "/metrics" in result
        assert "/logs" in result
        assert "/query" in result


class TestTelegramBotErrorHandling:
    """Test error handling in command handlers."""

    @pytest.mark.asyncio
    async def test_handles_database_errors(self, telegram_bot, mock_db_session):
        """Test that database errors are handled gracefully."""
        mock_db_session.execute = AsyncMock(side_effect=Exception("Database error"))
        
        message = {
            "chat": {"id": "12345"},
            "text": "/status",
        }
        
        with patch("src.services.alerts.telegram_bot.settings") as mock_settings:
            mock_settings.alerts.telegram_chat_id = "12345"
            
            result = await telegram_bot.handle_message(message)
            
            assert result is not None
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_handles_unknown_command(self, telegram_bot, mock_db_session):
        """Test that unknown commands return helpful message."""
        message = {
            "chat": {"id": "12345"},
            "text": "/unknown",
        }
        
        with patch("src.services.alerts.telegram_bot.settings") as mock_settings:
            mock_settings.alerts.telegram_chat_id = "12345"
            
            result = await telegram_bot.handle_message(message)
            
            assert result is not None
            assert "Unknown command" in result
            assert "/help" in result

    @pytest.mark.asyncio
    async def test_handles_non_command_messages(self, telegram_bot, mock_db_session):
        """Test that non-command messages are ignored."""
        message = {
            "chat": {"id": "12345"},
            "text": "Hello, bot!",
        }
        
        with patch("src.services.alerts.telegram_bot.settings") as mock_settings:
            mock_settings.alerts.telegram_chat_id = "12345"
            
            result = await telegram_bot.handle_message(message)
            
            assert result is None  # Should ignore non-commands

