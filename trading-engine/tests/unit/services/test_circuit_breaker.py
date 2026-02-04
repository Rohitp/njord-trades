"""
Unit tests for CircuitBreakerService.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import PortfolioState, SystemState, Trade, TradeOutcome
from src.services.circuit_breaker import CircuitBreakerService
from src.config import settings


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def circuit_breaker_service(mock_db_session: AsyncSession) -> CircuitBreakerService:
    """Create CircuitBreakerService instance."""
    return CircuitBreakerService(mock_db_session)


@pytest.mark.asyncio
class TestCircuitBreakerAutoResume:
    """Test auto-resume condition checking."""

    async def test_check_auto_resume_conditions_not_active(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test that conditions check returns False when circuit breaker is not active."""
        from unittest.mock import MagicMock
        
        # Mock database query to return None (no system state)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=mock_result)
        
        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()
        assert conditions_met is False
        assert reason is None

    async def test_check_auto_resume_conditions_drawdown_recovery(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test drawdown recovery condition."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active due to drawdown
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="Drawdown 21.0% exceeds 20% threshold",
        )

        # Create portfolio with recovered drawdown (< 15%)
        portfolio = PortfolioState(
            id=1,
            cash=50000.0,
            total_value=90000.0,
            peak_value=100000.0,  # 10% drawdown (below 15% threshold)
            deployed_capital=40000.0,
        )

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = portfolio

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "portfolio_state" in str(stmt).lower() or "PortfolioState" in str(stmt):
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()

        assert conditions_met is True
        assert reason is not None
        assert "drawdown" in reason.lower()
        assert "15%" in reason or "0.15" in reason

    async def test_check_auto_resume_conditions_drawdown_still_high(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test that drawdown still above threshold doesn't meet conditions."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active due to drawdown
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="Drawdown 21.0% exceeds 20% threshold",
        )

        # Create portfolio with drawdown still above 15%
        portfolio = PortfolioState(
            id=1,
            cash=50000.0,
            total_value=80000.0,
            peak_value=100000.0,  # 20% drawdown (still above 15% threshold)
            deployed_capital=30000.0,
        )

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = portfolio

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "portfolio_state" in str(stmt).lower() or "PortfolioState" in str(stmt):
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()

        assert conditions_met is False
        assert reason is None

    async def test_check_auto_resume_conditions_win_streak(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test win streak condition after consecutive loss halt."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active due to consecutive losses
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="10 consecutive losses exceeds 10 threshold",
        )

        # Create 3 winning trades (most recent)
        trades = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action="BUY",
                quantity=10,
                price=150.0,
                outcome=TradeOutcome.WIN.value,
                created_at=datetime.now() - timedelta(days=i),
            )
            for i in range(3)
        ]

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        trades_result = MagicMock()
        trades_result.scalars.return_value.all.return_value = trades

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "Trade" in str(stmt) or "trade" in str(stmt):
                return trades_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()

        assert conditions_met is True
        assert reason is not None
        assert "consecutive wins" in reason.lower() or "3" in reason

    async def test_check_auto_resume_conditions_insufficient_wins(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test that insufficient wins don't meet conditions."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active due to consecutive losses
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="10 consecutive losses exceeds 10 threshold",
        )

        # Create only 2 winning trades (need 3)
        trades = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action="BUY",
                quantity=10,
                price=150.0,
                outcome=TradeOutcome.WIN.value,
                created_at=datetime.now() - timedelta(days=i),
            )
            for i in range(2)
        ]

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        trades_result = MagicMock()
        trades_result.scalars.return_value.all.return_value = trades

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "Trade" in str(stmt) or "trade" in str(stmt):
                return trades_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()

        assert conditions_met is False
        assert reason is None

    async def test_check_auto_resume_logs_when_conditions_met(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test that check_auto_resume logs when conditions are met."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock, patch

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
            total_value=90000.0,
            peak_value=100000.0,
            deployed_capital=40000.0,
        )

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = portfolio

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "portfolio_state" in str(stmt).lower() or "PortfolioState" in str(stmt):
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        with patch("src.services.circuit_breaker.log") as mock_log:
            result = await circuit_breaker_service.check_auto_resume()

            assert result is True
            # Verify log was called with conditions met
            mock_log.info.assert_called_once()
            call_args = mock_log.info.call_args
            assert "circuit_breaker_auto_resume_conditions_met" in str(call_args)

    async def test_check_auto_resume_conditions_win_streak_with_loss(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test that win streak is broken by a loss."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active due to consecutive losses
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="10 consecutive losses exceeds 10 threshold",
        )

        # Create trades: 2 wins, then 1 loss (breaks streak)
        trades = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action="BUY",
                quantity=10,
                price=150.0,
                outcome=TradeOutcome.WIN.value,
                created_at=datetime.now() - timedelta(days=2),
            ),
            Trade(
                id=uuid4(),
                symbol="MSFT",
                action="BUY",
                quantity=10,
                price=200.0,
                outcome=TradeOutcome.WIN.value,
                created_at=datetime.now() - timedelta(days=1),
            ),
            Trade(
                id=uuid4(),
                symbol="GOOGL",
                action="BUY",
                quantity=10,
                price=100.0,
                outcome=TradeOutcome.LOSS.value,  # Loss breaks streak
                created_at=datetime.now(),
            ),
        ]

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        trades_result = MagicMock()
        trades_result.scalars.return_value.all.return_value = trades

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "Trade" in str(stmt) or "trade" in str(stmt):
                return trades_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()

        assert conditions_met is False
        assert reason is None

    async def test_check_auto_resume_conditions_zero_peak_value(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test that zero peak_value doesn't cause division by zero."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active due to drawdown
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="Drawdown 21.0% exceeds 20% threshold",
        )

        # Create portfolio with zero peak_value
        portfolio = PortfolioState(
            id=1,
            cash=50000.0,
            total_value=90000.0,
            peak_value=0.0,  # Zero peak value
            deployed_capital=40000.0,
        )

        # Mock database queries
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = portfolio

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "portfolio_state" in str(stmt).lower() or "PortfolioState" in str(stmt):
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()

        # Should handle zero peak_value gracefully
        assert conditions_met is False
        assert reason is None

    async def test_check_auto_resume_conditions_missing_portfolio(
        self, circuit_breaker_service: CircuitBreakerService, mock_db_session: AsyncSession
    ):
        """Test that missing portfolio state is handled gracefully."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active due to drawdown
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=True,
            circuit_breaker_reason="Drawdown 21.0% exceeds 20% threshold",
        )

        # Mock database queries - portfolio is None
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = None  # Missing portfolio

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            elif "portfolio_state" in str(stmt).lower() or "PortfolioState" in str(stmt):
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)

        conditions_met, reason = await circuit_breaker_service.check_auto_resume_conditions()

        assert conditions_met is False
        assert reason is None

