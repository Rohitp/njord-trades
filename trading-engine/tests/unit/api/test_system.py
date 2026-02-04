"""
Unit tests for system API endpoints.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.main import create_app
from src.database.models import PortfolioState, SystemState


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def client(mock_db_session: AsyncSession) -> TestClient:
    """Create test client with mocked database session."""
    app = create_app()
    
    # Override get_session dependency
    # FastAPI async dependencies should be async generators
    from src.database.connection import get_session
    async def override_get_session():
        yield mock_db_session
    
    app.dependency_overrides[get_session] = override_get_session
    
    return TestClient(app)


class TestCircuitBreakerResumeEndpoint:
    """Test POST /api/system/circuit-breaker/resume endpoint."""

    def test_resume_conditions_not_met(
        self, client: TestClient, mock_db_session: AsyncSession
    ):
        """Test that resume is rejected when conditions are not met."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active
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
            peak_value=100000.0,  # 20% drawdown (still above 15%)
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
        mock_db_session.commit = AsyncMock()

        response = client.post(
            "/api/system/circuit-breaker/resume",
            json={"reason": "Manual resume attempt"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["conditions_met"] is False
        assert "conditions are not met" in data["message"].lower()

    def test_resume_conditions_met_success(
        self, client: TestClient, mock_db_session: AsyncSession
    ):
        """Test successful resume when conditions are met."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker active
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
            peak_value=100000.0,  # 10% drawdown (below 15%)
            deployed_capital=40000.0,
        )

        # Mock database queries - first call returns system_state, second returns portfolio
        # Then after reset, system_state is updated
        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        portfolio_result = MagicMock()
        portfolio_result.scalar_one_or_none.return_value = portfolio

        call_count = {"system": 0, "portfolio": 0}

        async def execute_side_effect(stmt):
            stmt_str = str(stmt).lower()
            if "system_state" in stmt_str or "systemstate" in stmt_str:
                call_count["system"] += 1
                return system_result
            elif "portfolio_state" in stmt_str or "portfoliostate" in stmt_str:
                call_count["portfolio"] += 1
                return portfolio_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
        mock_db_session.commit = AsyncMock()

        response = client.post(
            "/api/system/circuit-breaker/resume",
            json={"reason": "Manual resume approved"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["conditions_met"] is True
        assert data["resume_reason"] is not None
        assert "drawdown" in data["resume_reason"].lower()

    def test_resume_circuit_breaker_not_active(
        self, client: TestClient, mock_db_session: AsyncSession
    ):
        """Test that resume fails when circuit breaker is not active."""
        from sqlalchemy import select
        from unittest.mock import AsyncMock, MagicMock

        # Create system state with circuit breaker NOT active
        system_state = SystemState(
            id=1,
            trading_enabled=True,
            circuit_breaker_active=False,
            circuit_breaker_reason=None,
        )

        system_result = MagicMock()
        system_result.scalar_one_or_none.return_value = system_state

        async def execute_side_effect(stmt):
            if "system_state" in str(stmt).lower() or "SystemState" in str(stmt):
                return system_result
            return MagicMock()

        mock_db_session.execute = AsyncMock(side_effect=execute_side_effect)
        mock_db_session.commit = AsyncMock()

        response = client.post(
            "/api/system/circuit-breaker/resume",
            json={"reason": "Manual resume attempt"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not active" in data["message"].lower() or "conditions are not met" in data["message"].lower()

