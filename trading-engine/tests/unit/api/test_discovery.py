"""
Unit tests for discovery API endpoints.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.database.models import PickerSuggestion


class TestDiscoveryPerformanceEndpoint:
    """Test GET /api/discovery/performance endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def sample_suggestions(self):
        """Create sample picker suggestions."""
        base_date = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        return [
            PickerSuggestion(
                id=uuid4(),
                symbol="AAPL",
                picker_name="metric",
                score=0.75,
                reason="High volume",
                suggested_at=base_date,
                forward_return_1d=1.0,
                forward_return_5d=2.0,
                forward_return_20d=5.0,
                calculated_at=base_date + timedelta(days=21),
            ),
            PickerSuggestion(
                id=uuid4(),
                symbol="MSFT",
                picker_name="metric",
                score=0.80,
                reason="Momentum",
                suggested_at=base_date + timedelta(days=1),
                forward_return_1d=0.5,
                forward_return_5d=1.5,
                forward_return_20d=3.0,
                calculated_at=base_date + timedelta(days=22),
            ),
            PickerSuggestion(
                id=uuid4(),
                symbol="TSLA",
                picker_name="fuzzy",
                score=0.65,
                reason="Volatility",
                suggested_at=base_date,
                forward_return_1d=-1.0,
                forward_return_5d=-2.0,
                forward_return_20d=-3.0,
                calculated_at=base_date + timedelta(days=21),
            ),
        ]

    def test_get_performance_all_pickers(self, client, mock_db_session, sample_suggestions):
        """Test getting performance for all pickers."""
        # Mock the analyze_picker_performance function
        with patch("src.api.routers.discovery.analyze_picker_performance") as mock_analyze:
            from src.services.discovery.performance import PickerPerformance

            mock_analyze.return_value = [
                PickerPerformance(
                    picker_name="metric",
                    total_suggestions=2,
                    suggestions_with_returns=2,
                    pending_suggestions=0,
                    win_rate_1d=100.0,
                    win_rate_5d=100.0,
                    win_rate_20d=100.0,
                    avg_return_1d=0.75,
                    avg_return_5d=1.75,
                    avg_return_20d=4.0,
                    median_return_1d=0.75,
                    median_return_5d=1.75,
                    median_return_20d=4.0,
                    best_return_1d=1.0,
                    best_return_5d=2.0,
                    best_return_20d=5.0,
                    worst_return_1d=0.5,
                    worst_return_5d=1.5,
                    worst_return_20d=3.0,
                ),
                PickerPerformance(
                    picker_name="fuzzy",
                    total_suggestions=1,
                    suggestions_with_returns=1,
                    pending_suggestions=0,
                    win_rate_1d=0.0,
                    win_rate_5d=0.0,
                    win_rate_20d=0.0,
                    avg_return_1d=-1.0,
                    avg_return_5d=-2.0,
                    avg_return_20d=-3.0,
                    median_return_1d=-1.0,
                    median_return_5d=-2.0,
                    median_return_20d=-3.0,
                    best_return_1d=-1.0,
                    best_return_5d=-2.0,
                    best_return_20d=-3.0,
                    worst_return_1d=-1.0,
                    worst_return_5d=-2.0,
                    worst_return_20d=-3.0,
                ),
            ]

            # Override get_session dependency
            from src.database.connection import get_session

            def override_get_session():
                yield mock_db_session

            client.app.dependency_overrides[get_session] = override_get_session

            response = client.get("/api/discovery/performance")

            assert response.status_code == 200
            data = response.json()

            assert data["total_pickers"] == 2
            assert data["min_suggestions"] == 10  # Default
            assert len(data["pickers"]) == 2

            # Check first picker (metric)
            metric_picker = data["pickers"][0]
            assert metric_picker["picker_name"] == "metric"
            assert metric_picker["total_suggestions"] == 2
            assert metric_picker["win_rate_20d"] == 100.0
            assert metric_picker["avg_return_20d"] == 4.0

            # Check second picker (fuzzy)
            fuzzy_picker = data["pickers"][1]
            assert fuzzy_picker["picker_name"] == "fuzzy"
            assert fuzzy_picker["win_rate_20d"] == 0.0
            assert fuzzy_picker["avg_return_20d"] == -3.0

            # Cleanup
            client.app.dependency_overrides.clear()

    def test_get_performance_filter_by_picker(self, client, mock_db_session):
        """Test filtering performance by picker name."""
        with patch("src.api.routers.discovery.analyze_picker_performance") as mock_analyze:
            from src.services.discovery.performance import PickerPerformance

            mock_analyze.return_value = [
                PickerPerformance(
                    picker_name="metric",
                    total_suggestions=5,
                    suggestions_with_returns=5,
                    pending_suggestions=0,
                    win_rate_20d=80.0,
                    avg_return_20d=5.0,
                ),
            ]

            from src.database.connection import get_session

            def override_get_session():
                yield mock_db_session

            client.app.dependency_overrides[get_session] = override_get_session

            response = client.get("/api/discovery/performance?picker_name=metric")

            assert response.status_code == 200
            data = response.json()

            assert len(data["pickers"]) == 1
            assert data["pickers"][0]["picker_name"] == "metric"

            # Verify analyze_picker_performance was called with picker_name filter
            mock_analyze.assert_called_once()
            call_kwargs = mock_analyze.call_args[1]
            assert call_kwargs["picker_name"] == "metric"

            client.app.dependency_overrides.clear()

    def test_get_performance_custom_min_suggestions(self, client, mock_db_session):
        """Test custom min_suggestions parameter."""
        with patch("src.api.routers.discovery.analyze_picker_performance") as mock_analyze:
            mock_analyze.return_value = []

            from src.database.connection import get_session

            def override_get_session():
                yield mock_db_session

            client.app.dependency_overrides[get_session] = override_get_session

            response = client.get("/api/discovery/performance?min_suggestions=20")

            assert response.status_code == 200
            data = response.json()

            assert data["min_suggestions"] == 20

            # Verify analyze_picker_performance was called with min_suggestions
            call_kwargs = mock_analyze.call_args[1]
            assert call_kwargs["min_suggestions"] == 20

            client.app.dependency_overrides.clear()

    def test_get_performance_validation(self, client, mock_db_session):
        """Test query parameter validation."""
        from src.database.connection import get_session

        def override_get_session():
            yield mock_db_session

        client.app.dependency_overrides[get_session] = override_get_session

        # Test invalid min_suggestions (too low)
        response = client.get("/api/discovery/performance?min_suggestions=0")
        assert response.status_code == 422

        # Test invalid min_suggestions (too high)
        response = client.get("/api/discovery/performance?min_suggestions=2000")
        assert response.status_code == 422

        client.app.dependency_overrides.clear()

    def test_get_performance_empty_results(self, client, mock_db_session):
        """Test endpoint when no pickers meet criteria."""
        with patch("src.api.routers.discovery.analyze_picker_performance") as mock_analyze:
            mock_analyze.return_value = []

            from src.database.connection import get_session

            def override_get_session():
                yield mock_db_session

            client.app.dependency_overrides[get_session] = override_get_session

            response = client.get("/api/discovery/performance?min_suggestions=1000")

            assert response.status_code == 200
            data = response.json()

            assert data["total_pickers"] == 0
            assert len(data["pickers"]) == 0

            client.app.dependency_overrides.clear()


class TestDiscoveryABTestEndpoint:
    """Test GET /api/discovery/ab-test endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    def test_get_ab_test_metrics(self, client, mock_db_session):
        """Test getting A/B test metrics."""
        with patch("src.api.routers.discovery.calculate_ab_test_metrics") as mock_calc:
            from src.services.discovery.paper_tracker import ABTestMetrics

            mock_calc.return_value = ABTestMetrics(
                total_hypothetical_trades=10,
                total_actual_trades=8,
                matched_trades=5,
                hypothetical_win_rate=60.0,
                actual_win_rate=50.0,
                hypothetical_avg_return=3.5,
                actual_avg_return=2.8,
                hypothetical_total_pnl=350.0,
                actual_total_pnl=224.0,
                convergence_rate=80.0,
                hypothetical_outperformed=3,
                actual_outperformed=2,
            )

            from src.database.connection import get_session

            def override_get_session():
                yield mock_db_session

            client.app.dependency_overrides[get_session] = override_get_session

            response = client.get("/api/discovery/ab-test")

            assert response.status_code == 200
            data = response.json()

            assert data["total_hypothetical_trades"] == 10
            assert data["total_actual_trades"] == 8
            assert data["matched_trades"] == 5
            assert data["hypothetical_win_rate"] == 60.0
            assert data["actual_win_rate"] == 50.0
            assert data["convergence_rate"] == 80.0

            client.app.dependency_overrides.clear()

    def test_get_ab_test_metrics_with_filters(self, client, mock_db_session):
        """Test A/B test metrics with query parameters."""
        with patch("src.api.routers.discovery.calculate_ab_test_metrics") as mock_calc:
            from src.services.discovery.paper_tracker import ABTestMetrics

            mock_calc.return_value = ABTestMetrics(
                total_hypothetical_trades=5,
                total_actual_trades=3,
                matched_trades=2,
            )

            from src.database.connection import get_session

            def override_get_session():
                yield mock_db_session

            client.app.dependency_overrides[get_session] = override_get_session

            response = client.get("/api/discovery/ab-test?picker_name=metric&time_window_days=60")

            assert response.status_code == 200

            # Verify calculate_ab_test_metrics was called with filters
            mock_calc.assert_called_once()
            call_kwargs = mock_calc.call_args[1]
            assert call_kwargs["picker_name"] == "metric"
            assert call_kwargs["time_window_days"] == 60

            client.app.dependency_overrides.clear()

    def test_get_ab_test_metrics_validation(self, client, mock_db_session):
        """Test query parameter validation for A/B test endpoint."""
        from src.database.connection import get_session

        def override_get_session():
            yield mock_db_session

        client.app.dependency_overrides[get_session] = override_get_session

        # Test invalid time_window_days (too low)
        response = client.get("/api/discovery/ab-test?time_window_days=0")
        assert response.status_code == 422

        # Test invalid time_window_days (too high)
        response = client.get("/api/discovery/ab-test?time_window_days=500")
        assert response.status_code == 422

        client.app.dependency_overrides.clear()

