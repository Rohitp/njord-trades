"""
Unit tests for picker performance analysis service.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.database.models import PickerSuggestion
from src.services.discovery.performance import PickerPerformance, analyze_picker_performance


class TestAnalyzePickerPerformance:
    """Test analyze_picker_performance function."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def base_date(self):
        """Base date for suggestions."""
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def sample_suggestions(self, base_date):
        """Create sample picker suggestions with forward returns."""
        return [
            # Picker "metric" - 3 suggestions, all positive returns
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
                symbol="GOOGL",
                picker_name="metric",
                score=0.70,
                reason="Breakout",
                suggested_at=base_date + timedelta(days=2),
                forward_return_1d=2.0,
                forward_return_5d=4.0,
                forward_return_20d=7.0,
                calculated_at=base_date + timedelta(days=23),
            ),
            # Picker "fuzzy" - 3 suggestions, mixed returns
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
            PickerSuggestion(
                id=uuid4(),
                symbol="NVDA",
                picker_name="fuzzy",
                score=0.85,
                reason="Momentum",
                suggested_at=base_date + timedelta(days=1),
                forward_return_1d=3.0,
                forward_return_5d=6.0,
                forward_return_20d=10.0,
                calculated_at=base_date + timedelta(days=22),
            ),
            PickerSuggestion(
                id=uuid4(),
                symbol="AMZN",
                picker_name="fuzzy",
                score=0.60,
                reason="Mean reversion",
                suggested_at=base_date + timedelta(days=2),
                forward_return_1d=0.0,
                forward_return_5d=1.0,
                forward_return_20d=2.0,
                calculated_at=base_date + timedelta(days=23),
            ),
            # Picker "llm" - 2 suggestions (below min_suggestions threshold)
            PickerSuggestion(
                id=uuid4(),
                symbol="META",
                picker_name="llm",
                score=0.90,
                reason="LLM analysis",
                suggested_at=base_date,
                forward_return_1d=2.0,
                forward_return_5d=3.0,
                forward_return_20d=4.0,
                calculated_at=base_date + timedelta(days=21),
            ),
            PickerSuggestion(
                id=uuid4(),
                symbol="NFLX",
                picker_name="llm",
                score=0.88,
                reason="LLM analysis",
                suggested_at=base_date + timedelta(days=1),
                forward_return_1d=1.0,
                forward_return_5d=2.0,
                forward_return_20d=3.0,
                calculated_at=base_date + timedelta(days=22),
            ),
        ]

    @pytest.mark.asyncio
    async def test_analyze_all_pickers(self, mock_session, sample_suggestions):
        """Test analyzing performance for all pickers."""
        # Mock query result
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = sample_suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        performances = await analyze_picker_performance(
            session=mock_session,
            min_suggestions=3,
        )

        # Should return 2 pickers (metric and fuzzy), llm excluded (only 2 suggestions)
        assert len(performances) == 2
        assert performances[0].picker_name == "fuzzy"  # Sorted by avg_return_20d descending
        assert performances[1].picker_name == "metric"

        # Check metric picker metrics
        metric_perf = next(p for p in performances if p.picker_name == "metric")
        assert metric_perf.total_suggestions == 3
        assert metric_perf.suggestions_with_returns == 3
        assert metric_perf.win_rate_20d == 100.0  # All positive
        assert metric_perf.avg_return_20d == pytest.approx(5.0, abs=0.01)  # (5+3+7)/3
        assert metric_perf.median_return_20d == pytest.approx(5.0, abs=0.01)
        assert metric_perf.best_return_20d == 7.0
        assert metric_perf.worst_return_20d == 3.0

        # Check fuzzy picker metrics
        fuzzy_perf = next(p for p in performances if p.picker_name == "fuzzy")
        assert fuzzy_perf.total_suggestions == 3
        assert fuzzy_perf.win_rate_20d == pytest.approx(66.67, abs=0.01)  # 2 out of 3 positive
        assert fuzzy_perf.avg_return_20d == pytest.approx(3.0, abs=0.01)  # (-3+10+2)/3

    @pytest.mark.asyncio
    async def test_analyze_specific_picker(self, mock_session, sample_suggestions):
        """Test analyzing performance for a specific picker."""
        # Filter to only metric picker
        metric_suggestions = [s for s in sample_suggestions if s.picker_name == "metric"]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = metric_suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        performances = await analyze_picker_performance(
            session=mock_session,
            picker_name="metric",
            min_suggestions=3,
        )

        assert len(performances) == 1
        assert performances[0].picker_name == "metric"
        assert performances[0].total_suggestions == 3

    @pytest.mark.asyncio
    async def test_min_suggestions_threshold(self, mock_session, sample_suggestions):
        """Test that pickers below min_suggestions threshold are excluded."""
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = sample_suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        # With min_suggestions=3, llm should be excluded
        performances = await analyze_picker_performance(
            session=mock_session,
            min_suggestions=3,
        )

        assert len(performances) == 2
        assert all(p.total_suggestions >= 3 for p in performances)

        # With min_suggestions=2, llm should be included
        performances = await analyze_picker_performance(
            session=mock_session,
            min_suggestions=2,
        )

        assert len(performances) == 3
        picker_names = {p.picker_name for p in performances}
        assert picker_names == {"metric", "fuzzy", "llm"}

    @pytest.mark.asyncio
    async def test_partial_returns(self, mock_session, base_date):
        """Test handling suggestions with only some return periods calculated."""
        suggestions = [
            # Only 1d return calculated
            PickerSuggestion(
                id=uuid4(),
                symbol="AAPL",
                picker_name="metric",
                score=0.75,
                reason="Test",
                suggested_at=base_date,
                forward_return_1d=1.0,
                forward_return_5d=None,
                forward_return_20d=None,
                calculated_at=base_date + timedelta(days=1),
            ),
            # All returns calculated
            PickerSuggestion(
                id=uuid4(),
                symbol="MSFT",
                picker_name="metric",
                score=0.80,
                reason="Test",
                suggested_at=base_date + timedelta(days=1),
                forward_return_1d=2.0,
                forward_return_5d=3.0,
                forward_return_20d=4.0,
                calculated_at=base_date + timedelta(days=22),
            ),
        ]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        performances = await analyze_picker_performance(
            session=mock_session,
            min_suggestions=1,
        )

        assert len(performances) == 1
        perf = performances[0]

        # 1d metrics should be calculated (2 suggestions)
        assert perf.win_rate_1d == 100.0
        assert perf.avg_return_1d == pytest.approx(1.5, abs=0.01)  # (1+2)/2

        # 5d and 20d should only use the second suggestion
        assert perf.win_rate_5d == 100.0
        assert perf.avg_return_5d == 3.0
        assert perf.win_rate_20d == 100.0
        assert perf.avg_return_20d == 4.0

    @pytest.mark.asyncio
    async def test_no_suggestions(self, mock_session):
        """Test handling when no suggestions are found."""
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)

        performances = await analyze_picker_performance(
            session=mock_session,
            min_suggestions=1,
        )

        assert len(performances) == 0

    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, mock_session, base_date):
        """Test win rate calculation (positive returns = wins)."""
        suggestions = [
            PickerSuggestion(
                id=uuid4(),
                symbol=f"SYM{i}",
                picker_name="test",
                score=0.75,
                reason="Test",
                suggested_at=base_date + timedelta(days=i),
                forward_return_20d=5.0 if i % 2 == 0 else -2.0,  # Alternating positive/negative
                calculated_at=base_date + timedelta(days=i+21),
            )
            for i in range(10)  # 5 wins, 5 losses
        ]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        performances = await analyze_picker_performance(
            session=mock_session,
            min_suggestions=1,
        )

        assert len(performances) == 1
        assert performances[0].win_rate_20d == 50.0  # 5 out of 10

    @pytest.mark.asyncio
    async def test_sorting_by_performance(self, mock_session, base_date):
        """Test that performances are sorted by 20d average return (descending)."""
        suggestions = [
            # Low performer
            PickerSuggestion(
                id=uuid4(),
                symbol="LOW",
                picker_name="low",
                score=0.5,
                reason="Test",
                suggested_at=base_date,
                forward_return_20d=1.0,
                calculated_at=base_date + timedelta(days=21),
            ),
            # High performer
            PickerSuggestion(
                id=uuid4(),
                symbol="HIGH",
                picker_name="high",
                score=0.9,
                reason="Test",
                suggested_at=base_date,
                forward_return_20d=10.0,
                calculated_at=base_date + timedelta(days=21),
            ),
            # Medium performer
            PickerSuggestion(
                id=uuid4(),
                symbol="MED",
                picker_name="med",
                score=0.7,
                reason="Test",
                suggested_at=base_date,
                forward_return_20d=5.0,
                calculated_at=base_date + timedelta(days=21),
            ),
        ]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        performances = await analyze_picker_performance(
            session=mock_session,
            min_suggestions=1,
        )

        assert len(performances) == 3
        assert performances[0].picker_name == "high"  # Highest return
        assert performances[1].picker_name == "med"
        assert performances[2].picker_name == "low"  # Lowest return

