"""
Unit tests for paper trade tracker service.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.database.models import PickerSuggestion, Trade, TradeAction, TradeOutcome, TradeStatus
from src.services.discovery.paper_tracker import (
    ABTestMetrics,
    HypotheticalTrade,
    TradeComparison,
    calculate_ab_test_metrics,
    compare_trades,
    create_hypothetical_trades,
)


class TestCreateHypotheticalTrades:
    """Test create_hypothetical_trades function."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        return AsyncMock()

    @pytest.fixture
    def base_date(self):
        """Base date for suggestions."""
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def sample_suggestions(self, base_date):
        """Create sample picker suggestions."""
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
                score=0.50,
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
                score=0.80,
                reason="Volatility",
                suggested_at=base_date + timedelta(days=2),
                forward_return_1d=None,  # Not calculated yet
                forward_return_5d=None,
                forward_return_20d=None,
                calculated_at=None,
            ),
        ]

    @pytest.mark.asyncio
    async def test_create_all_suggestions(self, mock_session, sample_suggestions):
        """Test creating hypothetical trades from all suggestions."""
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = sample_suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        trades = await create_hypothetical_trades(
            session=mock_session,
            time_window_days=30,
            base_quantity=10,
        )

        assert len(trades) == 3
        assert trades[0].symbol == "AAPL"
        assert trades[0].picker_name == "metric"
        assert trades[0].hypothetical_quantity == 7  # int(10 * 0.75) = 7 (not 8)
        assert trades[0].forward_return_20d == 5.0
        assert trades[0].hypothetical_pnl_pct == 5.0

        # MSFT with lower score
        assert trades[1].hypothetical_quantity == 5  # int(10 * 0.50) = 5

        # TSLA without returns
        assert trades[2].forward_return_20d is None
        assert trades[2].hypothetical_pnl is None

    @pytest.mark.asyncio
    async def test_filter_by_picker(self, mock_session, sample_suggestions):
        """Test filtering by picker name."""
        metric_suggestions = [s for s in sample_suggestions if s.picker_name == "metric"]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = metric_suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        trades = await create_hypothetical_trades(
            session=mock_session,
            picker_name="metric",
            time_window_days=30,
        )

        assert len(trades) == 2
        assert all(t.picker_name == "metric" for t in trades)

    @pytest.mark.asyncio
    async def test_min_score_filter(self, mock_session, sample_suggestions):
        """Test filtering by minimum score."""
        # Filter suggestions by min_score before returning (simulating SQL WHERE clause)
        filtered_suggestions = [s for s in sample_suggestions if s.score >= 0.70]
        
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = filtered_suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        trades = await create_hypothetical_trades(
            session=mock_session,
            min_score=0.70,
            time_window_days=30,
        )

        # Should only include AAPL (0.75) and TSLA (0.80), not MSFT (0.50)
        assert len(trades) == 2
        assert all(t.score >= 0.70 for t in trades)

    @pytest.mark.asyncio
    async def test_time_window_filter(self, mock_session, base_date):
        """Test filtering by time window."""
        # Create suggestions at different times
        old_suggestion = PickerSuggestion(
            id=uuid4(),
            symbol="OLD",
            picker_name="metric",
            score=0.75,
            reason="Old",
            suggested_at=base_date - timedelta(days=10),
            forward_return_20d=5.0,
            calculated_at=base_date - timedelta(days=10) + timedelta(days=21),
        )

        recent_suggestion = PickerSuggestion(
            id=uuid4(),
            symbol="RECENT",
            picker_name="metric",
            score=0.75,
            reason="Recent",
            suggested_at=base_date - timedelta(days=2),
            forward_return_20d=5.0,
            calculated_at=base_date - timedelta(days=2) + timedelta(days=21),
        )

        # Filter suggestions by time window before returning (simulating SQL WHERE clause)
        # Use base_date as reference (simulating "now" in the test context)
        cutoff_date = base_date - timedelta(days=5)
        filtered_suggestions = [s for s in [old_suggestion, recent_suggestion] if s.suggested_at >= cutoff_date]
        
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = filtered_suggestions
        mock_session.execute = AsyncMock(return_value=result_mock)

        # Mock datetime.now() to return base_date so the function's cutoff calculation matches
        with patch("src.services.discovery.paper_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = base_date
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            trades = await create_hypothetical_trades(
                session=mock_session,
                time_window_days=5,  # Only last 5 days
            )

        # Should only include recent suggestion
        assert len(trades) == 1
        assert trades[0].symbol == "RECENT"


class TestCompareTrades:
    """Test compare_trades function."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        return AsyncMock()

    @pytest.fixture
    def base_date(self):
        """Base date for trades."""
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def hypothetical_trades(self, base_date):
        """Create sample hypothetical trades."""
        return [
            HypotheticalTrade(
                suggestion_id=str(uuid4()),
                symbol="AAPL",
                picker_name="metric",
                suggested_at=base_date,
                score=0.75,
                reason="Test",
                hypothetical_quantity=10,
                hypothetical_price=150.0,
                hypothetical_value=1500.0,
                forward_return_20d=5.0,
                hypothetical_pnl=75.0,
                hypothetical_pnl_pct=5.0,
            ),
            HypotheticalTrade(
                suggestion_id=str(uuid4()),
                symbol="MSFT",
                picker_name="metric",
                suggested_at=base_date + timedelta(hours=2),
                score=0.80,
                reason="Test",
                hypothetical_quantity=10,
                hypothetical_price=300.0,
                hypothetical_value=3000.0,
                forward_return_20d=-2.0,
                hypothetical_pnl=-60.0,
                hypothetical_pnl_pct=-2.0,
            ),
            HypotheticalTrade(
                suggestion_id=str(uuid4()),
                symbol="TSLA",
                picker_name="fuzzy",
                suggested_at=base_date + timedelta(days=1),
                score=0.70,
                reason="Test",
                hypothetical_quantity=10,
                hypothetical_price=200.0,
                hypothetical_value=2000.0,
                forward_return_20d=3.0,
                hypothetical_pnl=60.0,
                hypothetical_pnl_pct=3.0,
            ),
        ]

    @pytest.mark.asyncio
    async def test_match_trades_by_symbol_and_time(self, mock_session, hypothetical_trades, base_date):
        """Test matching hypothetical trades with actual trades."""
        # Create actual trades
        actual_trades = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action=TradeAction.BUY.value,
                quantity=10,
                price=150.0,
                total_value=1500.0,
                status=TradeStatus.FILLED.value,
                outcome=TradeOutcome.WIN.value,
                pnl=100.0,  # Better than hypothetical
                pnl_pct=6.67,
                created_at=base_date + timedelta(hours=1),  # Within 24 hours
            ),
            Trade(
                id=uuid4(),
                symbol="MSFT",
                action=TradeAction.BUY.value,
                quantity=10,
                price=300.0,
                total_value=3000.0,
                status=TradeStatus.FILLED.value,
                outcome=TradeOutcome.LOSS.value,
                pnl=-80.0,  # Worse than hypothetical
                pnl_pct=-2.67,
                created_at=base_date + timedelta(hours=3),  # Within 24 hours
            ),
            # TSLA has no matching actual trade
        ]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = actual_trades
        mock_session.execute = AsyncMock(return_value=result_mock)

        comparisons = await compare_trades(mock_session, hypothetical_trades, time_window_hours=24)

        assert len(comparisons) == 3

        # AAPL comparison
        aapl_comp = next(c for c in comparisons if c.symbol == "AAPL")
        assert aapl_comp.actual_trade is not None
        assert aapl_comp.pnl_difference == pytest.approx(25.0, abs=0.01)  # 100 - 75
        assert aapl_comp.return_difference == pytest.approx(1.67, abs=0.01)  # 6.67 - 5.0
        assert aapl_comp.both_positive is True
        assert aapl_comp.diverged is False

        # MSFT comparison
        msft_comp = next(c for c in comparisons if c.symbol == "MSFT")
        assert msft_comp.actual_trade is not None
        assert msft_comp.both_negative is True
        assert msft_comp.diverged is False

        # TSLA comparison (no match)
        tsla_comp = next(c for c in comparisons if c.symbol == "TSLA")
        assert tsla_comp.actual_trade is None

    @pytest.mark.asyncio
    async def test_time_window_matching(self, mock_session, hypothetical_trades, base_date):
        """Test that trades outside time window don't match."""
        # Create actual trade outside time window
        actual_trades = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action=TradeAction.BUY.value,
                quantity=10,
                price=150.0,
                total_value=1500.0,
                status=TradeStatus.FILLED.value,
                created_at=base_date + timedelta(hours=25),  # Outside 24 hour window
            ),
        ]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = actual_trades
        mock_session.execute = AsyncMock(return_value=result_mock)

        comparisons = await compare_trades(mock_session, hypothetical_trades, time_window_hours=24)

        aapl_comp = next(c for c in comparisons if c.symbol == "AAPL")
        assert aapl_comp.actual_trade is None  # Should not match


class TestCalculateABTestMetrics:
    """Test calculate_ab_test_metrics function."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        return AsyncMock()

    @pytest.fixture
    def base_date(self):
        """Base date for trades."""
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, mock_session, base_date):
        """Test calculating A/B test metrics."""
        # Mock suggestions
        suggestions = [
            PickerSuggestion(
                id=uuid4(),
                symbol="AAPL",
                picker_name="metric",
                score=0.75,
                reason="Test",
                suggested_at=base_date - timedelta(days=10),
                forward_return_20d=5.0,
                calculated_at=base_date - timedelta(days=10) + timedelta(days=21),
            ),
            PickerSuggestion(
                id=uuid4(),
                symbol="MSFT",
                picker_name="metric",
                score=0.80,
                reason="Test",
                suggested_at=base_date - timedelta(days=5),
                forward_return_20d=-2.0,
                calculated_at=base_date - timedelta(days=5) + timedelta(days=21),
            ),
        ]

        # Mock actual trades
        actual_trades = [
            Trade(
                id=uuid4(),
                symbol="AAPL",
                action=TradeAction.BUY.value,
                quantity=10,
                price=150.0,
                total_value=1500.0,
                status=TradeStatus.FILLED.value,
                outcome=TradeOutcome.WIN.value,
                pnl=100.0,
                pnl_pct=6.67,
                created_at=base_date - timedelta(days=10) + timedelta(hours=1),
            ),
        ]

        # Mock database queries
        def execute_side_effect(stmt):
            stmt_str = str(stmt).lower()
            result_mock = MagicMock()
            if "picker_suggestion" in stmt_str:
                result_mock.scalars.return_value.all.return_value = suggestions
            elif "trade" in stmt_str:
                result_mock.scalars.return_value.all.return_value = actual_trades
            else:
                result_mock.scalars.return_value.all.return_value = []
            return result_mock

        mock_session.execute = AsyncMock(side_effect=execute_side_effect)

        metrics = await calculate_ab_test_metrics(
            session=mock_session,
            time_window_days=30,
        )

        assert metrics.total_hypothetical_trades == 2
        assert metrics.total_actual_trades == 1
        assert metrics.matched_trades == 1  # AAPL matches

        # Hypothetical: 1 win (AAPL), 1 loss (MSFT) = 50% win rate
        assert metrics.hypothetical_win_rate == pytest.approx(50.0, abs=0.01)

        # Actual: 1 win = 100% win rate
        assert metrics.actual_win_rate == pytest.approx(100.0, abs=0.01)

        # Hypothetical total P&L: 
        # AAPL: 1000 * (5.0 / 100) = 50
        # MSFT: 1000 * (-2.0 / 100) = -20
        # Total: 50 + (-20) = 30
        assert metrics.hypothetical_total_pnl == pytest.approx(30.0, abs=0.01)

        # Actual total P&L: 100
        assert metrics.actual_total_pnl == pytest.approx(100.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_data(self, mock_session):
        """Test handling when no data exists."""
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)

        metrics = await calculate_ab_test_metrics(
            session=mock_session,
            time_window_days=30,
        )

        assert metrics.total_hypothetical_trades == 0
        assert metrics.total_actual_trades == 0
        assert metrics.matched_trades == 0
        assert metrics.hypothetical_win_rate is None
        assert metrics.actual_win_rate is None

