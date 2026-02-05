"""
Unit tests for forward return calculation service.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.database.models import PickerSuggestion
from src.services.discovery.forward_returns import (
    _find_price_at_date,
    calculate_forward_returns,
    calculate_forward_returns_batch,
)
from src.services.market_data.provider import OHLCV


class TestFindPriceAtDate:
    """Test _find_price_at_date helper function."""

    def test_find_price_exact_match(self):
        """Test finding price on exact date."""
        target_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        bars = [
            OHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 14, tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                open=101.0,
                high=103.0,
                low=100.0,
                close=102.0,
                volume=1100000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 16, tzinfo=timezone.utc),
                open=102.0,
                high=104.0,
                low=101.0,
                close=103.0,
                volume=1200000,
            ),
        ]

        price = _find_price_at_date(bars, target_date)
        assert price == 102.0  # Close price on 2024-01-15

    def test_find_price_after_date(self):
        """Test finding price when no exact match, use next available."""
        target_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        bars = [
            OHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 14, tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 17, tzinfo=timezone.utc),  # 2 days after
                open=102.0,
                high=104.0,
                low=101.0,
                close=103.0,
                volume=1200000,
            ),
        ]

        price = _find_price_at_date(bars, target_date)
        assert price == 103.0  # Use next available bar

    def test_find_price_before_all_bars(self):
        """Test finding price when target is before all bars."""
        target_date = datetime(2024, 1, 10, tzinfo=timezone.utc)
        bars = [
            OHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
            ),
        ]

        price = _find_price_at_date(bars, target_date)
        assert price == 101.0  # Use first bar

    def test_find_price_empty_bars(self):
        """Test finding price with empty bars list."""
        target_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        bars = []

        price = _find_price_at_date(bars, target_date)
        assert price is None


class TestCalculateForwardReturns:
    """Test calculate_forward_returns function."""

    @pytest.fixture
    def mock_suggestion(self):
        """Create a mock PickerSuggestion."""
        suggestion = PickerSuggestion(
            id=uuid4(),
            symbol="AAPL",
            picker_name="metric",
            score=0.75,
            reason="High volume",
            suggested_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        return suggestion

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        return AsyncMock()

    @pytest.fixture
    def mock_market_data_service(self):
        """Mock market data service."""
        service = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_calculate_all_returns(self, mock_suggestion, mock_session, mock_market_data_service):
        """Test calculating all forward returns when enough time has passed."""
        # Mock current time (25 days after suggestion)
        now = mock_suggestion.suggested_at + timedelta(days=25)
        
        # Create bars from suggestion date forward
        base_date = mock_suggestion.suggested_at.date()
        bars = [
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date, datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=100.0,  # Base price
                volume=1000000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,  # +1% for 1d return
                volume=1100000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date + timedelta(days=5), datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=105.0,  # +5% for 5d return
                volume=1200000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date + timedelta(days=20), datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=110.0,
                low=99.0,
                close=110.0,  # +10% for 20d return
                volume=1300000,
            ),
        ]

        mock_market_data_service.get_bars = AsyncMock(return_value=bars)

        with patch("src.services.discovery.forward_returns.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            return_1d, return_5d, return_20d = await calculate_forward_returns(
                mock_suggestion, mock_session, mock_market_data_service
            )

        assert return_1d == pytest.approx(1.0, abs=0.01)  # (101 - 100) / 100 * 100
        assert return_5d == pytest.approx(5.0, abs=0.01)  # (105 - 100) / 100 * 100
        assert return_20d == pytest.approx(10.0, abs=0.01)  # (110 - 100) / 100 * 100

    @pytest.mark.asyncio
    async def test_calculate_partial_returns(self, mock_suggestion, mock_session, mock_market_data_service):
        """Test calculating returns when only some time periods have passed."""
        # Mock current time (3 days after suggestion - only 1d and 5d can be calculated)
        now = mock_suggestion.suggested_at + timedelta(days=3)
        
        base_date = mock_suggestion.suggested_at.date()
        bars = [
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date, datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=100.0,
                volume=1000000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1100000,
            ),
        ]

        mock_market_data_service.get_bars = AsyncMock(return_value=bars)

        with patch("src.services.discovery.forward_returns.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            return_1d, return_5d, return_20d = await calculate_forward_returns(
                mock_suggestion, mock_session, mock_market_data_service
            )

        assert return_1d == pytest.approx(1.0, abs=0.01)
        assert return_5d is None  # Not enough time has passed
        assert return_20d is None  # Not enough time has passed

    @pytest.mark.asyncio
    async def test_calculate_no_bars(self, mock_suggestion, mock_session, mock_market_data_service):
        """Test handling when no bars are available."""
        mock_market_data_service.get_bars = AsyncMock(return_value=[])

        return_1d, return_5d, return_20d = await calculate_forward_returns(
            mock_suggestion, mock_session, mock_market_data_service
        )

        assert return_1d is None
        assert return_5d is None
        assert return_20d is None

    @pytest.mark.asyncio
    async def test_calculate_no_bars_after_suggestion(self, mock_suggestion, mock_session, mock_market_data_service):
        """Test handling when bars exist but none after suggestion date."""
        # Bars before suggestion date
        bars = [
            OHLCV(
                symbol="AAPL",
                timestamp=datetime(2023, 12, 1, tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=100.0,
                volume=1000000,
            ),
        ]

        mock_market_data_service.get_bars = AsyncMock(return_value=bars)

        return_1d, return_5d, return_20d = await calculate_forward_returns(
            mock_suggestion, mock_session, mock_market_data_service
        )

        assert return_1d is None
        assert return_5d is None
        assert return_20d is None

    @pytest.mark.asyncio
    async def test_calculate_error_handling(self, mock_suggestion, mock_session, mock_market_data_service):
        """Test error handling when market data service raises exception."""
        mock_market_data_service.get_bars = AsyncMock(side_effect=Exception("API error"))

        return_1d, return_5d, return_20d = await calculate_forward_returns(
            mock_suggestion, mock_session, mock_market_data_service
        )

        assert return_1d is None
        assert return_5d is None
        assert return_20d is None


class TestCalculateForwardReturnsBatch:
    """Test calculate_forward_returns_batch function."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture
    def mock_suggestions(self):
        """Create mock suggestions."""
        base_date = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        return [
            PickerSuggestion(
                id=uuid4(),
                symbol="AAPL",
                picker_name="metric",
                score=0.75,
                reason="High volume",
                suggested_at=base_date,
            ),
            PickerSuggestion(
                id=uuid4(),
                symbol="MSFT",
                picker_name="fuzzy",
                score=0.80,
                reason="Momentum",
                suggested_at=base_date + timedelta(days=1),
            ),
        ]

    @pytest.mark.asyncio
    async def test_batch_no_suggestions(self, mock_session):
        """Test batch processing when no suggestions need calculation."""
        # Mock query returning no results
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        
        mock_session.execute = AsyncMock(return_value=result_mock)

        processed = await calculate_forward_returns_batch(mock_session, limit=100)

        assert processed == 0
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_processes_suggestions(self, mock_session, mock_suggestions):
        """Test batch processing multiple suggestions."""
        # Mock query returning suggestions
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = mock_suggestions
        
        mock_session.execute = AsyncMock(return_value=result_mock)

        # Mock market data service
        base_date = mock_suggestions[0].suggested_at.date()
        bars = [
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date, datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=100.0,
                volume=1000000,
            ),
            OHLCV(
                symbol="AAPL",
                timestamp=datetime.combine(base_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1100000,
            ),
        ]

        with patch("src.services.discovery.forward_returns.calculate_forward_returns") as mock_calc:
            # Mock calculate_forward_returns to return returns
            mock_calc.return_value = (1.0, None, None)  # 1d return only
            
            with patch("src.services.discovery.forward_returns.MarketDataService") as mock_mds:
                mock_service = AsyncMock()
                mock_service.get_bars = AsyncMock(return_value=bars)
                mock_mds.return_value = mock_service

                processed = await calculate_forward_returns_batch(mock_session, limit=100)

        assert processed == 2
        assert mock_suggestions[0].forward_return_1d == 1.0
        assert mock_suggestions[0].calculated_at is not None
        assert mock_suggestions[1].forward_return_1d == 1.0
        assert mock_suggestions[1].calculated_at is not None
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_handles_errors(self, mock_session, mock_suggestions):
        """Test batch processing handles errors gracefully."""
        # Mock query returning suggestions
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = mock_suggestions
        
        mock_session.execute = AsyncMock(return_value=result_mock)

        with patch("src.services.discovery.forward_returns.calculate_forward_returns") as mock_calc:
            # First suggestion succeeds, second fails
            mock_calc.side_effect = [
                (1.0, None, None),  # First succeeds
                Exception("Calculation error"),  # Second fails
            ]

            processed = await calculate_forward_returns_batch(mock_session, limit=100)

        # One processed, one error
        assert processed == 1
        assert mock_suggestions[0].forward_return_1d == 1.0
        assert mock_suggestions[0].calculated_at is not None
        # Second suggestion should not be updated
        assert mock_suggestions[1].forward_return_1d is None
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_respects_limit(self, mock_session):
        """Test batch processing respects limit parameter."""
        # Create 150 suggestions, but mock should only return 50 (respecting limit)
        all_suggestions = [
            PickerSuggestion(
                id=uuid4(),
                symbol=f"SYM{i}",
                picker_name="metric",
                score=0.75,
                reason="Test",
                suggested_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
            for i in range(150)
        ]
        
        # Only return first 50 (simulating SQL LIMIT)
        limited_suggestions = all_suggestions[:50]

        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = limited_suggestions
        
        mock_session.execute = AsyncMock(return_value=result_mock)

        with patch("src.services.discovery.forward_returns.calculate_forward_returns") as mock_calc:
            mock_calc.return_value = (1.0, None, None)

            processed = await calculate_forward_returns_batch(mock_session, limit=50)

        # Should process up to limit (50 suggestions returned by mock)
        assert processed == 50
        assert mock_calc.call_count == 50  # Verify it was called 50 times
        mock_session.commit.assert_called_once()

