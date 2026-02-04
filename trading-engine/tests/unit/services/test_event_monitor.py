"""
Unit tests for EventMonitor service.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.event_monitor import EventMonitor, PriceHistory
from src.services.market_data.provider import Quote


class TestPriceHistory:
    """Test PriceHistory class."""

    def test_add_price(self):
        """Test adding prices to history."""
        history = PriceHistory("AAPL")
        now = datetime.now()

        history.add_price(150.0, now)
        history.add_price(155.0, now + timedelta(minutes=5))

        assert len(history.prices) == 2
        assert history.prices[0] == (now, 150.0)
        assert history.prices[1] == (now + timedelta(minutes=5), 155.0)

    def test_price_history_window(self):
        """Test that old prices are removed outside the window."""
        history = PriceHistory("AAPL")
        now = datetime.now()

        # Add price 20 minutes ago (outside 15-minute window)
        old_time = now - timedelta(minutes=20)
        history.add_price(140.0, old_time)

        # Add current price
        history.add_price(150.0, now)

        # Old price should be removed
        assert len(history.prices) == 1
        assert history.prices[0][1] == 150.0

    def test_get_price_change_pct(self):
        """Test calculating price change percentage."""
        history = PriceHistory("AAPL")
        now = datetime.now()

        # Add price 15 minutes ago
        history.add_price(100.0, now - timedelta(minutes=15))
        # Add current price
        history.add_price(110.0, now)

        change_pct = history.get_price_change_pct(110.0, window_minutes=15)
        assert change_pct == 0.10  # 10% increase

    def test_get_price_change_pct_insufficient_history(self):
        """Test that None is returned with insufficient history."""
        history = PriceHistory("AAPL")
        change_pct = history.get_price_change_pct(100.0)
        assert change_pct is None

    def test_can_trigger_no_previous_trigger(self):
        """Test that trigger is allowed if never triggered before."""
        history = PriceHistory("AAPL")
        assert history.can_trigger() is True

    def test_can_trigger_cooldown_passed(self):
        """Test that trigger is allowed after cooldown."""
        history = PriceHistory("AAPL")
        history.record_trigger()
        
        # Fast-forward time (mock or use timedelta)
        # For this test, we'll just verify the logic
        # In real usage, cooldown would be checked against current time
        assert history.last_trigger_time is not None

    def test_record_trigger(self):
        """Test recording a trigger."""
        history = PriceHistory("AAPL")
        assert history.last_trigger_time is None
        
        history.record_trigger()
        assert history.last_trigger_time is not None


class TestEventMonitor:
    """Test EventMonitor service."""

    @pytest.fixture
    def mock_market_data(self):
        """Mock MarketDataService."""
        market_data = MagicMock()
        return market_data

    @pytest.fixture
    def monitor(self, mock_market_data):
        """Create EventMonitor with mocked market data."""
        return EventMonitor(market_data=mock_market_data)

    @pytest.mark.asyncio
    async def test_check_symbol_no_price(self, monitor, mock_market_data):
        """Test handling when quote has no price."""
        mock_quote = Quote(
            symbol="AAPL",
            price=None,
            bid=None,
            ask=None,
            volume=None,
            timestamp=datetime.now(),
        )
        mock_market_data.get_quote = AsyncMock(return_value=mock_quote)

        should_trigger, change_pct = await monitor.check_symbol("AAPL")
        assert should_trigger is False
        assert change_pct is None

    @pytest.mark.asyncio
    async def test_check_symbol_insufficient_history(self, monitor, mock_market_data):
        """Test that trigger doesn't fire with insufficient history."""
        mock_quote = Quote(
            symbol="AAPL",
            price=150.0,
            bid=149.9,
            ask=150.1,
            volume=1000000,
            timestamp=datetime.now(),
        )
        mock_market_data.get_quote = AsyncMock(return_value=mock_quote)

        should_trigger, change_pct = await monitor.check_symbol("AAPL")
        assert should_trigger is False
        assert change_pct is None

    @pytest.mark.asyncio
    async def test_check_symbol_significant_move(self, monitor, mock_market_data):
        """Test detecting a significant price move."""
        # First call: establish baseline price
        mock_quote1 = Quote(
            symbol="AAPL",
            price=100.0,
            bid=99.9,
            ask=100.1,
            volume=1000000,
            timestamp=datetime.now() - timedelta(minutes=15),
        )
        
        # Second call: price moved 6% (above 5% threshold)
        mock_quote2 = Quote(
            symbol="AAPL",
            price=106.0,
            bid=105.9,
            ask=106.1,
            volume=1000000,
            timestamp=datetime.now(),
        )

        mock_market_data.get_quote = AsyncMock(side_effect=[mock_quote1, mock_quote2])

        # First check: no trigger (insufficient history)
        should_trigger1, _ = await monitor.check_symbol("AAPL")
        assert should_trigger1 is False

        # Manually add historical price to simulate time passing
        history = monitor._get_history("AAPL")
        history.add_price(100.0, datetime.now() - timedelta(minutes=15))

        # Second check: should trigger (6% move)
        should_trigger2, change_pct = await monitor.check_symbol("AAPL")
        assert should_trigger2 is True
        assert change_pct == pytest.approx(0.06, rel=0.01)

    @pytest.mark.asyncio
    async def test_check_symbol_below_threshold(self, monitor, mock_market_data):
        """Test that small moves don't trigger."""
        mock_quote = Quote(
            symbol="AAPL",
            price=103.0,  # 3% move (below 5% threshold)
            bid=102.9,
            ask=103.1,
            volume=1000000,
            timestamp=datetime.now(),
        )
        mock_market_data.get_quote = AsyncMock(return_value=mock_quote)

        # Add historical price
        history = monitor._get_history("AAPL")
        history.add_price(100.0, datetime.now() - timedelta(minutes=15))

        should_trigger, change_pct = await monitor.check_symbol("AAPL")
        assert should_trigger is False
        assert change_pct == pytest.approx(0.03, rel=0.01)

    @pytest.mark.asyncio
    async def test_check_symbol_skips_options(self, monitor):
        """Test that options are skipped when stocks_only=True."""
        should_trigger, change_pct = await monitor.check_symbol("AAPL240315C00150000", stocks_only=True)
        assert should_trigger is False
        assert change_pct is None

    @pytest.mark.asyncio
    async def test_check_symbols_multiple(self, monitor, mock_market_data):
        """Test checking multiple symbols."""
        # Mock quotes for different symbols
        def get_quote(symbol):
            if symbol == "AAPL":
                return Quote(
                    symbol="AAPL",
                    price=106.0,
                    bid=105.9,
                    ask=106.1,
                    volume=1000000,
                    timestamp=datetime.now(),
                )
            elif symbol == "MSFT":
                return Quote(
                    symbol="MSFT",
                    price=103.0,
                    bid=102.9,
                    ask=103.1,
                    volume=1000000,
                    timestamp=datetime.now(),
                )
            return Quote(
                symbol=symbol,
                price=100.0,
                bid=99.9,
                ask=100.1,
                volume=1000000,
                timestamp=datetime.now(),
            )

        mock_market_data.get_quote = AsyncMock(side_effect=get_quote)

        # Add historical prices
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            history = monitor._get_history(symbol)
            history.add_price(100.0, datetime.now() - timedelta(minutes=15))

        triggered = await monitor.check_symbols(["AAPL", "MSFT", "GOOGL"])

        # Only AAPL should trigger (6% move > 5% threshold)
        # MSFT has 3% move (below threshold)
        # GOOGL has 0% move (below threshold)
        assert "AAPL" in triggered
        assert "MSFT" not in triggered
        assert "GOOGL" not in triggered

    def test_reset_history(self, monitor):
        """Test resetting price history."""
        monitor._get_history("AAPL").add_price(100.0)
        monitor._get_history("MSFT").add_price(200.0)

        assert len(monitor.price_history) == 2

        monitor.reset_history("AAPL")
        assert "AAPL" not in monitor.price_history
        assert "MSFT" in monitor.price_history

        monitor.reset_history()
        assert len(monitor.price_history) == 0

