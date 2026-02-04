"""
Unit tests for event monitor background job.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.scheduler.event_monitor_job import monitor_price_moves_job
from src.services.market_data.provider import Quote


class TestMonitorPriceMovesJob:
    """Test event monitor background job."""

    @pytest.mark.asyncio
    async def test_job_skips_when_market_closed(self):
        """Test that job skips when market is closed."""
        with patch("src.scheduler.event_monitor_job.should_run_scheduled_job", return_value=False):
            # Should return early without doing anything
            await monitor_price_moves_job()
            # No exceptions should be raised

    @pytest.mark.asyncio
    async def test_job_skips_when_disabled(self):
        """Test that job skips when event monitor is disabled."""
        with patch("src.scheduler.event_monitor_job.should_run_scheduled_job", return_value=True), \
             patch("src.scheduler.event_monitor_job.settings") as mock_settings:
            mock_settings.event_monitor.enabled = False
            mock_settings.trading.watchlist_symbols = ["AAPL"]
            
            await monitor_price_moves_job()
            # Should return early

    @pytest.mark.asyncio
    async def test_job_skips_when_no_watchlist(self):
        """Test that job skips when watchlist is empty."""
        with patch("src.scheduler.event_monitor_job.should_run_scheduled_job", return_value=True), \
             patch("src.scheduler.event_monitor_job.settings") as mock_settings:
            mock_settings.event_monitor.enabled = True
            mock_settings.event_monitor.stocks_only = True
            mock_settings.trading.watchlist_symbols = []
            
            await monitor_price_moves_job()
            # Should return early

    @pytest.mark.asyncio
    async def test_job_triggers_cycle_on_price_move(self):
        """Test that job triggers cycle when price move detected."""
        with patch("src.scheduler.event_monitor_job.should_run_scheduled_job", return_value=True), \
             patch("src.scheduler.event_monitor_job.settings") as mock_settings, \
             patch("src.scheduler.event_monitor_job.async_session_factory") as mock_factory, \
             patch("src.scheduler.event_monitor_job.EventMonitor") as mock_monitor_class, \
             patch("src.scheduler.event_monitor_job.TradingCycleRunner") as mock_runner_class:

            # Setup mocks
            mock_settings.event_monitor.enabled = True
            mock_settings.event_monitor.stocks_only = True
            mock_settings.trading.watchlist_symbols = ["AAPL"]

            # Mock event monitor
            mock_monitor = MagicMock()
            mock_monitor.check_symbols = AsyncMock(return_value=["AAPL"])
            mock_monitor_class.return_value = mock_monitor

            # Mock database session
            mock_session = AsyncMock()
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock CircuitBreakerService (imported inside function)
            with patch("src.services.circuit_breaker.CircuitBreakerService") as mock_cb_class:
                mock_cb_service = MagicMock()
                mock_cb_service.check_auto_resume = AsyncMock()
                mock_cb_class.return_value = mock_cb_service

                # Mock runner
                mock_runner = MagicMock()
                mock_result = MagicMock()
                mock_result.cycle_id = "test-cycle-id"
                mock_result.signals = []
                mock_result.final_decisions = []
                mock_runner.run_event_cycle = AsyncMock(return_value=mock_result)
                mock_runner_class.return_value = mock_runner

                await monitor_price_moves_job()

                # Verify cycle was triggered
                assert mock_runner.run_event_cycle.called
                call_args = mock_runner.run_event_cycle.call_args
                assert call_args[1]["trigger_symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_job_handles_circuit_breaker(self):
        """Test that job handles circuit breaker gracefully."""
        with patch("src.scheduler.event_monitor_job.should_run_scheduled_job", return_value=True), \
             patch("src.scheduler.event_monitor_job.settings") as mock_settings, \
             patch("src.scheduler.event_monitor_job.async_session_factory") as mock_factory, \
             patch("src.scheduler.event_monitor_job.EventMonitor") as mock_monitor_class, \
             patch("src.scheduler.event_monitor_job.TradingCycleRunner") as mock_runner_class:

            # Setup mocks
            mock_settings.event_monitor.enabled = True
            mock_settings.event_monitor.stocks_only = True
            mock_settings.trading.watchlist_symbols = ["AAPL"]

            # Mock event monitor
            mock_monitor = MagicMock()
            mock_monitor.check_symbols = AsyncMock(return_value=["AAPL"])
            mock_monitor_class.return_value = mock_monitor

            # Mock database session
            mock_session = AsyncMock()
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock CircuitBreakerService (imported inside function)
            with patch("src.services.circuit_breaker.CircuitBreakerService") as mock_cb_class:
                mock_cb_service = MagicMock()
                mock_cb_service.check_auto_resume = AsyncMock()
                mock_cb_class.return_value = mock_cb_service

                # Mock runner to raise ValueError (circuit breaker)
                mock_runner = MagicMock()
                mock_runner.run_event_cycle = AsyncMock(side_effect=ValueError("Trading halted by circuit breaker"))
                mock_runner_class.return_value = mock_runner

                # Should not raise exception
                await monitor_price_moves_job()

                # Verify cycle was attempted
                mock_runner.run_event_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_handles_errors_gracefully(self):
        """Test that job handles errors gracefully."""
        with patch("src.scheduler.event_monitor_job.should_run_scheduled_job", return_value=True), \
             patch("src.scheduler.event_monitor_job.settings") as mock_settings, \
             patch("src.scheduler.event_monitor_job.EventMonitor") as mock_monitor_class:

            # Setup mocks
            mock_settings.event_monitor.enabled = True
            mock_settings.trading.watchlist_symbols = ["AAPL"]

            # Mock event monitor to raise exception
            mock_monitor_class.side_effect = Exception("Monitor error")

            # Should not raise exception
            await monitor_price_moves_job()

