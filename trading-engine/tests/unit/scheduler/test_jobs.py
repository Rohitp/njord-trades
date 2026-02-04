"""Tests for scheduler jobs."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.scheduler.jobs import (
    get_scheduler,
    get_scheduled_jobs,
    register_scheduled_jobs,
    run_scheduled_trading_cycle,
    stop_scheduler,
)


class TestGetScheduler:
    """Tests for get_scheduler."""

    def test_returns_scheduler_instance(self):
        # Clean up any existing scheduler
        stop_scheduler()

        scheduler = get_scheduler()
        assert scheduler is not None

        # Same instance on second call
        scheduler2 = get_scheduler()
        assert scheduler is scheduler2

        # Clean up
        stop_scheduler()


class TestRegisterScheduledJobs:
    """Tests for register_scheduled_jobs."""

    def test_registers_jobs_from_config(self):
        # Clean up
        stop_scheduler()

        scheduler = get_scheduler()
        register_scheduled_jobs(scheduler)

        jobs = scheduler.get_jobs()
        # Should have 6 jobs: 2 trading cycles + 4 background jobs
        assert len(jobs) == 6

        # Check job IDs
        job_ids = [job.id for job in jobs]
        assert "trading_cycle_1100" in job_ids
        assert "trading_cycle_1430" in job_ids
        assert "background_trade_embeddings" in job_ids
        assert "background_market_condition_embeddings" in job_ids
        assert "background_discovery_cycle" in job_ids
        assert "background_event_monitor" in job_ids

        # Clean up
        stop_scheduler()

    def test_jobs_are_cron_triggers(self):
        stop_scheduler()

        scheduler = get_scheduler()
        register_scheduled_jobs(scheduler)

        for job in scheduler.get_jobs():
            # Check that trigger string contains cron or interval info
            # Trading cycles use cron, background jobs use cron or interval
            trigger_str = str(job.trigger)
            assert "cron" in trigger_str or "interval" in trigger_str

        stop_scheduler()


class TestGetScheduledJobs:
    """Tests for get_scheduled_jobs."""

    def test_returns_job_info_list(self):
        stop_scheduler()

        scheduler = get_scheduler()
        register_scheduled_jobs(scheduler)

        jobs = get_scheduled_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) == 6  # 2 trading cycles + 4 background jobs

        for job in jobs:
            assert "id" in job
            assert "name" in job
            assert "trigger" in job
            # next_run is None until scheduler starts
            assert job["next_run"] is None

        stop_scheduler()

    def test_empty_when_no_jobs(self):
        stop_scheduler()

        # Get fresh scheduler without registering jobs
        scheduler = get_scheduler()
        jobs = get_scheduled_jobs()

        assert jobs == []

        stop_scheduler()


class TestStopScheduler:
    """Tests for stop_scheduler."""

    def test_safe_to_call_multiple_times(self):
        stop_scheduler()
        stop_scheduler()
        stop_scheduler()
        # Should not raise

    def test_clears_scheduler_instance(self):
        stop_scheduler()

        # Get a scheduler
        scheduler1 = get_scheduler()
        stop_scheduler()

        # After stop, get_scheduler returns new instance
        scheduler2 = get_scheduler()
        assert scheduler1 is not scheduler2

        stop_scheduler()


@pytest.mark.asyncio
class TestRunScheduledTradingCycle:
    """Tests for run_scheduled_trading_cycle."""

    async def test_calls_auto_resume_check(self):
        """Test that auto-resume check is called before cycle."""
        with patch("src.scheduler.jobs.should_run_scheduled_job", return_value=True):
            # Patch the imports that happen inside the function
            with patch("src.database.connection.async_session_factory") as mock_factory:
                with patch("src.workflows.runner.TradingCycleRunner") as mock_runner_class:
                    with patch("src.services.circuit_breaker.CircuitBreakerService") as mock_cb_service_class:
                        # Setup mocks
                        mock_session = MagicMock()
                        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                        mock_session.__aexit__ = AsyncMock(return_value=None)
                        mock_factory.return_value = mock_session

                        mock_runner = MagicMock()
                        mock_runner.run_scheduled_cycle = AsyncMock()
                        mock_runner_class.return_value = mock_runner

                        mock_cb_service = MagicMock()
                        mock_cb_service.check_auto_resume = AsyncMock()
                        mock_cb_service_class.return_value = mock_cb_service

                        # Run the cycle
                        await run_scheduled_trading_cycle()

                        # Verify auto-resume check was called
                        mock_cb_service_class.assert_called_once_with(mock_session)
                        mock_cb_service.check_auto_resume.assert_called_once()
