"""Tests for scheduler jobs."""

import pytest

from src.scheduler.jobs import (
    get_scheduler,
    get_scheduled_jobs,
    register_scheduled_jobs,
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
        # Should have 2 jobs (11:00 and 14:30 from default config)
        assert len(jobs) == 2

        # Check job IDs
        job_ids = [job.id for job in jobs]
        assert "trading_cycle_1100" in job_ids
        assert "trading_cycle_1430" in job_ids

        # Clean up
        stop_scheduler()

    def test_jobs_are_cron_triggers(self):
        stop_scheduler()

        scheduler = get_scheduler()
        register_scheduled_jobs(scheduler)

        for job in scheduler.get_jobs():
            # Check that trigger string contains cron info
            trigger_str = str(job.trigger)
            assert "cron" in trigger_str

        stop_scheduler()


class TestGetScheduledJobs:
    """Tests for get_scheduled_jobs."""

    def test_returns_job_info_list(self):
        stop_scheduler()

        scheduler = get_scheduler()
        register_scheduled_jobs(scheduler)

        jobs = get_scheduled_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) == 2

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
