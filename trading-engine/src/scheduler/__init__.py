"""
Scheduler module for scheduled trading cycles.

Provides APScheduler integration for running trading cycles at configured times.
"""

from src.scheduler.jobs import (
    get_scheduled_jobs,
    get_scheduler,
    register_scheduled_jobs,
    run_scheduled_trading_cycle,
    start_scheduler,
    stop_scheduler,
)
from src.scheduler.triggers import (
    get_next_market_open,
    get_trading_timezone,
    is_market_open,
    is_trading_day,
    parse_scan_time,
    should_run_scheduled_job,
)

__all__ = [
    # Jobs
    "get_scheduler",
    "get_scheduled_jobs",
    "register_scheduled_jobs",
    "run_scheduled_trading_cycle",
    "start_scheduler",
    "stop_scheduler",
    # Triggers
    "get_trading_timezone",
    "get_next_market_open",
    "is_market_open",
    "is_trading_day",
    "parse_scan_time",
    "should_run_scheduled_job",
]
