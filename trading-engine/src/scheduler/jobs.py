"""
Scheduled job definitions for the trading engine.

Jobs are registered with APScheduler and execute trading cycles.
"""

import asyncio
from datetime import datetime
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import settings
from src.scheduler.triggers import (
    get_trading_timezone,
    parse_scan_time,
    should_run_scheduled_job,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# Global scheduler instance
_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone=get_trading_timezone())
    return _scheduler


async def run_scheduled_trading_cycle() -> None:
    """
    Execute a scheduled trading cycle.

    Called by APScheduler at configured times. Checks market conditions
    before running the actual cycle.
    """
    trace_id = str(uuid4())
    log.info("scheduled_cycle_starting", trace_id=trace_id)

    # Check if we should actually run
    if not should_run_scheduled_job(require_market_open=True):
        log.info("scheduled_cycle_skipped", trace_id=trace_id, reason="conditions_not_met")
        return

    try:
        # Import here to avoid circular imports
        from src.database.connection import async_session_factory
        from src.workflows.runner import TradingCycleRunner

        # Create database session
        async with async_session_factory() as session:
            runner = TradingCycleRunner(db_session=session)

            # Run with default watchlist
            symbols = settings.trading.watchlist_symbols

            log.info(
                "scheduled_cycle_running",
                trace_id=trace_id,
                symbols=symbols,
            )

            result = await runner.run_scheduled_cycle(
                symbols=symbols,
                trace_id=trace_id,
            )

            log.info(
                "scheduled_cycle_completed",
                trace_id=trace_id,
                cycle_id=str(result.cycle_id),
                signal_count=len(result.signals),
                decision_count=len(result.final_decisions),
                execute_count=len([d for d in result.final_decisions if d.decision.value == "EXECUTE"]),
            )

    except Exception as e:
        log.error(
            "scheduled_cycle_failed",
            trace_id=trace_id,
            error=str(e),
            exc_info=True,
        )


def register_scheduled_jobs(scheduler: AsyncIOScheduler) -> None:
    """
    Register all scheduled trading jobs.

    Reads scan_times from configuration and creates cron triggers.

    Args:
        scheduler: APScheduler instance to register jobs with.
    """
    tz = get_trading_timezone()
    scan_times = settings.scheduling.scan_times

    for time_str in scan_times:
        try:
            hour, minute = parse_scan_time(time_str)

            # Create cron trigger for weekdays only
            trigger = CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week="mon-fri",
                timezone=tz,
            )

            job_id = f"trading_cycle_{hour:02d}{minute:02d}"

            scheduler.add_job(
                run_scheduled_trading_cycle,
                trigger=trigger,
                id=job_id,
                name=f"Trading cycle at {time_str} ET",
                replace_existing=True,
                max_instances=1,  # Don't overlap runs
            )

            log.info(
                "scheduled_job_registered",
                job_id=job_id,
                time=time_str,
                timezone=str(tz),
            )

        except ValueError as e:
            log.error(
                "invalid_scan_time",
                time_str=time_str,
                error=str(e),
            )


def start_scheduler() -> AsyncIOScheduler:
    """
    Start the scheduler with all registered jobs.

    Returns:
        The running scheduler instance.
    """
    scheduler = get_scheduler()

    if scheduler.running:
        log.warning("scheduler_already_running")
        return scheduler

    register_scheduled_jobs(scheduler)
    scheduler.start()

    log.info(
        "scheduler_started",
        job_count=len(scheduler.get_jobs()),
        scan_times=settings.scheduling.scan_times,
    )

    return scheduler


def stop_scheduler() -> None:
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=True)
        log.info("scheduler_stopped")
    _scheduler = None


def get_scheduled_jobs() -> list[dict]:
    """
    Get information about all scheduled jobs.

    Returns:
        List of job info dictionaries.
    """
    scheduler = get_scheduler()
    jobs = []

    for job in scheduler.get_jobs():
        # next_run_time is only available after scheduler starts
        next_run = None
        if scheduler.running:
            try:
                next_run = job.next_run_time
            except AttributeError:
                pass

        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": next_run.isoformat() if next_run else None,
            "trigger": str(job.trigger),
        })

    return jobs
