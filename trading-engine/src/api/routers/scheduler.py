"""
Scheduler API endpoints.

Provides endpoints for viewing and managing scheduled jobs.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.scheduler import (
    get_scheduled_jobs,
    get_scheduler,
    is_market_open,
    is_trading_day,
    get_next_market_open,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/scheduler", tags=["Scheduler"])


class ScheduledJob(BaseModel):
    """Scheduled job information."""

    id: str
    name: str
    next_run: str | None = None
    trigger: str


class SchedulerStatus(BaseModel):
    """Scheduler status response."""

    running: bool
    job_count: int
    jobs: list[ScheduledJob]
    market_open: bool
    trading_day: bool
    next_market_open: str


class TriggerResponse(BaseModel):
    """Response from manually triggering a job."""

    success: bool
    message: str


@router.get("/status", response_model=SchedulerStatus)
async def get_scheduler_status() -> SchedulerStatus:
    """
    Get scheduler status and scheduled jobs.

    Returns current scheduler state, all scheduled jobs, and market status.
    """
    scheduler = get_scheduler()
    jobs = get_scheduled_jobs()

    return SchedulerStatus(
        running=scheduler.running,
        job_count=len(jobs),
        jobs=[ScheduledJob(**job) for job in jobs],
        market_open=is_market_open(),
        trading_day=is_trading_day(),
        next_market_open=get_next_market_open().isoformat(),
    )


@router.get("/jobs", response_model=list[ScheduledJob])
async def list_scheduled_jobs() -> list[ScheduledJob]:
    """
    List all scheduled jobs.

    Returns information about each scheduled trading cycle job.
    """
    jobs = get_scheduled_jobs()
    return [ScheduledJob(**job) for job in jobs]


@router.post("/jobs/{job_id}/trigger", response_model=TriggerResponse)
async def trigger_job(job_id: str) -> TriggerResponse:
    """
    Manually trigger a scheduled job.

    Runs the job immediately regardless of its schedule.
    """
    scheduler = get_scheduler()

    # Find the job
    job = scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    try:
        # Run the job immediately
        job.modify(next_run_time=datetime.now(scheduler.timezone))
        log.info("job_manually_triggered", job_id=job_id)

        return TriggerResponse(
            success=True,
            message=f"Job '{job_id}' has been triggered. It will run momentarily.",
        )
    except Exception as e:
        log.error("job_trigger_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to trigger job: {str(e)}")


@router.get("/market-status")
async def get_market_status() -> dict:
    """
    Get current market status.

    Returns whether the market is open and when it next opens.
    """
    return {
        "market_open": is_market_open(),
        "trading_day": is_trading_day(),
        "next_market_open": get_next_market_open().isoformat(),
    }
