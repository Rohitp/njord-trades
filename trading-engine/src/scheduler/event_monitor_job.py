"""
Event monitor background job.

Polls watchlist symbols every 60 seconds during market hours
and triggers event-driven trading cycles when significant price moves are detected.
"""

from uuid import uuid4

from src.config import settings
from src.database.connection import async_session_factory
from src.scheduler.triggers import should_run_scheduled_job
from src.services.event_monitor import EventMonitor
from src.utils.logging import get_logger
from src.workflows.runner import TradingCycleRunner

log = get_logger(__name__)


async def monitor_price_moves_job() -> None:
    """
    Background job to monitor price moves and trigger event cycles.

    Runs every 60 seconds during market hours. Checks watchlist symbols
    for >5% price moves within 15 minutes and triggers event cycles.
    """
    # Check if we should run (market hours, weekdays)
    if not should_run_scheduled_job(require_market_open=True):
        return

    # Check if event monitor is enabled
    if not getattr(settings.event_monitor, "enabled", True):
        log.debug("event_monitor_disabled")
        return

    trace_id = str(uuid4())
    log.debug("event_monitor_job_starting", trace_id=trace_id)

    try:
        # Get watchlist symbols
        symbols = settings.trading.watchlist_symbols
        if not symbols:
            log.debug("event_monitor_no_watchlist")
            return

        # Create event monitor
        monitor = EventMonitor()

        # Check all symbols for significant moves
        triggered_symbols = await monitor.check_symbols(
            symbols,
            stocks_only=settings.event_monitor.stocks_only,
        )

        if not triggered_symbols:
            log.debug("event_monitor_no_triggers", trace_id=trace_id)
            return

        # Trigger event cycles for each symbol that moved
        async with async_session_factory() as session:
            runner = TradingCycleRunner(db_session=session)

            for symbol in triggered_symbols:
                try:
                    log.info(
                        "event_monitor_triggering_cycle",
                        trace_id=trace_id,
                        symbol=symbol,
                    )

                    result = await runner.run_event_cycle(
                        trigger_symbol=symbol,
                        trace_id=trace_id,
                    )

                    log.info(
                        "event_monitor_cycle_completed",
                        trace_id=trace_id,
                        symbol=symbol,
                        cycle_id=str(result.cycle_id),
                        signal_count=len(result.signals),
                        decision_count=len(result.final_decisions),
                    )

                except ValueError as e:
                    # Circuit breaker or trading disabled
                    log.warning(
                        "event_monitor_cycle_skipped",
                        trace_id=trace_id,
                        symbol=symbol,
                        reason=str(e),
                    )
                except Exception as e:
                    log.error(
                        "event_monitor_cycle_failed",
                        trace_id=trace_id,
                        symbol=symbol,
                        error=str(e),
                        exc_info=True,
                    )

    except Exception as e:
        log.error(
            "event_monitor_job_failed",
            trace_id=trace_id,
            error=str(e),
            exc_info=True,
        )

