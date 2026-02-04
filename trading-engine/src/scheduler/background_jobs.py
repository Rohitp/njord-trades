"""
Background processing jobs for embeddings and discovery.

These jobs run asynchronously to avoid blocking API calls:
- Trade embeddings: Generate embeddings for completed trades (hourly)
- Market condition embeddings: Generate daily market condition embeddings (daily at market close)
- Discovery cycle: Run symbol discovery (weekly)
"""

import asyncio
from datetime import datetime, time

from src.config import settings
from src.database.connection import async_session_factory
from src.scheduler.triggers import get_trading_timezone
from src.services.embeddings.market_condition import MarketConditionService
from src.services.embeddings.trade_embedding import TradeEmbeddingService
from src.services.discovery.service import SymbolDiscoveryService
from src.utils.logging import get_logger

log = get_logger(__name__)


async def generate_trade_embeddings_job() -> None:
    """
    Background job to generate embeddings for completed trades.

    Runs hourly to process any new trades that don't have embeddings yet.
    """
    log.info("background_job_trade_embeddings_starting")

    try:
        from sqlalchemy import select
        from src.database.models import Trade, TradeEmbedding, TradeStatus

        async with async_session_factory() as session:
            # Find trades that are FILLED but don't have embeddings
            result = await session.execute(
                select(Trade)
                .where(Trade.status == TradeStatus.FILLED.value)
                .where(
                    ~select(TradeEmbedding.trade_id)
                    .where(TradeEmbedding.trade_id == Trade.id)
                    .exists()
                )
                .limit(100)  # Process up to 100 trades per run
            )
            trades = result.scalars().all()

            if not trades:
                log.info("background_job_trade_embeddings_no_trades")
                return

            embedding_service = TradeEmbeddingService()
            processed = 0
            errors = 0

            for trade in trades:
                try:
                    # Generate embedding (signal/decision may be None for older trades)
                    await embedding_service.embed_trade(
                        trade=trade,
                        signal=None,  # May not be available for historical trades
                        decision=None,
                        session=session,
                    )
                    processed += 1
                except Exception as e:
                    log.error(
                        "background_job_trade_embedding_error",
                        trade_id=str(trade.id),
                        symbol=trade.symbol,
                        error=str(e),
                    )
                    errors += 1

            await session.commit()

            log.info(
                "background_job_trade_embeddings_complete",
                processed=processed,
                errors=errors,
                total_trades=len(trades),
            )

    except Exception as e:
        log.error(
            "background_job_trade_embeddings_failed",
            error=str(e),
            exc_info=True,
        )


async def generate_market_condition_embeddings_job() -> None:
    """
    Background job to generate daily market condition embeddings.

    Runs daily at market close (4:00 PM ET) to capture end-of-day market state.
    """
    log.info("background_job_market_condition_embeddings_starting")

    try:
        async with async_session_factory() as session:
            service = MarketConditionService()
            timestamp = datetime.now()

            result = await service.embed_market_condition(
                timestamp=timestamp,
                session=session,
            )

            if result:
                await session.commit()
                log.info(
                    "background_job_market_condition_embeddings_complete",
                    timestamp=timestamp,
                )
            else:
                log.warning("background_job_market_condition_embeddings_skipped")

    except Exception as e:
        log.error(
            "background_job_market_condition_embeddings_failed",
            error=str(e),
            exc_info=True,
        )


async def run_discovery_cycle_job() -> None:
    """
    Background job to run symbol discovery cycle.

    Runs weekly to discover new symbols and update watchlist.
    """
    log.info("background_job_discovery_cycle_starting")

    try:
        async with async_session_factory() as session:
            # Build context (optional - can include portfolio state, market conditions)
            context = {
                # Could add portfolio positions, market conditions, etc.
            }

            service = SymbolDiscoveryService(db_session=session)
            result = await service.run_discovery_cycle(
                context=context,
                update_watchlist=True,
            )

            log.info(
                "background_job_discovery_cycle_complete",
                discovered_count=len(result["discovered_symbols"]),
                suggestions_count=len(result["picker_suggestions"]),
                ensemble_count=len(result["ensemble_results"]),
                watchlist_updates=result["watchlist_updates"],
            )

    except Exception as e:
        log.error(
            "background_job_discovery_cycle_failed",
            error=str(e),
            exc_info=True,
        )


def register_background_jobs(scheduler) -> None:
    """
    Register background processing jobs with scheduler.

    Args:
        scheduler: APScheduler instance
    """
    from apscheduler.triggers.cron import CronTrigger

    tz = get_trading_timezone()

    # Trade embeddings: Run hourly during market hours
    scheduler.add_job(
        generate_trade_embeddings_job,
        trigger=CronTrigger(
            minute=0,  # Top of every hour
            day_of_week="mon-fri",
            timezone=tz,
        ),
        id="background_trade_embeddings",
        name="Generate trade embeddings (hourly)",
        replace_existing=True,
        max_instances=1,
    )

    # Market condition embeddings: Run daily at market close (4:00 PM ET)
    scheduler.add_job(
        generate_market_condition_embeddings_job,
        trigger=CronTrigger(
            hour=16,  # 4:00 PM
            minute=0,
            day_of_week="mon-fri",
            timezone=tz,
        ),
        id="background_market_condition_embeddings",
        name="Generate market condition embeddings (daily at close)",
        replace_existing=True,
        max_instances=1,
    )

    # Discovery cycle: Run weekly on Sunday evening (prepare for Monday)
    scheduler.add_job(
        run_discovery_cycle_job,
        trigger=CronTrigger(
            day_of_week="sun",
            hour=20,  # 8:00 PM ET
            minute=0,
            timezone=tz,
        ),
        id="background_discovery_cycle",
        name="Run symbol discovery cycle (weekly)",
        replace_existing=True,
        max_instances=1,
    )

    log.info("background_jobs_registered", job_count=3)

