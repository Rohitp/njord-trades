"""
Forward return calculation service for picker suggestions.

Calculates actual returns (1d, 5d, 20d) for PickerSuggestion records
to measure picker performance over time.
"""

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import PickerSuggestion
from src.services.market_data.service import MarketDataService
from src.utils.logging import get_logger

log = get_logger(__name__)


def _find_price_at_date(bars: list, target_date: datetime) -> float | None:
    """
    Find the close price at or closest to the target date from historical bars.

    Args:
        bars: List of OHLCV bars (sorted by timestamp ascending)
        target_date: Target date to find price for

    Returns:
        Close price at target date, or None if not found
    """
    # Find the bar closest to target_date (within 1 day)
    for bar in bars:
        # Check if this bar is on or after the target date
        if bar.timestamp.date() >= target_date.date():
            return bar.close

    # If no bar found, return the last bar's close price
    if bars:
        return bars[-1].close

    return None


async def calculate_forward_returns(
    suggestion: PickerSuggestion,
    session: AsyncSession,
    market_data_service: Optional[MarketDataService] = None,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate forward returns for a picker suggestion using historical price data.

    Args:
        suggestion: PickerSuggestion record to calculate returns for
        session: Database session
        market_data_service: Market data service (creates default if None)

    Returns:
        Tuple of (1d_return, 5d_return, 20d_return) as percentages.
        Returns None for any return that cannot be calculated (e.g., not enough time has passed).
    """
    if market_data_service is None:
        market_data_service = MarketDataService()

    symbol = suggestion.symbol
    suggested_at = suggestion.suggested_at

    try:
        # Get historical bars from suggestion date forward (need 20+ days)
        # Calculate how many days to fetch: from suggestion date to now, plus buffer
        now = datetime.now(suggested_at.tzinfo)
        days_since_suggestion = (now - suggested_at).days

        # Fetch enough bars to cover from suggestion date to 20 days forward
        # Add buffer to ensure we have data
        days_to_fetch = max(30, days_since_suggestion + 5)

        bars = await market_data_service.get_bars(symbol, days=days_to_fetch)

        if not bars:
            log.warning(
                "forward_return_no_bars",
                symbol=symbol,
                suggestion_id=str(suggestion.id),
            )
            return None, None, None

        # Filter bars to only include those from suggestion date onwards
        bars_from_suggestion = [
            bar for bar in bars
            if bar.timestamp.date() >= suggested_at.date()
        ]

        if not bars_from_suggestion:
            log.warning(
                "forward_return_no_bars_after_suggestion",
                symbol=symbol,
                suggestion_id=str(suggestion.id),
                suggested_at=suggested_at.isoformat(),
            )
            return None, None, None

        # Base price is the first bar on or after suggestion date
        base_price = bars_from_suggestion[0].close

        # Calculate target dates
        target_1d = suggested_at + timedelta(days=1)
        target_5d = suggested_at + timedelta(days=5)
        target_20d = suggested_at + timedelta(days=20)

        # Calculate 1d return if enough time has passed
        return_1d = None
        if now >= target_1d:
            price_1d = _find_price_at_date(bars_from_suggestion, target_1d)
            if price_1d is not None:
                return_1d = ((price_1d - base_price) / base_price) * 100

        # Calculate 5d return if enough time has passed
        return_5d = None
        if now >= target_5d:
            price_5d = _find_price_at_date(bars_from_suggestion, target_5d)
            if price_5d is not None:
                return_5d = ((price_5d - base_price) / base_price) * 100

        # Calculate 20d return if enough time has passed
        return_20d = None
        if now >= target_20d:
            price_20d = _find_price_at_date(bars_from_suggestion, target_20d)
            if price_20d is not None:
                return_20d = ((price_20d - base_price) / base_price) * 100

        return return_1d, return_5d, return_20d

    except Exception as e:
        log.error(
            "forward_return_calculation_error",
            symbol=symbol,
            suggestion_id=str(suggestion.id),
            error=str(e),
            exc_info=True,
        )
        return None, None, None


async def calculate_forward_returns_batch(
    session: AsyncSession,
    limit: int = 100,
    market_data_service: Optional[MarketDataService] = None,
) -> int:
    """
    Calculate forward returns for a batch of picker suggestions.

    Finds suggestions that don't have forward returns calculated yet
    and calculates them.

    Args:
        session: Database session
        limit: Maximum number of suggestions to process per batch
        market_data_service: Market data service (creates default if None)

    Returns:
        Number of suggestions processed
    """
    if market_data_service is None:
        market_data_service = MarketDataService()

    # Find suggestions without calculated returns
    # Process oldest suggestions first
    stmt = (
        select(PickerSuggestion)
        .where(PickerSuggestion.calculated_at.is_(None))
        .order_by(PickerSuggestion.suggested_at.asc())
        .limit(limit)
    )

    result = await session.execute(stmt)
    suggestions = result.scalars().all()

    if not suggestions:
        log.info("forward_return_batch_no_suggestions")
        return 0

    processed = 0
    errors = 0

    for suggestion in suggestions:
        try:
            return_1d, return_5d, return_20d = await calculate_forward_returns(
                suggestion, session, market_data_service
            )

            # Update suggestion with calculated returns
            suggestion.forward_return_1d = return_1d
            suggestion.forward_return_5d = return_5d
            suggestion.forward_return_20d = return_20d
            suggestion.calculated_at = datetime.now(suggestion.suggested_at.tzinfo)

            processed += 1

            log.debug(
                "forward_return_calculated",
                symbol=suggestion.symbol,
                picker=suggestion.picker_name,
                return_1d=return_1d,
                return_5d=return_5d,
                return_20d=return_20d,
            )

        except Exception as e:
            log.error(
                "forward_return_batch_error",
                symbol=suggestion.symbol,
                suggestion_id=str(suggestion.id),
                error=str(e),
                exc_info=True,
            )
            errors += 1

    await session.commit()

    log.info(
        "forward_return_batch_complete",
        processed=processed,
        errors=errors,
        total=len(suggestions),
    )

    return processed

