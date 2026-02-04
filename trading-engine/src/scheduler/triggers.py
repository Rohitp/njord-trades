"""
Market hours and trading schedule utilities.

Provides functions to determine if the market is open and when to schedule jobs.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def get_trading_timezone() -> ZoneInfo:
    """Get the configured trading timezone."""
    return ZoneInfo(settings.scheduling.timezone)


def is_market_open(dt: datetime | None = None) -> bool:
    """
    Check if US stock market is currently open.

    Market hours: 9:30 AM - 4:00 PM Eastern Time, Mon-Fri
    Does not account for holidays.

    Args:
        dt: Datetime to check. If None, uses current time.

    Returns:
        True if market is open.
    """
    tz = get_trading_timezone()

    if dt is None:
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)

    # Check weekday (0=Monday, 6=Sunday)
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)

    current_time = dt.time()
    return market_open <= current_time < market_close


def is_trading_day(dt: datetime | None = None) -> bool:
    """
    Check if the given date is a trading day.

    Currently checks weekdays only. Does not account for holidays.

    Args:
        dt: Date to check. If None, uses current date.

    Returns:
        True if it's a trading day.
    """
    tz = get_trading_timezone()

    if dt is None:
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)

    return dt.weekday() < 5  # Monday through Friday


def should_run_scheduled_job(require_market_open: bool = True) -> bool:
    """
    Check if a scheduled job should run right now.

    Respects configuration for market_hours_only and weekdays_only.

    Args:
        require_market_open: If True, also requires market to be open.

    Returns:
        True if the job should run.
    """
    if settings.scheduling.weekdays_only and not is_trading_day():
        log.debug("skipping_scheduled_job", reason="not_trading_day")
        return False

    if require_market_open and settings.scheduling.market_hours_only and not is_market_open():
        log.debug("skipping_scheduled_job", reason="market_closed")
        return False

    return True


def parse_scan_time(time_str: str) -> tuple[int, int]:
    """
    Parse a time string in HH:MM format.

    Args:
        time_str: Time in "HH:MM" format.

    Returns:
        Tuple of (hour, minute).

    Raises:
        ValueError: If format is invalid.
    """
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")

    hour = int(parts[0])
    minute = int(parts[1])

    if not (0 <= hour <= 23):
        raise ValueError(f"Invalid hour: {hour}")
    if not (0 <= minute <= 59):
        raise ValueError(f"Invalid minute: {minute}")

    return hour, minute


def get_next_market_open(dt: datetime | None = None) -> datetime:
    """
    Get the next market open time.

    Args:
        dt: Reference datetime. If None, uses current time.

    Returns:
        Datetime of next market open.
    """
    tz = get_trading_timezone()

    if dt is None:
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)

    # Start with today's market open
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)

    # If market is already open or past, move to tomorrow
    if dt >= market_open:
        from datetime import timedelta
        market_open += timedelta(days=1)

    # Skip weekends
    while market_open.weekday() >= 5:
        from datetime import timedelta
        market_open += timedelta(days=1)

    return market_open
