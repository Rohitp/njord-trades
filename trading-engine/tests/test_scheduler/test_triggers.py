"""Tests for scheduler triggers and market hours utilities."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.scheduler.triggers import (
    get_trading_timezone,
    is_market_open,
    is_trading_day,
    parse_scan_time,
    get_next_market_open,
)


class TestGetTradingTimezone:
    """Tests for get_trading_timezone."""

    def test_returns_zoneinfo(self):
        tz = get_trading_timezone()
        assert isinstance(tz, ZoneInfo)
        assert str(tz) == "America/New_York"


class TestIsMarketOpen:
    """Tests for is_market_open."""

    def test_market_open_during_trading_hours(self):
        # Wednesday at 10:00 AM ET
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 10, 0, tzinfo=tz)  # Wednesday
        assert is_market_open(dt) is True

    def test_market_open_at_open(self):
        # 9:30 AM ET exactly
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 9, 30, tzinfo=tz)
        assert is_market_open(dt) is True

    def test_market_closed_before_open(self):
        # 9:00 AM ET
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 9, 0, tzinfo=tz)
        assert is_market_open(dt) is False

    def test_market_closed_at_close(self):
        # 4:00 PM ET exactly (market is closed at 4:00)
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 16, 0, tzinfo=tz)
        assert is_market_open(dt) is False

    def test_market_closed_after_hours(self):
        # 5:00 PM ET
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 17, 0, tzinfo=tz)
        assert is_market_open(dt) is False

    def test_market_closed_on_saturday(self):
        # Saturday at noon
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 20, 12, 0, tzinfo=tz)  # Saturday
        assert is_market_open(dt) is False

    def test_market_closed_on_sunday(self):
        # Sunday at noon
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 21, 12, 0, tzinfo=tz)  # Sunday
        assert is_market_open(dt) is False

    def test_handles_naive_datetime(self):
        # Naive datetime at noon (should assume ET)
        dt = datetime(2024, 1, 17, 12, 0)  # Wednesday, no timezone
        # Should not raise
        result = is_market_open(dt)
        assert isinstance(result, bool)


class TestIsTradingDay:
    """Tests for is_trading_day."""

    def test_monday_is_trading_day(self):
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 15, 12, 0, tzinfo=tz)  # Monday
        assert is_trading_day(dt) is True

    def test_friday_is_trading_day(self):
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 19, 12, 0, tzinfo=tz)  # Friday
        assert is_trading_day(dt) is True

    def test_saturday_not_trading_day(self):
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 20, 12, 0, tzinfo=tz)  # Saturday
        assert is_trading_day(dt) is False

    def test_sunday_not_trading_day(self):
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 21, 12, 0, tzinfo=tz)  # Sunday
        assert is_trading_day(dt) is False


class TestParseScanTime:
    """Tests for parse_scan_time."""

    def test_valid_time(self):
        hour, minute = parse_scan_time("11:00")
        assert hour == 11
        assert minute == 0

    def test_valid_time_with_minutes(self):
        hour, minute = parse_scan_time("14:30")
        assert hour == 14
        assert minute == 30

    def test_midnight(self):
        hour, minute = parse_scan_time("00:00")
        assert hour == 0
        assert minute == 0

    def test_end_of_day(self):
        hour, minute = parse_scan_time("23:59")
        assert hour == 23
        assert minute == 59

    def test_strips_whitespace(self):
        hour, minute = parse_scan_time("  11:00  ")
        assert hour == 11
        assert minute == 0

    def test_invalid_format_no_colon(self):
        with pytest.raises(ValueError, match="Invalid time format"):
            parse_scan_time("1100")

    def test_invalid_format_too_many_parts(self):
        with pytest.raises(ValueError, match="Invalid time format"):
            parse_scan_time("11:00:00")

    def test_invalid_hour(self):
        with pytest.raises(ValueError, match="Invalid hour"):
            parse_scan_time("25:00")

    def test_invalid_minute(self):
        with pytest.raises(ValueError, match="Invalid minute"):
            parse_scan_time("11:60")


class TestGetNextMarketOpen:
    """Tests for get_next_market_open."""

    def test_during_market_hours_returns_tomorrow(self):
        # Wednesday at 10 AM ET - should return Thursday 9:30 AM
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 10, 0, tzinfo=tz)  # Wednesday
        next_open = get_next_market_open(dt)

        assert next_open.year == 2024
        assert next_open.month == 1
        assert next_open.day == 18  # Thursday
        assert next_open.hour == 9
        assert next_open.minute == 30

    def test_before_market_open_returns_same_day(self):
        # Wednesday at 8 AM ET - should return Wednesday 9:30 AM
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 8, 0, tzinfo=tz)  # Wednesday
        next_open = get_next_market_open(dt)

        assert next_open.day == 17  # Same day
        assert next_open.hour == 9
        assert next_open.minute == 30

    def test_friday_afternoon_returns_monday(self):
        # Friday at 5 PM ET - should return Monday 9:30 AM
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 19, 17, 0, tzinfo=tz)  # Friday
        next_open = get_next_market_open(dt)

        assert next_open.day == 22  # Monday
        assert next_open.hour == 9
        assert next_open.minute == 30

    def test_saturday_returns_monday(self):
        # Saturday - should return Monday 9:30 AM
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 20, 12, 0, tzinfo=tz)  # Saturday
        next_open = get_next_market_open(dt)

        assert next_open.day == 22  # Monday
        assert next_open.hour == 9
        assert next_open.minute == 30

    def test_sunday_returns_monday(self):
        # Sunday - should return Monday 9:30 AM
        tz = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 21, 12, 0, tzinfo=tz)  # Sunday
        next_open = get_next_market_open(dt)

        assert next_open.day == 22  # Monday
        assert next_open.hour == 9
        assert next_open.minute == 30
