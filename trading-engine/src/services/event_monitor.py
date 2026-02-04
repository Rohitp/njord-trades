"""
Event monitor service for detecting significant price moves.

Tracks price history and detects when symbols move >5% within a time window,
triggering event-driven trading cycles.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

from src.config import settings
from src.services.market_data.service import MarketDataService
from src.utils.logging import get_logger

log = get_logger(__name__)


class PriceHistory:
    """Stores price history for a symbol."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.prices: List[tuple[datetime, float]] = []  # (timestamp, price)
        self.last_trigger_time: datetime | None = None

    def add_price(self, price: float, timestamp: datetime | None = None) -> None:
        """Add a price point to history."""
        if timestamp is None:
            timestamp = datetime.now()
        self.prices.append((timestamp, price))

        # Keep only prices within the monitoring window
        window_start = timestamp - timedelta(minutes=settings.event_monitor.move_window_minutes)
        self.prices = [
            (ts, p) for ts, p in self.prices if ts >= window_start
        ]

    def get_price_change_pct(
        self,
        current_price: float,
        window_minutes: int | None = None,
    ) -> float | None:
        """
        Calculate price change percentage over the time window.

        Args:
            current_price: Current price
            window_minutes: Time window in minutes (default: from config)

        Returns:
            Price change percentage, or None if insufficient history
        """
        if window_minutes is None:
            window_minutes = settings.event_monitor.move_window_minutes

        if not self.prices:
            return None

        # Find price from window_minutes ago
        now = datetime.now()
        window_seconds = window_minutes * 60
        
        # Find prices that are at least window_minutes old
        # Calculate age of each price: (now - ts).total_seconds()
        historical_prices = [
            (ts, p) for ts, p in self.prices 
            if (now - ts).total_seconds() >= window_seconds
        ]

        if not historical_prices:
            # Need at least one price that's old enough
            return None

        # Use most recent historical price (closest to the window boundary)
        _, historical_price = max(historical_prices, key=lambda x: x[0])

        if historical_price == 0:
            return None

        change_pct = abs((current_price - historical_price) / historical_price)
        return change_pct

    def can_trigger(self, cooldown_minutes: int | None = None) -> bool:
        """
        Check if enough time has passed since last trigger (cooldown).

        Args:
            cooldown_minutes: Cooldown period in minutes (default: from config)

        Returns:
            True if cooldown period has passed or never triggered
        """
        if cooldown_minutes is None:
            cooldown_minutes = settings.event_monitor.cooldown_minutes

        if self.last_trigger_time is None:
            return True

        cooldown_end = self.last_trigger_time + timedelta(minutes=cooldown_minutes)
        return datetime.now() >= cooldown_end

    def record_trigger(self) -> None:
        """Record that a trigger occurred (for cooldown tracking)."""
        self.last_trigger_time = datetime.now()


class EventMonitor:
    """
    Monitors price moves and detects significant changes.

    Tracks price history for symbols and detects when they move
    more than the configured threshold within the time window.
    """

    def __init__(self, market_data: MarketDataService | None = None):
        """
        Initialize event monitor.

        Args:
            market_data: MarketDataService instance (creates default if None)
        """
        self.market_data = market_data or MarketDataService()
        self.price_history: Dict[str, PriceHistory] = {}

    def _get_history(self, symbol: str) -> PriceHistory:
        """Get or create price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = PriceHistory(symbol)
        return self.price_history[symbol]

    async def check_symbol(
        self,
        symbol: str,
        stocks_only: bool = True,
    ) -> tuple[bool, float | None]:
        """
        Check if a symbol has moved significantly.

        Args:
            symbol: Symbol to check
            stocks_only: If True, skip if symbol is an option (contains '/')

        Returns:
            Tuple of (should_trigger, price_change_pct)
            - should_trigger: True if price move exceeds threshold and cooldown passed
            - price_change_pct: Percentage change (None if insufficient data)
        """
        # Skip options if stocks_only is True
        if stocks_only and "/" in symbol:
            return False, None

        try:
            # Get current quote
            quote = await self.market_data.get_quote(symbol)
            if not quote.price or quote.price <= 0:
                log.debug("event_monitor_no_price", symbol=symbol)
                return False, None

            current_price = quote.price
            history = self._get_history(symbol)

            # Check cooldown first
            if not history.can_trigger():
                log.debug(
                    "event_monitor_cooldown",
                    symbol=symbol,
                    last_trigger=history.last_trigger_time,
                )
                # Still add price to history even if cooldown active
                history.add_price(current_price)
                return False, None

            # Calculate price change BEFORE adding current price to history
            # This ensures we compare against historical prices, not the current one
            change_pct = history.get_price_change_pct(current_price)
            
            # Now add current price to history for next check
            history.add_price(current_price)
            
            if change_pct is None:
                log.debug("event_monitor_insufficient_history", symbol=symbol)
                return False, None

            # Check if move exceeds threshold
            threshold = settings.event_monitor.price_move_threshold_pct
            should_trigger = change_pct >= threshold

            if should_trigger:
                log.info(
                    "event_monitor_move_detected",
                    symbol=symbol,
                    change_pct=change_pct,
                    threshold=threshold,
                    current_price=current_price,
                )
                # Record trigger for cooldown
                history.record_trigger()

            return should_trigger, change_pct

        except Exception as e:
            log.error(
                "event_monitor_check_error",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )
            return False, None

    async def check_symbols(
        self,
        symbols: List[str],
        stocks_only: bool = True,
    ) -> List[str]:
        """
        Check multiple symbols and return those that should trigger.

        Args:
            symbols: List of symbols to check
            stocks_only: If True, skip options

        Returns:
            List of symbols that should trigger event cycles
        """
        triggered_symbols = []

        for symbol in symbols:
            should_trigger, change_pct = await self.check_symbol(
                symbol, stocks_only=stocks_only
            )
            if should_trigger:
                triggered_symbols.append(symbol)

        return triggered_symbols

    def reset_history(self, symbol: str | None = None) -> None:
        """
        Reset price history for a symbol or all symbols.

        Args:
            symbol: Symbol to reset (None = reset all)
        """
        if symbol:
            if symbol in self.price_history:
                del self.price_history[symbol]
        else:
            self.price_history.clear()

