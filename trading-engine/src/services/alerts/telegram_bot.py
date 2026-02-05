"""
Telegram bot command handler for querying trading system.

Provides mobile-friendly command interface:
- /status - System status
- /portfolio - Current holdings
- /trades - Recent trades
- /metrics - Performance metrics
- /logs - Query logs
- /query - Natural language queries
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.models import (
    CapitalEvent,
    CapitalEventType,
    Position,
    PortfolioState,
    SystemState,
    Trade,
    TradeOutcome,
)
from src.services.alerts.telegram import TelegramAlertProvider
from src.utils.logging import get_logger

log = get_logger(__name__)


class TelegramBot:
    """
    Telegram bot command handler.
    
    Handles incoming Telegram messages and executes commands.
    Provides query interface for trading system status, portfolio, trades, metrics.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        """
        Initialize Telegram bot.
        
        Args:
            db_session: Database session for queries
            bot_token: Telegram bot token (default: from config)
            chat_id: Telegram chat ID (default: from config)
        """
        self.db_session = db_session
        self.telegram = TelegramAlertProvider(
            bot_token=bot_token or settings.alerts.telegram_bot_token,
            chat_id=chat_id or settings.alerts.telegram_chat_id,
        )
        
        # Rate limiting: track commands per chat_id
        self._rate_limit: dict[str, list[datetime]] = defaultdict(list)
        self._rate_limit_window = timedelta(minutes=1)
        self._max_commands_per_window = 10

    def _check_rate_limit(self, chat_id: str) -> bool:
        """
        Check if chat_id has exceeded rate limit.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            True if within rate limit, False if exceeded
        """
        now = datetime.now()
        # Remove old entries outside window
        self._rate_limit[chat_id] = [
            ts for ts in self._rate_limit[chat_id]
            if now - ts < self._rate_limit_window
        ]
        
        # Check limit
        if len(self._rate_limit[chat_id]) >= self._max_commands_per_window:
            return False
        
        # Record this command
        self._rate_limit[chat_id].append(now)
        return True

    async def handle_message(self, message: dict[str, Any]) -> str | None:
        """
        Handle incoming Telegram message.
        
        Args:
            message: Telegram message object from webhook
            
        Returns:
            Response text to send, or None if no response needed
        """
        # Extract message data
        chat = message.get("chat", {})
        chat_id = str(chat.get("id"))
        text_content = message.get("text", "").strip()
        
        # Verify chat_id matches configured chat_id
        expected_chat_id = settings.alerts.telegram_chat_id
        if chat_id != expected_chat_id:
            log.warning(
                "telegram_unauthorized_chat",
                chat_id=chat_id,
                expected_chat_id=expected_chat_id,
            )
            return None  # Don't respond to unauthorized chats
        
        # Check rate limit
        if not self._check_rate_limit(chat_id):
            return "⚠️ Rate limit exceeded. Please wait a minute before sending more commands."
        
        # Parse command
        if not text_content.startswith("/"):
            return None  # Ignore non-command messages
        
        parts = text_content.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            # Route to command handler
            if command == "/status":
                return await self._handle_status()
            elif command == "/portfolio":
                return await self._handle_portfolio()
            elif command == "/trades":
                return await self._handle_trades(args)
            elif command == "/metrics":
                return await self._handle_metrics()
            elif command == "/logs":
                return await self._handle_logs(args)
            elif command == "/query":
                return await self._handle_query(args)
            elif command in ["/start", "/help"]:
                return self._handle_help()
            else:
                return f"❓ Unknown command: {command}\n\nUse /help to see available commands."
        except Exception as e:
            log.error(
                "telegram_command_error",
                command=command,
                error=str(e),
                exc_info=True,
            )
            return f"❌ Error executing command: {str(e)}"

    async def _handle_status(self) -> str:
        """Handle /status command."""
        # Load system state
        result = await self.db_session.execute(
            select(SystemState).where(SystemState.id == 1)
        )
        system_state = result.scalar_one_or_none()
        
        # Load portfolio state
        result = await self.db_session.execute(
            select(PortfolioState).where(PortfolioState.id == 1)
        )
        portfolio = result.scalar_one_or_none()
        
        # Count positions
        result = await self.db_session.execute(
            select(func.count()).select_from(Position)
        )
        position_count = result.scalar_one() or 0
        
        # Build response
        trading_status = "✅ Enabled" if (system_state and system_state.trading_enabled) else "❌ Disabled"
        circuit_breaker = "🔴 Active" if (system_state and system_state.circuit_breaker_active) else "✅ Inactive"
        
        if system_state and system_state.circuit_breaker_active:
            circuit_breaker += f"\n   Reason: {system_state.circuit_breaker_reason or 'Unknown'}"
        
        portfolio_value = f"${portfolio.total_value:,.2f}" if portfolio else "N/A"
        cash = f"${portfolio.cash:,.2f}" if portfolio else "N/A"
        
        return f"""📊 <b>System Status</b>

Trading: {trading_status}
Circuit Breaker: {circuit_breaker}
Portfolio Value: {portfolio_value}
Cash Available: {cash}
Positions: {position_count}"""

    async def _handle_portfolio(self) -> str:
        """Handle /portfolio command."""
        # Load portfolio state
        result = await self.db_session.execute(
            select(PortfolioState).where(PortfolioState.id == 1)
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            return "❌ Portfolio not initialized"
        
        # Load positions
        result = await self.db_session.execute(
            select(Position).order_by(Position.current_value.desc())
        )
        positions = list(result.scalars().all())
        
        # Calculate sector allocation
        sector_allocation: dict[str, float] = {}
        for pos in positions:
            if pos.sector:
                sector_allocation[pos.sector] = (
                    sector_allocation.get(pos.sector, 0.0) + pos.current_value
                )
        
        # Build response
        lines = [
            f"💼 <b>Portfolio</b>",
            "",
            f"Total Value: <b>${portfolio.total_value:,.2f}</b>",
            f"Cash: ${portfolio.cash:,.2f}",
            f"Deployed: ${portfolio.deployed_capital:,.2f}",
            "",
        ]
        
        if positions:
            lines.append("<b>Positions:</b>")
            for pos in positions:
                pnl = pos.current_value - (pos.quantity * pos.avg_cost)
                pnl_pct = (pnl / (pos.quantity * pos.avg_cost) * 100) if pos.quantity > 0 else 0
                pnl_emoji = "📈" if pnl >= 0 else "📉"
                lines.append(
                    f"{pnl_emoji} <b>{pos.symbol}</b>: {pos.quantity} @ ${pos.avg_cost:.2f} "
                    f"= ${pos.current_value:,.2f} ({pnl_pct:+.2f}%)"
                )
        else:
            lines.append("No positions")
        
        if sector_allocation:
            lines.append("")
            lines.append("<b>Sector Allocation:</b>")
            for sector, value in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
                pct = (value / portfolio.total_value * 100) if portfolio.total_value > 0 else 0
                lines.append(f"  {sector}: ${value:,.2f} ({pct:.1f}%)")
        
        return "\n".join(lines)

    async def _handle_trades(self, args: str) -> str:
        """Handle /trades command."""
        # Parse limit from args
        limit = 10
        if args:
            try:
                limit = int(args.strip())
                limit = max(1, min(limit, 50))  # Clamp between 1 and 50
            except ValueError:
                pass
        
        # Load recent trades
        result = await self.db_session.execute(
            select(Trade)
            .order_by(Trade.created_at.desc())
            .limit(limit)
        )
        trades = list(result.scalars().all())
        
        if not trades:
            return "📋 No trades found"
        
        # Build response
        lines = [
            f"📋 <b>Recent Trades</b> (last {len(trades)})",
            "",
        ]
        
        for trade in trades:
            outcome_emoji = {
                TradeOutcome.WIN.value: "✅",
                TradeOutcome.LOSS.value: "❌",
                TradeOutcome.BREAKEVEN.value: "➖",
                TradeOutcome.OPEN.value: "⏳",
            }.get(trade.outcome, "❓")
            
            pnl_str = ""
            if trade.pnl is not None:
                pnl_str = f" | P&L: ${trade.pnl:+,.2f}"
            
            timestamp = trade.created_at.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"{outcome_emoji} <b>{trade.symbol}</b> {trade.action} "
                f"{trade.quantity} @ ${trade.price:.2f}{pnl_str}"
            )
            lines.append(f"   {timestamp}")
        
        return "\n".join(lines)

    async def _handle_metrics(self) -> str:
        """Handle /metrics command."""
        # Load portfolio state
        result = await self.db_session.execute(
            select(PortfolioState).where(PortfolioState.id == 1)
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            return "❌ Portfolio not initialized"
        
        # Calculate drawdown
        drawdown_pct = 0.0
        if portfolio.peak_value > 0:
            drawdown_pct = (
                (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value * 100
            )
        
        # Calculate win rate (30-day)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        result = await self.db_session.execute(
            select(Trade)
            .where(Trade.created_at >= thirty_days_ago)
            .where(Trade.outcome.in_([TradeOutcome.WIN.value, TradeOutcome.LOSS.value]))
        )
        recent_trades = list(result.scalars().all())
        
        win_count = sum(1 for t in recent_trades if t.outcome == TradeOutcome.WIN.value)
        total_closed = len(recent_trades)
        win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0.0
        
        # Calculate total P&L (all-time)
        result = await self.db_session.execute(
            select(func.sum(Trade.pnl)).where(Trade.pnl.isnot(None))
        )
        total_pnl = result.scalar_one() or 0.0
        
        # Calculate P&L by period
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        
        # Calculate today P&L
        result = await self.db_session.execute(
            select(func.sum(Trade.pnl))
            .where(Trade.pnl.isnot(None))
            .where(Trade.created_at >= today_start)
        )
        today_pnl = result.scalar_one() or 0.0
        
        # Calculate week P&L
        result = await self.db_session.execute(
            select(func.sum(Trade.pnl))
            .where(Trade.pnl.isnot(None))
            .where(Trade.created_at >= week_start)
        )
        week_pnl = result.scalar_one() or 0.0
        
        # Calculate month P&L
        result = await self.db_session.execute(
            select(func.sum(Trade.pnl))
            .where(Trade.pnl.isnot(None))
            .where(Trade.created_at >= month_start)
        )
        month_pnl = result.scalar_one() or 0.0
        
        # Calculate alpha vs deposits
        result = await self.db_session.execute(
            select(func.sum(CapitalEvent.amount))
            .where(CapitalEvent.event_type == CapitalEventType.DEPOSIT.value)
        )
        total_deposits = result.scalar_one() or 0.0
        
        alpha = portfolio.total_value - total_deposits if total_deposits > 0 else 0.0
        
        # Build response
        return f"""📈 <b>Performance Metrics</b>

Win Rate (30d): {win_rate:.1f}% ({win_count}/{total_closed})
Drawdown: {drawdown_pct:.2f}%

<b>P&L:</b>
Today: ${today_pnl:+,.2f}
Week: ${week_pnl:+,.2f}
Month: ${month_pnl:+,.2f}
All-Time: ${total_pnl:+,.2f}

Alpha vs Deposits: ${alpha:+,.2f}
(Total Deposits: ${total_deposits:,.2f})"""

    async def _handle_logs(self, args: str) -> str:
        """Handle /logs command."""
        # Parse args: /logs ERROR last_hour
        parts = args.split() if args else []
        level = parts[0].upper() if parts else "ERROR"
        time_range = parts[1].lower() if len(parts) > 1 else "last_hour"
        
        # Map time range to timedelta
        time_deltas = {
            "last_hour": timedelta(hours=1),
            "last_day": timedelta(days=1),
            "last_week": timedelta(days=7),
        }
        delta = time_deltas.get(time_range, timedelta(hours=1))
        since = datetime.now() - delta
        
        # Query logs from events table (structured logs)
        # Note: This is a simplified version. In production, you'd query from a log aggregation system
        # For now, query recent events and filter by event_type pattern
        from src.database.models import Event
        
        result = await self.db_session.execute(
            select(Event)
            .where(Event.timestamp >= since)
            .order_by(Event.timestamp.desc())
            .limit(20)
        )
        events = list(result.scalars().all())
        
        # Filter by level if specified (check event_type or data JSONB)
        filtered_events = []
        for event in events:
            # Simple filtering: check if event_type contains level or if data contains level
            if level in event.event_type.upper() or (
                event.data and level in str(event.data).upper()
            ):
                filtered_events.append(event)
        
        # For now, return a message indicating logs would be queried
        # In production, this would query from Loki, PostgreSQL log table, or similar
        return f"""📋 <b>Logs</b>

Level: {level}
Time Range: {time_range}
Found: {len(filtered_events)} events

<i>Note: Log querying from structured log system not yet fully implemented. This queries from events table as a basic implementation.</i>

To fully implement:
1. Set up log aggregation (Loki, PostgreSQL log table, etc.)
2. Query logs by level and time range
3. Format and return recent log entries"""

    async def _handle_query(self, args: str) -> str:
        """Handle /query command (natural language)."""
        if not args:
            return "❓ Please provide a query. Example: /query What trades did we make on AAPL this week?"
        
        # For now, return a message indicating NL query would be implemented
        # In production, this would use an LLM to convert natural language to SQL/API calls
        return f"""🤖 <b>Natural Language Query</b>

Query: "{args}"

<i>Note: Natural language query processing not yet implemented. This would:</i>
1. Use LLM to convert query to SQL/API calls
2. Execute query against database
3. Format and return results
4. Integrate with Langfuse for tracing

<i>This will be implemented in Phase 9.4 (Operations Portal) with full LLM integration.</i>"""

    def _handle_help(self) -> str:
        """Handle /help command."""
        return """🤖 <b>Trading Bot Commands</b>

<b>/status</b> - System status
  Trading enabled/disabled, circuit breaker, portfolio value, cash, positions

<b>/portfolio</b> - Current holdings
  All positions with P&L, cash balance, sector allocation

<b>/trades [N]</b> - Recent trades
  Last N trades (default: 10, max: 50)
  Example: /trades 20

<b>/metrics</b> - Performance metrics
  Win rate, drawdown, P&L (today/week/month/all-time), alpha vs deposits

<b>/logs [LEVEL] [RANGE]</b> - Query logs
  Filter by level (ERROR, WARNING, INFO)
  Time range: last_hour, last_day, last_week
  Example: /logs ERROR last_hour

<b>/query [QUESTION]</b> - Natural language query
  Ask questions about trades, portfolio, performance
  Example: /query What trades did we make on AAPL this week?

<b>/help</b> - Show this help message"""

