"""
Alert service for routing notifications.

Coordinates between different alert providers (Telegram, Email) and
provides a unified interface for sending alerts.
"""

from datetime import datetime
from typing import Optional

from src.services.alerts.telegram import TelegramAlertProvider
from src.utils.logging import get_logger

log = get_logger(__name__)


class AlertService:
    """
    Central alert service for routing notifications.
    
    Supports multiple providers (Telegram, Email) and provides
    alert templates for common scenarios.
    """

    def __init__(
        self,
        telegram_provider: Optional[TelegramAlertProvider] = None,
    ):
        """
        Initialize alert service.
        
        Args:
            telegram_provider: Telegram provider instance (creates default if None)
        """
        self.telegram = telegram_provider or TelegramAlertProvider()

    async def send_circuit_breaker_alert(
        self,
        reason: str,
        drawdown_pct: Optional[float] = None,
        consecutive_losses: Optional[int] = None,
    ) -> bool:
        """
        Send alert when circuit breaker activates.
        
        Args:
            reason: Why the circuit breaker activated
            drawdown_pct: Current drawdown percentage (if applicable)
            consecutive_losses: Number of consecutive losses (if applicable)
            
        Returns:
            True if alert was sent successfully
        """
        title = "Circuit Breaker Activated"
        
        message_parts = [f"<b>Reason:</b> {reason}"]
        
        if drawdown_pct is not None:
            message_parts.append(f"<b>Drawdown:</b> {drawdown_pct:.1f}%")
        
        if consecutive_losses is not None:
            message_parts.append(f"<b>Consecutive Losses:</b> {consecutive_losses}")
        
        message_parts.append("\n⚠️ Trading has been halted. Manual intervention required.")
        
        message = "\n".join(message_parts)
        
        return await self.telegram.send_alert(
            title=title,
            message=message,
            severity="error",
            disable_notification=False,  # Always notify for circuit breaker
        )

    async def send_auto_resume_conditions_met_alert(
        self,
        resume_reason: str,
        drawdown_pct: Optional[float] = None,
    ) -> bool:
        """
        Send alert when auto-resume conditions are met.
        
        Args:
            resume_reason: Why conditions are met
            drawdown_pct: Current drawdown percentage (if applicable)
            
        Returns:
            True if alert was sent successfully
        """
        title = "Auto-Resume Conditions Met"
        
        message_parts = [
            f"<b>Condition:</b> {resume_reason}",
            "",
            "✅ Circuit breaker can be manually resumed via API.",
            "Use: POST /api/system/circuit-breaker/resume",
        ]
        
        if drawdown_pct is not None:
            message_parts.insert(1, f"<b>Current Drawdown:</b> {drawdown_pct:.1f}%")
        
        message = "\n".join(message_parts)
        
        return await self.telegram.send_alert(
            title=title,
            message=message,
            severity="info",
            disable_notification=False,
        )

    async def send_position_change_alert(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        total_value: float,
        min_alert_value: float | None = None,
    ) -> bool:
        """
        Send alert for significant position changes.
        
        Only sends alerts for trades above a minimum value threshold (severity gating).
        This prevents alert spam for small routine trades.
        
        Args:
            symbol: Stock symbol
            action: "BUY" or "SELL"
            quantity: Number of shares
            price: Price per share
            total_value: Total trade value
            min_alert_value: Minimum trade value to trigger alert (defaults to config)
            
        Returns:
            True if alert was sent successfully, False if below threshold
        """
        from src.config import settings
        
        # Severity gating: only alert on significant trades
        threshold = min_alert_value or settings.alerts.min_position_change_alert_value
        if threshold > 0 and total_value < threshold:
            # Trade is below threshold - don't send alert
            return False
        title = f"Position {action}: {symbol}"
        
        message = f"""
<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Quantity:</b> {quantity}
<b>Price:</b> ${price:.2f}
<b>Total Value:</b> ${total_value:.2f}
        """.strip()
        
        return await self.telegram.send_alert(
            title=title,
            message=message,
            severity="info",
            disable_notification=True,  # Silent for routine trades
        )

    async def send_daily_pnl_summary(
        self,
        date: datetime,
        daily_pnl: float,
        daily_pnl_pct: float,
        total_value: float,
        cash: float,
        position_count: int,
    ) -> bool:
        """
        Send daily P&L summary.
        
        Args:
            date: Date of summary
            daily_pnl: Daily profit/loss in dollars
            daily_pnl_pct: Daily profit/loss percentage
            total_value: Total portfolio value
            cash: Available cash
            position_count: Number of open positions
            
        Returns:
            True if alert was sent successfully
        """
        title = f"Daily P&L Summary - {date.strftime('%Y-%m-%d')}"
        
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        
        message = f"""
{pnl_emoji} <b>Daily P&L:</b> {pnl_sign}${daily_pnl:,.2f} ({pnl_sign}{daily_pnl_pct:.2f}%)

<b>Portfolio Value:</b> ${total_value:,.2f}
<b>Cash:</b> ${cash:,.2f}
<b>Open Positions:</b> {position_count}
        """.strip()
        
        return await self.telegram.send_alert(
            title=title,
            message=message,
            severity="info" if daily_pnl >= 0 else "warning",
            disable_notification=True,  # Silent for daily summaries
        )

    async def send_system_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: Optional[dict] = None,
    ) -> bool:
        """
        Send alert for system errors.
        
        Args:
            error_type: Type of error (e.g., "DatabaseError", "APIError")
            error_message: Error message
            context: Additional context dictionary
            
        Returns:
            True if alert was sent successfully
        """
        title = f"System Error: {error_type}"
        
        message_parts = [f"<b>Error:</b> {error_message}"]
        
        if context:
            for key, value in context.items():
                message_parts.append(f"<b>{key}:</b> {value}")
        
        message = "\n".join(message_parts)
        
        return await self.telegram.send_alert(
            title=title,
            message=message,
            severity="error",
            disable_notification=False,  # Always notify for errors
        )

    async def send_test_alert(self) -> bool:
        """
        Send a test alert to verify configuration.
        
        Returns:
            True if alert was sent successfully
        """
        return await self.telegram.send_alert(
            title="Test Alert",
            message="This is a test alert from the trading system. If you receive this, Telegram integration is working correctly!",
            severity="info",
            disable_notification=False,
        )

