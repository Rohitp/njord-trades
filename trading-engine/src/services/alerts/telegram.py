"""
Telegram alert provider.

Sends alerts via Telegram bot API.
"""

import httpx
from typing import Optional

from src.config import settings
from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

log = get_logger(__name__)


class TelegramAlertProvider:
    """
    Sends alerts via Telegram bot API.
    
    Requires:
    - ALERT_TELEGRAM_BOT_TOKEN: Bot token from @BotFather
    - ALERT_TELEGRAM_CHAT_ID: Channel or chat ID (negative for channels)
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """
        Initialize Telegram provider.
        
        Args:
            bot_token: Telegram bot token (default: from config)
            chat_id: Telegram chat/channel ID (default: from config)
        """
        # Use provided values, or fall back to config
        # Empty string is falsy, so we check explicitly for None
        self.bot_token = bot_token if bot_token is not None else settings.alerts.telegram_bot_token
        self.chat_id = chat_id if chat_id is not None else settings.alerts.telegram_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)

    @retry_with_backoff(
        max_retries=3,
        backoff_base=2.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError, httpx.HTTPError),
    )
    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a message via Telegram.
        
        Args:
            text: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown" (default: HTML)
            disable_notification: If True, send silently
            
        Returns:
            True if message was sent successfully
            
        Raises:
            httpx.HTTPError: If API call fails after retries
        """
        if not self.is_configured():
            log.warning("telegram_not_configured", reason="Missing bot_token or chat_id")
            return False

        url = f"{self.base_url}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                if result.get("ok"):
                    log.debug("telegram_message_sent", chat_id=self.chat_id)
                    return True
                else:
                    error = result.get("description", "Unknown error")
                    log.error("telegram_api_error", error=error, response=result)
                    return False

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.error(
                "telegram_send_failed",
                error=str(e),
                chat_id=self.chat_id,
                exc_info=True,
            )
            raise

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a formatted alert message.
        
        Args:
            title: Alert title
            message: Alert message body
            severity: "info", "warning", "error" (affects emoji)
            disable_notification: If True, send silently
            
        Returns:
            True if sent successfully
        """
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🔴",
            "success": "✅",
        }
        
        emoji = emoji_map.get(severity, "ℹ️")
        
        # Format as HTML
        formatted_text = f"<b>{emoji} {title}</b>\n\n{message}"
        
        return await self.send_message(
            text=formatted_text,
            parse_mode="HTML",
            disable_notification=disable_notification,
        )

