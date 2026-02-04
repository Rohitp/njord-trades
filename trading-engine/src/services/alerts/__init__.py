"""Alert service package for notifications via Telegram and Email."""

from src.services.alerts.service import AlertService
from src.services.alerts.telegram import TelegramAlertProvider

__all__ = ["AlertService", "TelegramAlertProvider"]

