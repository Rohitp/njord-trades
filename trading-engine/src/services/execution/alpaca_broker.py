"""
Alpaca broker implementation for real/paper trading.

Uses the Alpaca Trade API for order submission and management.
"""

import asyncio
from datetime import datetime

from src.config import settings
from src.services.execution.broker import (
    AccountInfo,
    Broker,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

log = get_logger(__name__)

# Map Alpaca order statuses to our OrderStatus
ALPACA_STATUS_MAP = {
    "new": OrderStatus.PENDING,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.ACCEPTED,
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.PENDING,
    "pending_replace": OrderStatus.PENDING,
    "stopped": OrderStatus.CANCELLED,
    "suspended": OrderStatus.PENDING,
    "calculated": OrderStatus.PENDING,
}


class AlpacaBroker(Broker):
    """
    Alpaca broker for real/paper trading.

    Requires alpaca-py package and valid API credentials.
    """

    def __init__(self):
        """Initialize Alpaca client."""
        self._trading_client = None
        self._is_paper = settings.alpaca.paper

    def _get_client(self):
        """Lazy initialization of Alpaca client."""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient

                self._trading_client = TradingClient(
                    api_key=settings.alpaca.api_key,
                    secret_key=settings.alpaca.secret_key,
                    paper=settings.alpaca.paper,
                )
                log.info(
                    "alpaca_client_initialized",
                    paper=settings.alpaca.paper,
                )
            except ImportError:
                raise ImportError(
                    "alpaca-py package is required for Alpaca broker. "
                    "Install with: pip install alpaca-py"
                )
        return self._trading_client

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    @retry_with_backoff()
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> OrderResult:
        """Submit order to Alpaca."""
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import OrderType as AlpacaType
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        client = self._get_client()

        log.info(
            "alpaca_order_submitting",
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            order_type=order_type.value,
            limit_price=limit_price,
        )

        try:
            alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

            if order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                if limit_price is None:
                    return OrderResult(
                        success=False,
                        status=OrderStatus.REJECTED,
                        error_code="MISSING_LIMIT_PRICE",
                        error_message="Limit price required for limit orders",
                    )
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )

            # Submit order (wrap sync call in thread executor)
            order = await asyncio.to_thread(client.submit_order, request)

            status = ALPACA_STATUS_MAP.get(order.status.value, OrderStatus.PENDING)

            log.info(
                "alpaca_order_submitted",
                order_id=str(order.id),
                symbol=symbol,
                status=status.value,
            )

            return OrderResult(
                success=True,
                broker_order_id=str(order.id),
                status=status,
                filled_quantity=int(order.filled_qty or 0),
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                filled_at=order.filled_at,
                raw_response={"order_id": str(order.id), "status": order.status.value},
            )

        except Exception as e:
            error_msg = str(e)
            log.error("alpaca_order_failed", symbol=symbol, error=error_msg)

            # Parse common Alpaca errors
            error_code = "UNKNOWN_ERROR"
            if "insufficient" in error_msg.lower():
                error_code = "INSUFFICIENT_FUNDS"
            elif "not found" in error_msg.lower():
                error_code = "SYMBOL_NOT_FOUND"
            elif "market" in error_msg.lower() and "closed" in error_msg.lower():
                error_code = "MARKET_CLOSED"

            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                error_code=error_code,
                error_message=error_msg,
            )

    @retry_with_backoff()
    async def get_order_status(self, broker_order_id: str) -> OrderResult:
        """Get order status from Alpaca."""
        client = self._get_client()

        try:
            order = await asyncio.to_thread(client.get_order_by_id, broker_order_id)
            status = ALPACA_STATUS_MAP.get(order.status.value, OrderStatus.PENDING)

            return OrderResult(
                success=True,
                broker_order_id=str(order.id),
                status=status,
                filled_quantity=int(order.filled_qty or 0),
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                filled_at=order.filled_at,
            )

        except Exception as e:
            log.error("alpaca_order_status_failed", order_id=broker_order_id, error=str(e))
            return OrderResult(
                success=False,
                broker_order_id=broker_order_id,
                status=OrderStatus.REJECTED,
                error_code="ORDER_NOT_FOUND",
                error_message=str(e),
            )

    @retry_with_backoff()
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order on Alpaca."""
        client = self._get_client()

        try:
            await asyncio.to_thread(client.cancel_order_by_id, broker_order_id)
            log.info("alpaca_order_cancelled", order_id=broker_order_id)
            return True
        except Exception as e:
            log.error("alpaca_cancel_failed", order_id=broker_order_id, error=str(e))
            return False

    async def get_position(self, symbol: str) -> Position | None:
        """Get position from Alpaca."""
        client = self._get_client()

        try:
            pos = await asyncio.to_thread(client.get_open_position, symbol)
            return Position(
                symbol=pos.symbol,
                quantity=int(pos.qty),
                market_value=float(pos.market_value),
                avg_entry_price=float(pos.avg_entry_price),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc),
            )
        except Exception:
            # Position doesn't exist
            return None

    async def get_positions(self) -> list[Position]:
        """Get all positions from Alpaca."""
        client = self._get_client()

        try:
            positions = await asyncio.to_thread(client.get_all_positions)
            return [
                Position(
                    symbol=pos.symbol,
                    quantity=int(pos.qty),
                    market_value=float(pos.market_value),
                    avg_entry_price=float(pos.avg_entry_price),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                )
                for pos in positions
            ]
        except Exception as e:
            log.error("alpaca_positions_failed", error=str(e))
            return []

    async def get_account(self) -> AccountInfo:
        """Get account info from Alpaca."""
        client = self._get_client()

        try:
            account = await asyncio.to_thread(client.get_account)
            return AccountInfo(
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                buying_power=float(account.buying_power),
                currency=account.currency or "USD",
            )
        except Exception as e:
            log.error("alpaca_account_failed", error=str(e))
            raise
