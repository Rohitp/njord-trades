"""
Paper trading broker for testing and simulation.

Simulates order execution locally without connecting to a real broker.
Useful for testing the execution pipeline and backtesting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from src.services.execution.broker import (
    AccountInfo,
    Broker,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from src.services.market_data import MarketDataService
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PaperOrder:
    """Internal order tracking for paper broker."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: float | None
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float | None = None
    filled_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)


class PaperBroker(Broker):
    """
    Paper trading broker for simulation.

    Maintains local state for:
    - Cash balance
    - Positions
    - Order history

    Uses MarketDataService for current prices to simulate fills.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        market_data: MarketDataService | None = None,
        slippage_pct: float = 0.001,  # 0.1% slippage simulation
    ):
        """
        Initialize paper broker.

        Args:
            initial_cash: Starting cash balance
            market_data: Service for getting current prices
            slippage_pct: Simulated slippage percentage (0.001 = 0.1%)
        """
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._market_data = market_data or MarketDataService()
        self._slippage_pct = slippage_pct

    @property
    def name(self) -> str:
        return "paper"

    @property
    def is_paper(self) -> bool:
        return True

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> OrderResult:
        """Submit and immediately execute a paper order."""
        order_id = str(uuid4())[:8]

        log.info(
            "paper_order_submitted",
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            order_type=order_type.value,
        )

        # Validate order
        is_valid, error = await self.validate_order(symbol, side, quantity)
        if not is_valid:
            log.warning("paper_order_rejected", order_id=order_id, error=error)
            return OrderResult(
                success=False,
                broker_order_id=order_id,
                status=OrderStatus.REJECTED,
                error_code="VALIDATION_FAILED",
                error_message=error,
            )

        # Get current market price
        try:
            quote = await self._market_data.get_quote(symbol)
            market_price = quote.price
        except Exception as e:
            log.error("paper_order_price_fetch_failed", order_id=order_id, error=str(e))
            return OrderResult(
                success=False,
                broker_order_id=order_id,
                status=OrderStatus.REJECTED,
                error_code="PRICE_UNAVAILABLE",
                error_message=f"Could not get price for {symbol}: {e}",
            )

        # Apply slippage (worse price for us)
        if side == OrderSide.BUY:
            fill_price = market_price * (1 + self._slippage_pct)
        else:
            fill_price = market_price * (1 - self._slippage_pct)

        # Check limit price for limit orders
        if order_type == OrderType.LIMIT and limit_price is not None:
            if side == OrderSide.BUY and fill_price > limit_price:
                # Would fill above limit - reject for now (real broker would queue)
                return OrderResult(
                    success=False,
                    broker_order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_code="LIMIT_EXCEEDED",
                    error_message=f"Market price {fill_price:.2f} exceeds limit {limit_price:.2f}",
                )
            elif side == OrderSide.SELL and fill_price < limit_price:
                return OrderResult(
                    success=False,
                    broker_order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_code="LIMIT_EXCEEDED",
                    error_message=f"Market price {fill_price:.2f} below limit {limit_price:.2f}",
                )

        # Calculate order value
        order_value = fill_price * quantity

        # Execute the order
        if side == OrderSide.BUY:
            if order_value > self._cash:
                return OrderResult(
                    success=False,
                    broker_order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_code="INSUFFICIENT_FUNDS",
                    error_message=f"Order value {order_value:.2f} exceeds cash {self._cash:.2f}",
                )

            # Deduct cash
            self._cash -= order_value

            # Update position
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_cost = (pos.avg_entry_price * pos.quantity) + order_value
                pos.quantity += quantity
                if pos.quantity > 0:  # Safety check for division by zero
                    pos.avg_entry_price = total_cost / pos.quantity
                else:
                    # Should not happen, but handle edge case
                    pos.avg_entry_price = fill_price
                pos.market_value = pos.quantity * fill_price
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=order_value,
                    avg_entry_price=fill_price,
                )

        else:  # SELL
            pos = self._positions.get(symbol)
            if pos is None or pos.quantity < quantity:
                return OrderResult(
                    success=False,
                    broker_order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_code="INSUFFICIENT_SHARES",
                    error_message=f"Cannot sell {quantity} shares of {symbol}",
                )

            # Add cash
            self._cash += order_value

            # Update position
            pos.quantity -= quantity
            if pos.quantity == 0:
                del self._positions[symbol]
            else:
                pos.market_value = pos.quantity * fill_price

        # Record the order
        now = datetime.now()
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=fill_price,
            filled_at=now,
        )
        self._orders[order_id] = order

        log.info(
            "paper_order_filled",
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            fill_price=fill_price,
            cash_remaining=self._cash,
        )

        return OrderResult(
            success=True,
            broker_order_id=order_id,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=fill_price,
            filled_at=now,
        )

    async def get_order_status(self, broker_order_id: str) -> OrderResult:
        """Get status of a paper order."""
        order = self._orders.get(broker_order_id)
        if order is None:
            return OrderResult(
                success=False,
                broker_order_id=broker_order_id,
                status=OrderStatus.REJECTED,
                error_code="ORDER_NOT_FOUND",
                error_message=f"Order {broker_order_id} not found",
            )

        return OrderResult(
            success=True,
            broker_order_id=order.order_id,
            status=order.status,
            filled_quantity=order.filled_quantity,
            filled_price=order.filled_price,
            filled_at=order.filled_at,
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a paper order (no-op since orders fill immediately)."""
        order = self._orders.get(broker_order_id)
        if order is None:
            return False

        # Paper orders fill immediately, so cancellation is a no-op
        if order.status == OrderStatus.FILLED:
            log.warning("paper_order_already_filled", order_id=broker_order_id)
            return False

        order.status = OrderStatus.CANCELLED
        return True

    async def get_position(self, symbol: str) -> Position | None:
        """Get current position in a symbol."""
        return self._positions.get(symbol)

    async def get_positions(self) -> list[Position]:
        """Get all current positions."""
        return list(self._positions.values())

    async def get_account(self) -> AccountInfo:
        """Get account information."""
        # Calculate portfolio value
        portfolio_value = self._cash
        for pos in self._positions.values():
            portfolio_value += pos.market_value

        return AccountInfo(
            cash=self._cash,
            portfolio_value=portfolio_value,
            buying_power=self._cash,  # Simplified - no margin
            currency="USD",
        )

    async def refresh_positions(self) -> None:
        """Update position market values with current prices."""
        for symbol, pos in self._positions.items():
            try:
                quote = await self._market_data.get_quote(symbol)
                pos.market_value = pos.quantity * quote.price
                pos.unrealized_pnl = pos.market_value - (pos.avg_entry_price * pos.quantity)
                pos.unrealized_pnl_pct = pos.unrealized_pnl / (pos.avg_entry_price * pos.quantity) if pos.quantity > 0 else 0
            except Exception as e:
                log.warning("position_refresh_failed", symbol=symbol, error=str(e))

    def reset(self) -> None:
        """Reset broker to initial state (for testing)."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._orders.clear()
