"""
Broker abstraction for trade execution.

Defines the protocol that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order status from broker."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderResult:
    """
    Result of an order submission.

    Returned by broker.submit_order() with execution details.
    """
    success: bool
    broker_order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING

    # Execution details (populated on fill)
    filled_quantity: int = 0
    filled_price: float | None = None
    filled_at: datetime | None = None

    # Error details (populated on failure)
    error_code: str | None = None
    error_message: str | None = None

    # Raw broker response for debugging
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state (no more updates expected)."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: int
    market_value: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class AccountInfo:
    """Broker account information."""
    cash: float
    portfolio_value: float
    buying_power: float
    currency: str = "USD"


class Broker(ABC):
    """
    Abstract broker interface for trade execution.

    Implementations:
    - AlpacaBroker: Real trading via Alpaca API
    - PaperBroker: Local simulation for testing
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name for logging."""
        pass

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Whether this is paper trading (simulation)."""
        pass

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> OrderResult:
        """
        Submit an order to the broker.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            side: Buy or sell
            quantity: Number of shares
            order_type: Market or limit order
            limit_price: Required for limit orders

        Returns:
            OrderResult with execution details
        """
        pass

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> OrderResult:
        """
        Get current status of an order.

        Args:
            broker_order_id: Order ID from broker

        Returns:
            OrderResult with current status
        """
        pass

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            broker_order_id: Order ID to cancel

        Returns:
            True if cancellation was submitted successfully
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """
        Get current position in a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position if held, None otherwise
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get all current positions.

        Returns:
            List of all positions
        """
        pass

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with cash, buying power, etc.
        """
        pass

    async def validate_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
    ) -> tuple[bool, str | None]:
        """
        Validate an order before submission.

        Default implementation checks basic constraints.
        Override in subclasses for broker-specific validation.

        Returns:
            (is_valid, error_message)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"

        if not symbol or not symbol.strip():
            return False, "Symbol is required"

        # Check buying power for buys
        if side == OrderSide.BUY:
            account = await self.get_account()
            # Note: We don't have current price here, so this is a basic check
            # Real validation happens at order submission
            if account.buying_power <= 0:
                return False, "Insufficient buying power"

        # Check position for sells
        if side == OrderSide.SELL:
            position = await self.get_position(symbol)
            if position is None or position.quantity < quantity:
                current_qty = position.quantity if position else 0
                return False, f"Insufficient shares (have {current_qty}, need {quantity})"

        return True, None
