"""
Trade execution services.

Provides broker abstraction and implementations for executing trades.

Usage:
    from src.services.execution import ExecutionService, get_broker

    # Get configured broker
    broker = get_broker()

    # Execute trades from cycle results
    executor = ExecutionService(broker=broker, db_session=session)
    results = await executor.execute_decisions(state)
"""

from src.services.execution.broker import Broker, OrderResult, OrderSide, OrderType
from src.services.execution.service import ExecutionService, get_broker

__all__ = [
    "Broker",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "ExecutionService",
    "get_broker",
]
