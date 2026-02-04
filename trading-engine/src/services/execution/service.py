"""
Execution service for processing trading decisions.

Takes EXECUTE decisions from the MetaAgent and submits orders to the broker.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.models import CapitalEvent, CapitalEventType, Position as DBPosition, PortfolioState, Trade, TradeAction, TradeStatus
from src.services.execution.broker import Broker, OrderSide, OrderStatus, OrderType
from src.utils.logging import get_logger
from src.utils.metrics import execute_decisions as execute_decisions_counter
from src.workflows.state import Decision, ExecutionResult, FinalDecision, Signal, TradingState

log = get_logger(__name__)


def get_broker() -> Broker:
    """
    Get the configured broker instance.

    Uses Alpaca if configured, otherwise falls back to paper broker.
    """
    # Check for non-empty API keys (empty string is falsy but we're explicit)
    has_alpaca = bool(settings.alpaca.api_key.strip()) and bool(settings.alpaca.secret_key.strip())

    if has_alpaca:
        from src.services.execution.alpaca_broker import AlpacaBroker
        return AlpacaBroker()
    else:
        from src.services.execution.paper_broker import PaperBroker
        log.warning("broker_fallback_to_paper", reason="alpaca_not_configured")
        return PaperBroker(initial_cash=settings.trading.initial_capital)


class ExecutionService:
    """
    Service for executing trading decisions.

    Takes TradingState with EXECUTE decisions and:
    1. Submits orders to the broker
    2. Updates database (trades, positions, portfolio)
    3. Returns ExecutionResults
    """

    def __init__(
        self,
        broker: Broker | None = None,
        db_session: AsyncSession | None = None,
    ):
        """
        Initialize execution service.

        Args:
            broker: Broker to use for order submission
            db_session: Database session for persistence
        """
        self.broker = broker or get_broker()
        self.db_session = db_session

    async def execute_decisions(self, state: TradingState) -> list[ExecutionResult]:
        """
        Execute all EXECUTE decisions from the trading state.

        Args:
            state: TradingState with final_decisions

        Returns:
            List of ExecutionResults
        """
        execute_list = state.get_execute_decisions()

        if not execute_list:
            log.debug("no_execute_decisions", cycle_id=str(state.cycle_id))
            return []

        log.info(
            "executing_decisions",
            cycle_id=str(state.cycle_id),
            decision_count=len(execute_list),
            broker=self.broker.name,
        )

        results = []
        for decision in execute_list:
            result = await self._execute_single(state, decision)
            results.append(result)

            # Add to state
            state.execution_results.append(result)

        # Evaluate circuit breaker after all trades
        if self.db_session is not None:
            await self._evaluate_circuit_breaker()

        return results

    async def _evaluate_circuit_breaker(self) -> None:
        """Evaluate circuit breaker conditions after trades."""
        try:
            from src.services.circuit_breaker import evaluate_circuit_breaker
            triggered = await evaluate_circuit_breaker(self.db_session)
            if triggered:
                log.warning("circuit_breaker_triggered_after_execution")
        except Exception as e:
            # Don't fail execution if circuit breaker check fails
            log.error("circuit_breaker_evaluation_failed", error=str(e))

    async def _execute_single(
        self,
        state: TradingState,
        decision: FinalDecision,
    ) -> ExecutionResult:
        """Execute a single trading decision."""
        # Find the original signal
        signal = state.get_signal(decision.signal_id)
        if signal is None:
            log.error("signal_not_found", signal_id=str(decision.signal_id))
            return ExecutionResult(
                signal_id=decision.signal_id,
                success=False,
                error=f"Signal {decision.signal_id} not found",
            )

        # Map signal action to order side
        if signal.action.value == "BUY":
            side = OrderSide.BUY
        elif signal.action.value == "SELL":
            side = OrderSide.SELL
        else:
            # HOLD signals shouldn't have EXECUTE decisions
            log.warning("hold_signal_with_execute", signal_id=str(signal.id))
            return ExecutionResult(
                signal_id=signal.id,
                success=False,
                error="Cannot execute HOLD signal",
            )

        log.info(
            "submitting_order",
            signal_id=str(signal.id),
            symbol=signal.symbol,
            side=side.value,
            quantity=decision.final_quantity,
        )

        # Submit order to broker
        order_result = await self.broker.submit_order(
            symbol=signal.symbol,
            side=side,
            quantity=decision.final_quantity,
            order_type=OrderType.MARKET,
        )

        # Calculate slippage if filled
        slippage = None
        if order_result.filled_price is not None and signal.price > 0:
            slippage = (order_result.filled_price - signal.price) / signal.price

        # Build execution result
        execution_result = ExecutionResult(
            signal_id=signal.id,
            success=order_result.success and order_result.is_filled,
            symbol=signal.symbol,
            action=signal.action.value,
            quantity=decision.final_quantity,
            requested_price=signal.price,
            fill_price=order_result.filled_price,
            slippage=slippage,
            broker_order_id=order_result.broker_order_id,
            error=order_result.error_message,
        )

        # Record metrics
        if execution_result.success:
            execute_decisions_counter.labels(symbol=signal.symbol).inc()

        log.info(
            "order_result",
            signal_id=str(signal.id),
            success=execution_result.success,
            fill_price=execution_result.fill_price,
            broker_order_id=execution_result.broker_order_id,
            error=execution_result.error,
        )

        # Persist to database if session available
        if self.db_session is not None and execution_result.success:
            execution_result.trade_id = await self._persist_trade(
                state, signal, decision, execution_result
            )

        return execution_result

    async def _persist_trade(
        self,
        state: TradingState,
        signal: Signal,
        decision: FinalDecision,
        result: ExecutionResult,
    ) -> str | None:
        """
        Persist trade to database.

        Updates:
        - trades table (new record)
        - positions table (update quantity/cost)
        - portfolio_state table (update cash)
        - capital_events table (realized P&L for sells)

        Returns:
            Trade ID if persisted, None on error
        """
        if self.db_session is None:
            return None

        try:
            trade_id = uuid4()
            trade_value = (result.fill_price or signal.price) * result.quantity

            # Find risk assessment for this signal
            risk_assessment = next(
                (ra for ra in state.risk_assessments if ra.signal_id == signal.id),
                None
            )

            # Create trade record
            trade = Trade(
                id=trade_id,
                symbol=signal.symbol,
                action=signal.action.value,
                quantity=result.quantity,
                price=result.fill_price or signal.price,
                total_value=trade_value,
                status=TradeStatus.FILLED.value,
                signal_confidence=signal.confidence,
                risk_score=risk_assessment.risk_score if risk_assessment else 0.0,
                broker_order_id=result.broker_order_id,
                fill_price=result.fill_price,
                slippage=result.slippage,
            )
            self.db_session.add(trade)

            # Update position
            await self._update_position(signal, result, trade_value)

            # Update portfolio state
            await self._update_portfolio(signal, trade_value)

            await self.db_session.commit()

            log.info("trade_persisted", trade_id=str(trade_id), symbol=signal.symbol)
            return trade_id

        except Exception as e:
            log.error("trade_persistence_failed", error=str(e), exc_info=True)
            await self.db_session.rollback()
            return None

    async def _update_position(
        self,
        signal: Signal,
        result: ExecutionResult,
        trade_value: float,
    ) -> None:
        """Update position in database."""
        # Find existing position
        stmt = select(DBPosition).where(DBPosition.symbol == signal.symbol)
        db_result = await self.db_session.execute(stmt)
        position = db_result.scalar_one_or_none()

        fill_price = result.fill_price or signal.price

        if signal.action.value == "BUY":
            if position is None:
                # Create new position
                position = DBPosition(
                    symbol=signal.symbol,
                    quantity=result.quantity,
                    avg_cost=fill_price,
                    current_price=fill_price,
                    current_value=trade_value,
                    sector=None,  # Could fetch from market data
                )
                self.db_session.add(position)
            else:
                # Update existing position (average up/down)
                total_cost = (position.avg_cost * position.quantity) + trade_value
                position.quantity += result.quantity
                position.avg_cost = total_cost / position.quantity
                position.current_price = fill_price
                position.current_value = position.quantity * fill_price

        else:  # SELL
            if position is None:
                log.warning("sell_without_position", symbol=signal.symbol)
                return

            # Calculate realized P&L
            cost_basis = position.avg_cost * result.quantity
            realized_pnl = trade_value - cost_basis

            # Create capital event for realized P&L
            event_type = CapitalEventType.REALIZED_PROFIT if realized_pnl > 0 else CapitalEventType.REALIZED_LOSS
            capital_event = CapitalEvent(
                event_type=event_type.value,
                amount=abs(realized_pnl),
                balance_after=0.0,  # Will be updated below
                description=f"Closed {result.quantity} shares of {signal.symbol}",
            )
            self.db_session.add(capital_event)

            # Update position
            position.quantity -= result.quantity
            if position.quantity <= 0:
                await self.db_session.delete(position)
            else:
                position.current_price = fill_price
                position.current_value = position.quantity * fill_price

    async def _update_portfolio(
        self,
        signal: Signal,
        trade_value: float,
    ) -> None:
        """Update portfolio state in database."""
        # Get portfolio state (single row)
        stmt = select(PortfolioState).where(PortfolioState.id == 1)
        db_result = await self.db_session.execute(stmt)
        portfolio = db_result.scalar_one_or_none()

        if portfolio is None:
            # Create initial portfolio state
            portfolio = PortfolioState(
                id=1,
                cash=settings.trading.initial_capital,
                total_value=settings.trading.initial_capital,
                deployed_capital=0.0,
                peak_value=settings.trading.initial_capital,
            )
            self.db_session.add(portfolio)

        if signal.action.value == "BUY":
            portfolio.cash -= trade_value
            portfolio.deployed_capital += trade_value
        else:  # SELL
            portfolio.cash += trade_value
            portfolio.deployed_capital -= trade_value

        # Recalculate total value (cash + positions)
        positions_stmt = select(DBPosition)
        positions_result = await self.db_session.execute(positions_stmt)
        positions = positions_result.scalars().all()

        portfolio.total_value = portfolio.cash + sum(p.current_value for p in positions)

        # Update peak if new high
        if portfolio.total_value > portfolio.peak_value:
            portfolio.peak_value = portfolio.total_value
