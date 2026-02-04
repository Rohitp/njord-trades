"""
Circuit breaker service for risk management.

Monitors portfolio health and halts trading when thresholds are breached:
- Drawdown exceeds 20% from peak
- 10 consecutive losing trades
- Negative Sharpe ratio for 30 days

The circuit breaker is checked before each trading cycle and can only be
reset manually or via auto-resume conditions.
"""

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.models import PortfolioState, SystemState, Trade, TradeOutcome
from src.utils.logging import get_logger
from src.utils.metrics import (
    circuit_breaker_triggers,
    consecutive_losses as consecutive_losses_gauge,
    current_drawdown,
)

log = get_logger(__name__)


class CircuitBreakerService:
    """
    Evaluates circuit breaker conditions and activates when thresholds breach.

    Conditions that trigger the circuit breaker:
    1. Drawdown > 20% from peak portfolio value
    2. 10 consecutive losing trades
    3. Negative Sharpe ratio for 30 days (not yet implemented)
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def evaluate(self) -> tuple[bool, str | None]:
        """
        Evaluate all circuit breaker conditions.

        Returns:
            (should_trigger, reason) - True if circuit breaker should activate
        """
        # Check drawdown
        triggered, reason = await self._check_drawdown()
        if triggered:
            return True, reason

        # Check consecutive losses
        triggered, reason = await self._check_consecutive_losses()
        if triggered:
            return True, reason

        # TODO: Check Sharpe ratio (requires historical returns calculation)

        return False, None

    async def evaluate_and_update(self) -> bool:
        """
        Evaluate conditions and update SystemState if triggered.

        Call this after each trade to check if circuit breaker should activate.

        Returns:
            True if circuit breaker was activated
        """
        should_trigger, reason = await self.evaluate()

        if should_trigger:
            await self._activate(reason)
            return True

        return False

    async def _check_drawdown(self) -> tuple[bool, str | None]:
        """Check if drawdown exceeds threshold."""
        stmt = select(PortfolioState).where(PortfolioState.id == 1)
        result = await self.db_session.execute(stmt)
        portfolio = result.scalar_one_or_none()

        if portfolio is None:
            return False, None

        if portfolio.peak_value <= 0:
            return False, None

        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value

        # Update metrics
        current_drawdown.set(drawdown * 100)

        if drawdown >= settings.trading.drawdown_halt_pct:
            reason = f"Drawdown {drawdown:.1%} exceeds {settings.trading.drawdown_halt_pct:.0%} threshold"
            log.warning("circuit_breaker_drawdown", drawdown=drawdown, threshold=settings.trading.drawdown_halt_pct)
            return True, reason

        return False, None

    async def _check_consecutive_losses(self) -> tuple[bool, str | None]:
        """Check if consecutive losses exceed threshold."""
        # Get recent trades ordered by date descending
        stmt = (
            select(Trade)
            .where(Trade.outcome.isnot(None))
            .order_by(Trade.created_at.desc())
            .limit(settings.trading.consecutive_loss_halt + 1)
        )
        result = await self.db_session.execute(stmt)
        trades = result.scalars().all()

        if not trades:
            return False, None

        # Count consecutive losses from most recent
        consecutive = 0
        for trade in trades:
            if trade.outcome == TradeOutcome.LOSS.value:
                consecutive += 1
            else:
                break

        # Update metrics
        consecutive_losses_gauge.set(consecutive)

        if consecutive >= settings.trading.consecutive_loss_halt:
            reason = f"{consecutive} consecutive losses exceeds {settings.trading.consecutive_loss_halt} threshold"
            log.warning("circuit_breaker_losses", consecutive=consecutive, threshold=settings.trading.consecutive_loss_halt)
            return True, reason

        return False, None

    async def _activate(self, reason: str) -> None:
        """Activate the circuit breaker."""
        stmt = select(SystemState).where(SystemState.id == 1)
        result = await self.db_session.execute(stmt)
        system_state = result.scalar_one_or_none()

        if system_state is None:
            system_state = SystemState(
                id=1,
                trading_enabled=True,
                circuit_breaker_active=True,
                circuit_breaker_reason=reason,
            )
            self.db_session.add(system_state)
        else:
            if system_state.circuit_breaker_active:
                # Already active
                return

            system_state.circuit_breaker_active = True
            system_state.circuit_breaker_reason = reason

        await self.db_session.commit()

        # Record metrics
        if "drawdown" in reason.lower():
            circuit_breaker_triggers.labels(reason="drawdown").inc()
        elif "consecutive" in reason.lower():
            circuit_breaker_triggers.labels(reason="consecutive_losses").inc()
        else:
            circuit_breaker_triggers.labels(reason="other").inc()

        log.error("circuit_breaker_activated", reason=reason)

    async def reset(self, reason: str = "Manual reset") -> bool:
        """
        Manually reset the circuit breaker.

        Args:
            reason: Why the circuit breaker is being reset

        Returns:
            True if reset was successful
        """
        stmt = select(SystemState).where(SystemState.id == 1)
        result = await self.db_session.execute(stmt)
        system_state = result.scalar_one_or_none()

        if system_state is None or not system_state.circuit_breaker_active:
            return False

        system_state.circuit_breaker_active = False
        system_state.circuit_breaker_reason = None

        await self.db_session.commit()

        log.info("circuit_breaker_reset", reason=reason)
        return True

    async def check_auto_resume(self) -> bool:
        """
        Check if auto-resume conditions are met.

        Auto-resume conditions:
        - Drawdown recovered to < 15% (was > 20%)
        - 3 consecutive wins after loss-triggered halt
        - Sharpe > 0.3 for 7 days after Sharpe-triggered halt

        Returns:
            True if auto-resumed
        """
        stmt = select(SystemState).where(SystemState.id == 1)
        result = await self.db_session.execute(stmt)
        system_state = result.scalar_one_or_none()

        if system_state is None or not system_state.circuit_breaker_active:
            return False

        reason = system_state.circuit_breaker_reason or ""

        # Check drawdown recovery
        if "drawdown" in reason.lower():
            portfolio_stmt = select(PortfolioState).where(PortfolioState.id == 1)
            portfolio_result = await self.db_session.execute(portfolio_stmt)
            portfolio = portfolio_result.scalar_one_or_none()

            if portfolio and portfolio.peak_value > 0:
                current_drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
                if current_drawdown < settings.trading.drawdown_resume_pct:
                    await self.reset(f"Drawdown recovered to {current_drawdown:.1%}")
                    return True

        # Check win streak after loss halt
        if "consecutive" in reason.lower():
            trades_stmt = (
                select(Trade)
                .where(Trade.outcome.isnot(None))
                .order_by(Trade.created_at.desc())
                .limit(settings.trading.win_streak_resume)
            )
            trades_result = await self.db_session.execute(trades_stmt)
            trades = trades_result.scalars().all()

            if len(trades) >= settings.trading.win_streak_resume:
                all_wins = all(t.outcome == TradeOutcome.WIN.value for t in trades)
                if all_wins:
                    await self.reset(f"{settings.trading.win_streak_resume} consecutive wins")
                    return True

        return False


async def evaluate_circuit_breaker(db_session: AsyncSession) -> bool:
    """
    Convenience function to evaluate circuit breaker after a trade.

    Returns:
        True if circuit breaker was activated
    """
    service = CircuitBreakerService(db_session)
    return await service.evaluate_and_update()
