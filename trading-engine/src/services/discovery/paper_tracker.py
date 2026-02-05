"""
Paper trade tracker for comparing hypothetical trades from suggestions vs actual trades.

Simulates what would have happened if we traded based on picker suggestions
and compares performance to actual trades executed by the system.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import PickerSuggestion, Trade, TradeAction, TradeOutcome
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HypotheticalTrade:
    """A hypothetical trade based on a picker suggestion."""

    suggestion_id: str
    symbol: str
    picker_name: str
    suggested_at: datetime
    score: float
    reason: str

    # Simulated trade details
    hypothetical_quantity: int  # Based on suggestion score or fixed amount
    hypothetical_price: float  # Price at suggestion time (would need market data)
    hypothetical_value: float  # quantity * price

    # Forward returns (from suggestion)
    forward_return_1d: Optional[float] = None
    forward_return_5d: Optional[float] = None
    forward_return_20d: Optional[float] = None

    # Calculated P&L (hypothetical)
    hypothetical_pnl: Optional[float] = None
    hypothetical_pnl_pct: Optional[float] = None


@dataclass
class TradeComparison:
    """Comparison between hypothetical and actual trades."""

    symbol: str
    time_window_days: int  # How close in time (e.g., same day, within 5 days)

    # Hypothetical trade
    hypothetical_trade: Optional[HypotheticalTrade] = None

    # Actual trade (if exists)
    actual_trade: Optional[Trade] = None

    # Comparison metrics
    pnl_difference: Optional[float] = None  # actual_pnl - hypothetical_pnl
    return_difference: Optional[float] = None  # actual_return - hypothetical_return
    both_positive: Optional[bool] = None  # Both trades profitable
    both_negative: Optional[bool] = None  # Both trades losing
    diverged: Optional[bool] = None  # One positive, one negative


@dataclass
class ABTestMetrics:
    """A/B test metrics comparing hypothetical vs actual trading."""

    total_hypothetical_trades: int
    total_actual_trades: int
    matched_trades: int  # Trades that can be compared (same symbol, close time)

    # Win rates
    hypothetical_win_rate: Optional[float] = None
    actual_win_rate: Optional[float] = None

    # Average returns
    hypothetical_avg_return: Optional[float] = None
    actual_avg_return: Optional[float] = None

    # Total P&L
    hypothetical_total_pnl: Optional[float] = None
    actual_total_pnl: Optional[float] = None

    # Divergence metrics
    convergence_rate: Optional[float] = None  # % of matched trades where both same direction
    hypothetical_outperformed: int = 0  # Count where hypothetical > actual
    actual_outperformed: int = 0  # Count where actual > hypothetical


async def create_hypothetical_trades(
    session: AsyncSession,
    picker_name: Optional[str] = None,
    min_score: float = 0.0,
    base_quantity: int = 10,  # Fixed quantity per suggestion
    time_window_days: int = 5,  # Only include suggestions from last N days
) -> list[HypotheticalTrade]:
    """
    Create hypothetical trades from picker suggestions.

    Args:
        session: Database session
        picker_name: Filter by picker (None = all pickers)
        min_score: Minimum suggestion score to include
        base_quantity: Fixed quantity to simulate per suggestion
        time_window_days: Only include suggestions from last N days

    Returns:
        List of hypothetical trades
    """
    # Build query
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)

    query = (
        select(PickerSuggestion)
        .where(PickerSuggestion.suggested_at >= cutoff_date)
        .where(PickerSuggestion.score >= min_score)
    )

    if picker_name:
        query = query.where(PickerSuggestion.picker_name == picker_name)

    result = await session.execute(query)
    suggestions = result.scalars().all()

    hypothetical_trades = []

    for suggestion in suggestions:
        # For now, use a fixed quantity based on score
        # In a real system, this would be based on position sizing logic
        quantity = int(base_quantity * suggestion.score) or 1

        # We don't have historical price at suggestion time stored
        # For now, we'll use forward returns to calculate hypothetical P&L
        # In a real implementation, we'd fetch price at suggestion time from market data

        hypothetical_trade = HypotheticalTrade(
            suggestion_id=str(suggestion.id),
            symbol=suggestion.symbol,
            picker_name=suggestion.picker_name,
            suggested_at=suggestion.suggested_at,
            score=float(suggestion.score),
            reason=suggestion.reason,
            hypothetical_quantity=quantity,
            hypothetical_price=0.0,  # Would need market data to fill this
            hypothetical_value=0.0,  # Would need market data to fill this
            forward_return_1d=float(suggestion.forward_return_1d) if suggestion.forward_return_1d else None,
            forward_return_5d=float(suggestion.forward_return_5d) if suggestion.forward_return_5d else None,
            forward_return_20d=float(suggestion.forward_return_20d) if suggestion.forward_return_20d else None,
        )

        # Calculate hypothetical P&L from forward return
        if suggestion.forward_return_20d is not None:
            # Use suggested_price if available, otherwise fall back to fixed notional
            if suggestion.suggested_price and suggestion.suggested_price > 0:
                # Realistic pricing: use actual price at suggestion time
                hypothetical_trade.hypothetical_price = float(suggestion.suggested_price)
                base_value = hypothetical_trade.hypothetical_quantity * hypothetical_trade.hypothetical_price
                hypothetical_trade.hypothetical_value = base_value
                hypothetical_trade.hypothetical_pnl = base_value * (suggestion.forward_return_20d / 100.0)
                hypothetical_trade.hypothetical_pnl_pct = suggestion.forward_return_20d
            else:
                # Fallback: use fixed $1000 notional (for older suggestions without price)
                base_value = 1000.0
                hypothetical_trade.hypothetical_value = base_value
                hypothetical_trade.hypothetical_pnl = base_value * (suggestion.forward_return_20d / 100.0)
                hypothetical_trade.hypothetical_pnl_pct = suggestion.forward_return_20d
                log.debug(
                    "paper_tracker_no_suggested_price",
                    symbol=suggestion.symbol,
                    suggestion_id=str(suggestion.id),
                    message="Using fixed $1000 notional (suggested_price not available)",
                )

        hypothetical_trades.append(hypothetical_trade)

    return hypothetical_trades


async def compare_trades(
    session: AsyncSession,
    hypothetical_trades: list[HypotheticalTrade],
    time_window_hours: int = 24,  # Match trades within N hours
) -> list[TradeComparison]:
    """
    Compare hypothetical trades with actual trades.

    Matches trades by symbol and time proximity.

    Args:
        session: Database session
        hypothetical_trades: List of hypothetical trades to compare
        time_window_hours: Maximum time difference to consider a match

    Returns:
        List of trade comparisons
    """
    if not hypothetical_trades:
        return []

    # Get all actual trades
    result = await session.execute(select(Trade).where(Trade.action == TradeAction.BUY.value))
    actual_trades = result.scalars().all()

    comparisons = []

    for hyp_trade in hypothetical_trades:
        # Find matching actual trades (same symbol, within time window)
        matching_actual = None
        min_time_diff = timedelta(hours=time_window_hours)

        for actual_trade in actual_trades:
            if actual_trade.symbol != hyp_trade.symbol:
                continue

            time_diff = abs(actual_trade.created_at - hyp_trade.suggested_at)
            if time_diff < min_time_diff:
                matching_actual = actual_trade
                min_time_diff = time_diff

        comparison = TradeComparison(
            symbol=hyp_trade.symbol,
            time_window_days=time_window_hours // 24,
            hypothetical_trade=hyp_trade,
            actual_trade=matching_actual,
        )

        # Calculate comparison metrics if both exist
        if matching_actual and hyp_trade.hypothetical_pnl is not None:
            if matching_actual.pnl is not None:
                comparison.pnl_difference = matching_actual.pnl - hyp_trade.hypothetical_pnl

            if matching_actual.pnl_pct is not None and hyp_trade.hypothetical_pnl_pct is not None:
                comparison.return_difference = matching_actual.pnl_pct - hyp_trade.hypothetical_pnl_pct

            # Check if both positive/negative
            hyp_positive = hyp_trade.hypothetical_pnl > 0
            actual_positive = matching_actual.pnl is not None and matching_actual.pnl > 0

            comparison.both_positive = hyp_positive and actual_positive
            comparison.both_negative = not hyp_positive and not actual_positive
            comparison.diverged = hyp_positive != actual_positive

        comparisons.append(comparison)

    return comparisons


async def calculate_ab_test_metrics(
    session: AsyncSession,
    picker_name: Optional[str] = None,
    time_window_days: int = 30,
) -> ABTestMetrics:
    """
    Calculate A/B test metrics comparing hypothetical vs actual trading.

    Args:
        session: Database session
        picker_name: Filter by picker (None = all pickers)
        time_window_days: Time window for analysis

    Returns:
        ABTestMetrics with comparison statistics
    """
    # Get hypothetical trades
    hypothetical_trades = await create_hypothetical_trades(
        session=session,
        picker_name=picker_name,
        time_window_days=time_window_days,
    )

    # Get actual trades
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)
    query = select(Trade).where(Trade.created_at >= cutoff_date)
    result = await session.execute(query)
    actual_trades = result.scalars().all()

    # Compare trades
    comparisons = await compare_trades(session, hypothetical_trades)

    # Calculate metrics
    matched = [c for c in comparisons if c.actual_trade is not None]

    # Hypothetical metrics
    hyp_with_returns = [t for t in hypothetical_trades if t.hypothetical_pnl is not None]
    hyp_wins = sum(1 for t in hyp_with_returns if t.hypothetical_pnl > 0)
    hyp_win_rate = (hyp_wins / len(hyp_with_returns) * 100) if hyp_with_returns else None
    hyp_avg_return = (
        sum(t.hypothetical_pnl_pct for t in hyp_with_returns) / len(hyp_with_returns)
        if hyp_with_returns
        else None
    )
    hyp_total_pnl = sum(t.hypothetical_pnl for t in hyp_with_returns) if hyp_with_returns else None

    # Actual metrics
    actual_with_outcome = [t for t in actual_trades if t.outcome and t.outcome != TradeOutcome.OPEN.value]
    actual_wins = sum(1 for t in actual_with_outcome if t.outcome == TradeOutcome.WIN.value)
    actual_win_rate = (actual_wins / len(actual_with_outcome) * 100) if actual_with_outcome else None
    actual_avg_return = (
        sum(t.pnl_pct for t in actual_with_outcome if t.pnl_pct is not None) / len(actual_with_outcome)
        if actual_with_outcome
        else None
    )
    actual_total_pnl = (
        sum(t.pnl for t in actual_with_outcome if t.pnl is not None) if actual_with_outcome else None
    )

    # Divergence metrics
    convergence_count = sum(1 for c in matched if c.both_positive or c.both_negative)
    convergence_rate = (convergence_count / len(matched) * 100) if matched else None

    hyp_outperformed = sum(1 for c in matched if c.return_difference and c.return_difference < 0)
    actual_outperformed = sum(1 for c in matched if c.return_difference and c.return_difference > 0)

    return ABTestMetrics(
        total_hypothetical_trades=len(hypothetical_trades),
        total_actual_trades=len(actual_trades),
        matched_trades=len(matched),
        hypothetical_win_rate=hyp_win_rate,
        actual_win_rate=actual_win_rate,
        hypothetical_avg_return=hyp_avg_return,
        actual_avg_return=actual_avg_return,
        hypothetical_total_pnl=hyp_total_pnl,
        actual_total_pnl=actual_total_pnl,
        convergence_rate=convergence_rate,
        hypothetical_outperformed=hyp_outperformed,
        actual_outperformed=actual_outperformed,
    )

