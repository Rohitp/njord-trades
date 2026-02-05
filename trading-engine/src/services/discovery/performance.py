"""
Performance analysis service for symbol discovery pickers.

Calculates win rates, average returns, and other metrics to compare
picker performance over time.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import PickerSuggestion
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PickerPerformance:
    """Performance metrics for a single picker."""

    picker_name: str
    total_suggestions: int
    suggestions_with_returns: int

    # Win rates (positive return = win)
    win_rate_1d: Optional[float] = None  # % of suggestions with positive 1d return
    win_rate_5d: Optional[float] = None
    win_rate_20d: Optional[float] = None

    # Average returns
    avg_return_1d: Optional[float] = None  # Average 1d return %
    avg_return_5d: Optional[float] = None
    avg_return_20d: Optional[float] = None

    # Median returns
    median_return_1d: Optional[float] = None
    median_return_5d: Optional[float] = None
    median_return_20d: Optional[float] = None

    # Best/worst returns
    best_return_1d: Optional[float] = None
    best_return_5d: Optional[float] = None
    best_return_20d: Optional[float] = None

    worst_return_1d: Optional[float] = None
    worst_return_5d: Optional[float] = None
    worst_return_20d: Optional[float] = None


async def analyze_picker_performance(
    session: AsyncSession,
    picker_name: Optional[str] = None,
    min_suggestions: int = 10,
) -> list[PickerPerformance]:
    """
    Analyze performance metrics for pickers.

    Args:
        session: Database session
        picker_name: Filter by specific picker (None = all pickers)
        min_suggestions: Minimum number of suggestions required to include picker

    Returns:
        List of PickerPerformance objects, sorted by avg_return_20d (descending)
    """
    # Build query
    query = select(PickerSuggestion)

    if picker_name:
        query = query.where(PickerSuggestion.picker_name == picker_name)

    # Only include suggestions with calculated returns
    query = query.where(PickerSuggestion.calculated_at.isnot(None))

    result = await session.execute(query)
    suggestions = result.scalars().all()

    # Group by picker
    picker_data: dict[str, list[PickerSuggestion]] = defaultdict(list)
    for suggestion in suggestions:
        picker_data[suggestion.picker_name].append(suggestion)

    # Calculate metrics for each picker
    performances = []

    for picker, picker_suggestions in picker_data.items():
        if len(picker_suggestions) < min_suggestions:
            log.debug(
                "picker_performance_insufficient_data",
                picker=picker,
                count=len(picker_suggestions),
                min_required=min_suggestions,
            )
            continue

        # Filter suggestions with actual return values
        suggestions_1d = [s for s in picker_suggestions if s.forward_return_1d is not None]
        suggestions_5d = [s for s in picker_suggestions if s.forward_return_5d is not None]
        suggestions_20d = [s for s in picker_suggestions if s.forward_return_20d is not None]

        performance = PickerPerformance(
            picker_name=picker,
            total_suggestions=len(picker_suggestions),
            suggestions_with_returns=len(suggestions_20d),  # Use 20d as primary metric
        )

        # Calculate 1d metrics
        if suggestions_1d:
            returns_1d = [float(s.forward_return_1d) for s in suggestions_1d]
            wins_1d = sum(1 for r in returns_1d if r > 0)

            performance.win_rate_1d = (wins_1d / len(returns_1d)) * 100
            performance.avg_return_1d = sum(returns_1d) / len(returns_1d)
            performance.median_return_1d = sorted(returns_1d)[len(returns_1d) // 2]
            performance.best_return_1d = max(returns_1d)
            performance.worst_return_1d = min(returns_1d)

        # Calculate 5d metrics
        if suggestions_5d:
            returns_5d = [float(s.forward_return_5d) for s in suggestions_5d]
            wins_5d = sum(1 for r in returns_5d if r > 0)

            performance.win_rate_5d = (wins_5d / len(returns_5d)) * 100
            performance.avg_return_5d = sum(returns_5d) / len(returns_5d)
            performance.median_return_5d = sorted(returns_5d)[len(returns_5d) // 2]
            performance.best_return_5d = max(returns_5d)
            performance.worst_return_5d = min(returns_5d)

        # Calculate 20d metrics
        if suggestions_20d:
            returns_20d = [float(s.forward_return_20d) for s in suggestions_20d]
            wins_20d = sum(1 for r in returns_20d if r > 0)

            performance.win_rate_20d = (wins_20d / len(returns_20d)) * 100
            performance.avg_return_20d = sum(returns_20d) / len(returns_20d)
            performance.median_return_20d = sorted(returns_20d)[len(returns_20d) // 2]
            performance.best_return_20d = max(returns_20d)
            performance.worst_return_20d = min(returns_20d)

        performances.append(performance)

    # Sort by 20d average return (descending), fallback to picker name
    performances.sort(
        key=lambda p: (
            p.avg_return_20d if p.avg_return_20d is not None else float("-inf"),
            p.picker_name,
        ),
        reverse=True,
    )

    return performances

