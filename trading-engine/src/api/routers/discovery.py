"""Discovery and picker performance endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import (
    ABTestMetricsResponse,
    DiscoveryPerformanceResponse,
    PickerPerformanceResponse,
)
from src.services.discovery.paper_tracker import calculate_ab_test_metrics
from src.services.discovery.performance import PickerPerformance, analyze_picker_performance

router = APIRouter(prefix="/api/discovery", tags=["Discovery"])


@router.get("/performance", response_model=DiscoveryPerformanceResponse)
async def get_picker_performance(
    picker_name: str | None = Query(None, description="Filter by specific picker name"),
    min_suggestions: int = Query(10, ge=1, le=1000, description="Minimum suggestions required to include picker"),
    session: AsyncSession = Depends(get_db),
) -> DiscoveryPerformanceResponse:
    """
    Get performance metrics for symbol discovery pickers.

    Returns win rates, average returns, and other metrics comparing
    picker performance based on forward returns calculated from suggestions.

    Metrics are calculated from PickerSuggestion records that have
    forward returns calculated (1d, 5d, 20d).

    Returns:
        Performance metrics for each picker, sorted by 20d average return (descending).
    """
    performances = await analyze_picker_performance(
        session=session,
        picker_name=picker_name,
        min_suggestions=min_suggestions,
    )

    # Convert to response models
    picker_responses = [
        PickerPerformanceResponse(
            picker_name=p.picker_name,
            total_suggestions=p.total_suggestions,
            suggestions_with_returns=p.suggestions_with_returns,
            win_rate_1d=p.win_rate_1d,
            win_rate_5d=p.win_rate_5d,
            win_rate_20d=p.win_rate_20d,
            avg_return_1d=p.avg_return_1d,
            avg_return_5d=p.avg_return_5d,
            avg_return_20d=p.avg_return_20d,
            median_return_1d=p.median_return_1d,
            median_return_5d=p.median_return_5d,
            median_return_20d=p.median_return_20d,
            best_return_1d=p.best_return_1d,
            best_return_5d=p.best_return_5d,
            best_return_20d=p.best_return_20d,
            worst_return_1d=p.worst_return_1d,
            worst_return_5d=p.worst_return_5d,
            worst_return_20d=p.worst_return_20d,
        )
        for p in performances
    ]

    return DiscoveryPerformanceResponse(
        pickers=picker_responses,
        total_pickers=len(picker_responses),
        min_suggestions=min_suggestions,
    )


@router.get("/ab-test", response_model=ABTestMetricsResponse)
async def get_ab_test_metrics(
    picker_name: str | None = Query(None, description="Filter by specific picker name"),
    time_window_days: int = Query(30, ge=1, le=365, description="Time window for analysis (days)"),
    session: AsyncSession = Depends(get_db),
) -> ABTestMetricsResponse:
    """
    Get A/B test metrics comparing hypothetical trades from suggestions vs actual trades.

    Simulates what would have happened if we traded based on picker suggestions
    and compares performance to actual trades executed by the system.

    Returns:
        A/B test metrics including win rates, average returns, and divergence analysis.
    """
    metrics = await calculate_ab_test_metrics(
        session=session,
        picker_name=picker_name,
        time_window_days=time_window_days,
    )

    return ABTestMetricsResponse(
        total_hypothetical_trades=metrics.total_hypothetical_trades,
        total_actual_trades=metrics.total_actual_trades,
        matched_trades=metrics.matched_trades,
        hypothetical_win_rate=metrics.hypothetical_win_rate,
        actual_win_rate=metrics.actual_win_rate,
        hypothetical_avg_return=metrics.hypothetical_avg_return,
        actual_avg_return=metrics.actual_avg_return,
        hypothetical_total_pnl=metrics.hypothetical_total_pnl,
        actual_total_pnl=metrics.actual_total_pnl,
        convergence_rate=metrics.convergence_rate,
        hypothetical_outperformed=metrics.hypothetical_outperformed,
        actual_outperformed=metrics.actual_outperformed,
    )

