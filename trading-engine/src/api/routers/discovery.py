"""Discovery and picker performance endpoints."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import (
    ABTestMetricsResponse,
    DiscoveryPerformanceResponse,
    PickerPerformanceResponse,
)
from src.services.discovery.paper_tracker import calculate_ab_test_metrics
from src.services.discovery.performance import PickerPerformance, analyze_picker_performance
from src.services.discovery.service import SymbolDiscoveryService
from src.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["Discovery"])


class DiscoveryRunRequest(BaseModel):
    """Request to run a discovery cycle."""
    
    update_watchlist: bool = True
    """If True, update the watchlist with discovered symbols."""
    
    timeout_seconds: int = 300
    """Maximum time to wait for discovery cycle (default: 300 = 5 minutes)."""
    
    max_symbols: int = 500
    """Maximum number of symbols to process in MetricPicker (default: 500)."""


class DiscoveryRunResponse(BaseModel):
    """Response from running a discovery cycle."""
    
    success: bool
    discovered_count: int
    suggestions_count: int
    ensemble_count: int
    watchlist_updates: int
    message: str
    completed_at: datetime


@router.post("/run", response_model=DiscoveryRunResponse)
async def run_discovery_cycle(
    request: DiscoveryRunRequest,
    session: AsyncSession = Depends(get_db),
) -> DiscoveryRunResponse:
    """
    Manually trigger a symbol discovery cycle.
    
    Runs all enabled pickers (Metric, Fuzzy, LLM) in parallel, combines results,
    and optionally updates the watchlist with top-ranked symbols.
    
    This is useful for:
    - Initial setup when there are no watchlist symbols
    - Manual discovery outside of scheduled cycles
    - Testing picker configurations
    
    Returns:
        Summary of discovery results including counts of discovered symbols,
        picker suggestions, ensemble results, and watchlist updates.
    """
    start_time = datetime.now()
    
    try:
        log.info("discovery_manual_trigger", update_watchlist=request.update_watchlist)
        
        service = SymbolDiscoveryService(db_session=session)
        
        # Build context (can include portfolio state, market conditions, etc.)
        context = {
            "max_symbols": request.max_symbols,  # Limit symbols processed
            # Could add portfolio positions, market conditions, etc. in the future
        }
        
        result = await service.run_discovery_cycle(
            context=context,
            update_watchlist=request.update_watchlist,
            timeout_seconds=request.timeout_seconds,
        )
        
        completed_at = datetime.now()
        duration = (completed_at - start_time).total_seconds()
        
        log.info(
            "discovery_manual_complete",
            discovered_count=len(result["discovered_symbols"]),
            suggestions_count=len(result["picker_suggestions"]),
            ensemble_count=len(result["ensemble_results"]),
            watchlist_updates=result["watchlist_updates"],
            duration_seconds=duration,
        )
        
        return DiscoveryRunResponse(
            success=True,
            discovered_count=len(result["discovered_symbols"]),
            suggestions_count=len(result["picker_suggestions"]),
            ensemble_count=len(result["ensemble_results"]),
            watchlist_updates=result["watchlist_updates"],
            message=f"Discovery cycle completed successfully. Found {len(result['ensemble_results'])} symbols.",
            completed_at=completed_at,
        )
        
    except Exception as e:
        log.error(
            "discovery_manual_failed",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Discovery cycle failed: {str(e)}"
        )


@router.get("/performance", response_model=DiscoveryPerformanceResponse)
async def get_picker_performance(
    picker_name: str | None = Query(None, description="Filter by specific picker name"),
    min_suggestions: int = Query(10, ge=1, le=1000, description="Minimum suggestions required to include picker"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of pickers to return (pagination)"),
    offset: int = Query(0, ge=0, description="Number of pickers to skip (pagination)"),
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
        Includes pagination support for large datasets.
    """
    performances = await analyze_picker_performance(
        session=session,
        picker_name=picker_name,
        min_suggestions=min_suggestions,
    )

    # Apply pagination
    total_pickers = len(performances)
    paginated_performances = performances[offset : offset + limit]

    # Convert to response models
    picker_responses = [
        PickerPerformanceResponse(
            picker_name=p.picker_name,
            total_suggestions=p.total_suggestions,
            suggestions_with_returns=p.suggestions_with_returns,
            pending_suggestions=p.pending_suggestions,
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
        for p in paginated_performances
    ]

    return DiscoveryPerformanceResponse(
        pickers=picker_responses,
        total_pickers=total_pickers,
        min_suggestions=min_suggestions,
        limit=limit,
        offset=offset,
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

