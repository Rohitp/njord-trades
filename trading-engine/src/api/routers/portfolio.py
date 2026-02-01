"""Portfolio and positions endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import (
    PortfolioStateResponse,
    PositionListResponse,
    PositionResponse,
)
from src.database.models import PortfolioState, Position

router = APIRouter(prefix="/api", tags=["Portfolio"])


@router.get("/portfolio", response_model=PortfolioStateResponse)
async def get_portfolio(
    session: AsyncSession = Depends(get_db),
) -> PortfolioState:
    """Get current portfolio state."""
    result = await session.execute(select(PortfolioState).where(PortfolioState.id == 1))
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not initialized")
    return portfolio


@router.get("/positions", response_model=PositionListResponse)
async def list_positions(
    sector: str | None = Query(None, description="Filter by sector"),
    session: AsyncSession = Depends(get_db),
) -> PositionListResponse:
    """List all current positions."""
    query = select(Position).order_by(Position.symbol)
    count_query = select(func.count()).select_from(Position)

    if sector:
        query = query.where(Position.sector == sector)
        count_query = count_query.where(Position.sector == sector)

    result = await session.execute(query)
    positions = list(result.scalars().all())

    count_result = await session.execute(count_query)
    total = count_result.scalar_one()

    return PositionListResponse(positions=positions, total_count=total)


@router.get("/positions/{symbol}", response_model=PositionResponse)
async def get_position(
    symbol: str,
    session: AsyncSession = Depends(get_db),
) -> Position:
    """Get position for a specific symbol."""
    result = await session.execute(
        select(Position).where(Position.symbol == symbol.upper())
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=404, detail=f"No position for {symbol}")
    return position

