"""Strategy endpoints."""

from uuid import UUID as UUIDType

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.api.schemas import (
    StrategyCreate,
    StrategyListResponse,
    StrategyResponse,
    StrategyUpdate,
)
from src.database.models import Strategy
from src.utils.logging import get_logger

router = APIRouter(prefix="/api/strategies", tags=["Strategies"])
log = get_logger(__name__)


@router.post("", response_model=StrategyResponse, status_code=201)
async def create_strategy(
    strategy: StrategyCreate,
    session: AsyncSession = Depends(get_db),
) -> Strategy:
    """Create a new trading strategy."""
    # Check if name already exists
    existing = await session.execute(
        select(Strategy).where(Strategy.name == strategy.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Strategy '{strategy.name}' already exists")

    db_strategy = Strategy(
        name=strategy.name,
        description=strategy.description,
        allocation=strategy.allocation,
        capital=strategy.capital,
    )
    session.add(db_strategy)
    log.info("strategy_created", name=strategy.name, allocation=strategy.allocation)
    return db_strategy


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    status: str | None = Query(None, description="Filter by status"),
    session: AsyncSession = Depends(get_db),
) -> StrategyListResponse:
    """List all strategies."""
    query = select(Strategy).order_by(Strategy.name)

    if status:
        query = query.where(Strategy.status == status.upper())

    result = await session.execute(query)
    strategies = list(result.scalars().all())

    return StrategyListResponse(strategies=strategies, total=len(strategies))


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    session: AsyncSession = Depends(get_db),
) -> Strategy:
    """Get a strategy by ID."""
    try:
        uuid = UUIDType(strategy_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid strategy ID format")

    result = await session.execute(select(Strategy).where(Strategy.id == uuid))
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    return strategy


@router.patch("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str,
    update: StrategyUpdate,
    session: AsyncSession = Depends(get_db),
) -> Strategy:
    """Update a strategy."""
    try:
        uuid = UUIDType(strategy_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid strategy ID format")

    result = await session.execute(select(Strategy).where(Strategy.id == uuid))
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Update only provided fields
    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(strategy, field, value)

    log.info("strategy_updated", strategy_id=strategy_id, updates=update_data)
    return strategy


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(
    strategy_id: str,
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete a strategy."""
    try:
        uuid = UUIDType(strategy_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid strategy ID format")

    result = await session.execute(select(Strategy).where(Strategy.id == uuid))
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    await session.delete(strategy)
    log.info("strategy_deleted", strategy_id=strategy_id, name=strategy.name)
