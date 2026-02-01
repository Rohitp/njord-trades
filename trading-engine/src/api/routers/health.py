"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db
from src.utils.logging import get_logger

router = APIRouter(tags=["Health"])
log = get_logger(__name__)


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_db)) -> dict:
    """
    Health check endpoint that verifies database connectivity.
    """
    try:
        await session.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        log.error("health_check_failed", error=str(e))
        db_status = f"unhealthy: {e}"

    status = "ok" if db_status == "healthy" else "degraded"
    log.debug("health_check", status=status, database=db_status)

    return {
        "status": status,
        "database": db_status,
    }

