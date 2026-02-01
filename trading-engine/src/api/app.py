from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_session

app = FastAPI(title="Trading Engine", version="0.1.0")


@app.get("/health")
async def health_check(session: AsyncSession = Depends(get_session)) -> dict:
    """
    Health check endpoint that verifies database connectivity.
    """
    try:
        await session.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "database": db_status,
    }
