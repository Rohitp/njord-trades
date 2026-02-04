"""
Tests for pgvector extension setup.

These are integration tests that require a PostgreSQL database with pgvector.
The tests create their own engine to avoid event loop conflicts with pytest-asyncio.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.exc import OperationalError

from src.config import settings


@pytest.fixture
async def db_session():
    """
    Create a fresh database session for testing.

    Creates its own engine within the test's event loop to avoid
    the 'Future attached to a different loop' error.
    """
    engine = create_async_engine(settings.db.url, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with session_factory() as session:
            # Quick connectivity check
            await session.execute(text("SELECT 1"))
            yield session
    except OperationalError as e:
        pytest.skip(f"Database unavailable: {e}")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_pgvector_extension_available(db_session):
    """Test that pgvector extension is available in PostgreSQL."""
    result = await db_session.execute(
        text("SELECT name FROM pg_available_extensions WHERE name = 'vector'")
    )
    row = result.first()
    assert row is not None, (
        "pgvector extension not available. "
        "Ensure you're using pgvector/pgvector:pg15 Docker image."
    )
    assert row.name == "vector"


@pytest.mark.asyncio
async def test_pgvector_extension_can_be_enabled(db_session):
    """Test that pgvector extension can be enabled."""
    # Enable the extension (idempotent)
    await db_session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await db_session.commit()

    # Verify it's enabled
    result = await db_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    row = result.first()
    assert row is not None, "pgvector extension failed to enable"
    assert row.extname == "vector"


@pytest.mark.asyncio
async def test_pgvector_vector_type_works(db_session):
    """Test that vector type can be used after extension is enabled."""
    # Ensure extension is enabled
    await db_session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Create a temporary table with vector column
    await db_session.execute(text("""
        CREATE TEMPORARY TABLE test_vectors (
            id SERIAL PRIMARY KEY,
            embedding vector(3)
        )
    """))

    # Insert a vector
    await db_session.execute(text("""
        INSERT INTO test_vectors (embedding) VALUES ('[1.0, 2.0, 3.0]')
    """))

    # Query it back
    result = await db_session.execute(text("SELECT embedding FROM test_vectors"))
    row = result.first()
    assert row is not None, "Failed to retrieve vector"

    # Verify cosine similarity works
    result = await db_session.execute(text("""
        SELECT embedding <=> '[1.0, 2.0, 3.0]' as distance FROM test_vectors
    """))
    row = result.first()
    assert row is not None
    assert row.distance == 0.0, "Cosine distance to self should be 0"

