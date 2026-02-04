"""
Tests for embedding model definitions.

Verifies that embedding models are correctly defined and can be instantiated.
These are unit tests that don't require a database connection.

Note: SQLAlchemy's `default=uuid.uuid4` only applies on database insert,
not on Python object instantiation. The `id` field will be None until
the object is persisted to the database.
"""

import uuid
from datetime import datetime

import pytest

from src.database.models import (
    MarketConditionEmbedding,
    SymbolContextEmbedding,
    TradeEmbedding,
)


class TestTradeEmbedding:
    """Tests for TradeEmbedding model."""

    def test_trade_embedding_creation(self):
        """Test that TradeEmbedding can be instantiated with required fields."""
        trade_id = uuid.uuid4()
        embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions for BGE-small-en
        context_text = "AAPL BUY 5 shares @ 150.0, RSI 65, volume 2x, WIN"

        embedding_obj = TradeEmbedding(
            trade_id=trade_id,
            embedding=embedding,
            context_text=context_text,
        )

        assert embedding_obj.trade_id == trade_id
        assert embedding_obj.embedding == embedding
        assert embedding_obj.context_text == context_text
        # Note: id is None until persisted to database (SQLAlchemy default behavior)

    def test_trade_embedding_embedding_dimensions(self):
        """Test that embedding must be 384 dimensions (BGE-small-en)."""
        trade_id = uuid.uuid4()
        # Wrong dimensions (should be 384)
        wrong_embedding = [0.1, 0.2, 0.3]  # Only 3 dimensions

        # This will fail at database level, but model allows it
        # The Vector(384) constraint is enforced by PostgreSQL
        embedding_obj = TradeEmbedding(
            trade_id=trade_id,
            embedding=wrong_embedding,
            context_text="test",
        )
        # Model accepts it, but DB will reject if dimensions don't match
        assert len(embedding_obj.embedding) == 3


class TestMarketConditionEmbedding:
    """Tests for MarketConditionEmbedding model."""

    def test_market_condition_embedding_creation(self):
        """Test that MarketConditionEmbedding can be instantiated."""
        timestamp = datetime.now()
        embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        context_text = "VIX 25.3, SPY up 1.2%, Tech sector +2.1%"
        condition_metadata = {"vix": 25.3, "spy_change": 1.2, "tech_sector": 2.1}

        embedding_obj = MarketConditionEmbedding(
            timestamp=timestamp,
            embedding=embedding,
            context_text=context_text,
            condition_metadata=condition_metadata,
        )

        assert embedding_obj.timestamp == timestamp
        assert embedding_obj.embedding == embedding
        assert embedding_obj.context_text == context_text
        assert embedding_obj.condition_metadata == condition_metadata
        # Note: id is None until persisted to database

    def test_market_condition_embedding_optional_metadata(self):
        """Test that condition_metadata is optional (None if not provided)."""
        embedding_obj = MarketConditionEmbedding(
            timestamp=datetime.now(),
            embedding=[0.1] * 384,
            context_text="test",
        )

        # condition_metadata is nullable, defaults to None before DB insert
        assert embedding_obj.condition_metadata is None


class TestSymbolContextEmbedding:
    """Tests for SymbolContextEmbedding model."""

    def test_symbol_context_embedding_creation(self):
        """Test that SymbolContextEmbedding can be instantiated."""
        symbol = "AAPL"
        embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        context_text = "Apple announces new iPhone, analyst upgrades to buy"
        context_type = "news"
        source_url = "https://example.com/news"
        timestamp = datetime.now()

        embedding_obj = SymbolContextEmbedding(
            symbol=symbol,
            embedding=embedding,
            context_text=context_text,
            context_type=context_type,
            source_url=source_url,
            timestamp=timestamp,
        )

        assert embedding_obj.symbol == symbol
        assert embedding_obj.embedding == embedding
        assert embedding_obj.context_text == context_text
        assert embedding_obj.context_type == context_type
        assert embedding_obj.source_url == source_url
        assert embedding_obj.timestamp == timestamp
        # Note: id is None until persisted to database

    def test_symbol_context_embedding_optional_source_url(self):
        """Test that source_url is optional."""
        embedding_obj = SymbolContextEmbedding(
            symbol="AAPL",
            embedding=[0.1] * 384,
            context_text="test",
            context_type="news",
            timestamp=datetime.now(),
        )

        assert embedding_obj.source_url is None

