"""
Tests for market condition embedding service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from src.database.models import MarketConditionEmbedding
from src.services.embeddings.market_condition import MarketConditionService


class TestMarketConditionService:
    """Tests for MarketConditionService."""

    @pytest.mark.asyncio
    async def test_embed_market_condition_creates_embedding(self):
        """Test that embedding is created for market conditions."""
        timestamp = datetime.now()

        # Mock market data
        mock_vix_quote = MagicMock()
        mock_vix_quote.price = 22.5

        mock_spy_indicators = MagicMock()
        mock_spy_indicators.price = 450.0
        mock_spy_indicators.sma_200 = 440.0

        mock_sector_quotes = [
            MagicMock(symbol="XLK", price=180.0),
            MagicMock(symbol="XLF", price=40.0),
        ]

        with patch("src.services.embeddings.market_condition.MarketDataService") as mock_market, \
             patch("src.services.embeddings.market_condition.EmbeddingService") as mock_emb_svc:
            
            mock_market_service = MagicMock()
            mock_market_service.get_quote = AsyncMock(return_value=mock_vix_quote)
            mock_market_service.get_technical_indicators = AsyncMock(return_value=mock_spy_indicators)
            mock_market_service.get_quotes = AsyncMock(return_value=mock_sector_quotes)
            mock_market.return_value = mock_market_service

            mock_embedding = [0.1] * 384
            mock_emb_service = MagicMock()
            mock_emb_service.embed_text = AsyncMock(return_value=mock_embedding)
            mock_emb_svc.return_value = mock_emb_service

            # Mock database session
            mock_session = MagicMock()
            mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: None))

            service = MarketConditionService(
                embedding_service=mock_emb_service,
                market_data_service=mock_market_service,
            )
            result = await service.embed_market_condition(timestamp, mock_session)

            assert result is not None
            assert result.timestamp == timestamp
            assert result.embedding == mock_embedding
            assert "VIX" in result.context_text
            assert "SPY" in result.context_text
            assert result.condition_metadata["vix"] == 22.5
            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_market_condition_skips_existing(self):
        """Test that existing embeddings are not recreated."""
        timestamp = datetime.now()

        existing_embedding = MarketConditionEmbedding(
            id=uuid4(),
            timestamp=timestamp,
            embedding=[0.1] * 384,
            context_text="Existing",
            condition_metadata={},
        )

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = lambda: existing_embedding
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = MarketConditionService()
        result = await service.embed_market_condition(timestamp, mock_session)

        assert result == existing_embedding
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_format_context(self):
        """Test that context is formatted correctly."""
        condition_data = {
            "vix": 22.5,
            "spy_price": 450.0,
            "spy_sma_200": 440.0,
            "spy_above_sma_200": True,
            "spy_trend_pct": 2.27,
            "sector_performance": {
                "XLK": {"name": "Technology", "price": 180.0},
                "XLF": {"name": "Financials", "price": 40.0},
            },
        }

        service = MarketConditionService()
        context = service._format_context(condition_data)

        assert "VIX" in context
        assert "22.5" in context
        assert "SPY" in context
        assert "Bullish" in context
        assert "Technology" in context
        assert "Financials" in context

    @pytest.mark.asyncio
    async def test_find_similar_conditions(self):
        """Test similarity search for market conditions."""
        query_text = "VIX 25 High volatility, SPY Bearish"

        mock_embedding = [0.1] * 384
        with patch("src.services.embeddings.market_condition.EmbeddingService") as mock_emb_svc:
            mock_service = MagicMock()
            mock_service.embed_text = AsyncMock(return_value=mock_embedding)
            mock_emb_svc.return_value = mock_service

            # Mock database query
            similar_condition = MarketConditionEmbedding(
                id=uuid4(),
                timestamp=datetime.now(),
                embedding=[0.2] * 384,
                context_text="VIX 24 High volatility",
                condition_metadata={},
            )

            mock_session = MagicMock()
            mock_result = MagicMock()
            # The code uses result.scalars().all()
            mock_result.scalars.return_value.all.return_value = [similar_condition]
            mock_session.execute = AsyncMock(return_value=mock_result)

            service = MarketConditionService(embedding_service=mock_service)
            # Use lower min_similarity to ensure mock results pass
            results = await service.find_similar_conditions(
                query_text, 
                session=mock_session,
                min_similarity=0.0  # Accept all results in test
            )

            assert len(results) == 1
            assert results[0] == similar_condition

