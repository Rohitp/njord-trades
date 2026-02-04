"""
Tests for embedding service.

Tests BGE-small-en provider and EmbeddingService.
"""

import pytest

from src.services.embeddings.providers.bge import BGESmallProvider
from src.services.embeddings.service import EmbeddingService


class TestBGESmallProvider:
    """Tests for BGE-small-en provider."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text."""
        provider = BGESmallProvider()
        text = "AAPL BUY 5 shares @ 150.0, RSI 65, volume 2x, WIN"
        
        embedding = await provider.embed(text)
        
        assert embedding is not None
        assert len(embedding) == 384  # BGE-small-en dimensions
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test embedding multiple texts."""
        provider = BGESmallProvider()
        texts = [
            "AAPL BUY 5 shares @ 150.0, RSI 65, WIN",
            "MSFT SELL 3 shares @ 300.0, RSI 75, LOSS",
        ]
        
        embeddings = await provider.embed_batch(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(isinstance(x, float) for emb in embeddings for x in emb)

    @pytest.mark.asyncio
    async def test_embed_similarity(self):
        """Test that similar texts produce similar embeddings."""
        provider = BGESmallProvider()
        
        text1 = "Apple stock breakout with high volume"
        text2 = "AAPL surge on heavy trading"
        text3 = "Microsoft earnings report beats expectations"
        
        emb1 = await provider.embed(text1)
        emb2 = await provider.embed(text2)
        emb3 = await provider.embed(text3)
        
        # Calculate cosine similarity (embeddings are normalized)
        similarity_12 = sum(a * b for a, b in zip(emb1, emb2))
        similarity_13 = sum(a * b for a, b in zip(emb1, emb3))
        
        # Similar texts should have higher similarity
        assert similarity_12 > similarity_13, "Similar texts should have higher cosine similarity"

    def test_get_dimensions(self):
        """Test that dimensions are correct."""
        provider = BGESmallProvider()
        assert provider.get_dimensions() == 384


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.mark.asyncio
    async def test_embed_text(self):
        """Test embedding via service."""
        service = EmbeddingService(provider="bge-small")
        text = "Test embedding text"
        
        embedding = await service.embed_text(text)
        
        assert len(embedding) == 384
        assert service.get_dimensions() == 384

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding via service."""
        service = EmbeddingService(provider="bge-small")
        texts = ["Text 1", "Text 2", "Text 3"]
        
        embeddings = await service.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingService(provider="unknown-provider")

    def test_default_provider(self):
        """Test that default provider is used from config."""
        service = EmbeddingService()
        assert service.provider_name == "bge-small"
        assert service.dimensions == 384

