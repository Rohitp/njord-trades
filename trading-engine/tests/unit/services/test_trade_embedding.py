"""
Tests for trade embedding service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.database.models import Trade, TradeStatus, TradeEmbedding
from src.services.embeddings.trade_embedding import TradeEmbeddingService
from src.workflows.state import Signal, SignalAction, FinalDecision


class TestTradeEmbeddingService:
    """Tests for TradeEmbeddingService."""

    @pytest.mark.asyncio
    async def test_embed_trade_creates_embedding(self):
        """Test that embedding is created for a trade."""
        # Create mock trade
        trade = Trade(
            id=uuid4(),
            symbol="AAPL",
            action="BUY",
            quantity=5,
            price=150.0,
            total_value=750.0,
            status=TradeStatus.FILLED.value,
            signal_confidence=0.75,
            risk_score=0.5,
        )

        # Create mock signal
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.75,
            proposed_quantity=5,
            reasoning="RSI oversold, price above SMA_200",
            rsi_14=28.0,
            sma_200=145.0,
        )

        # Create mock decision
        decision = FinalDecision(
            signal_id=signal.id,
            decision="EXECUTE",
            final_quantity=5,
            reasoning="Strong signal, acceptable risk",
            confidence=0.8,
        )

        # Mock embedding service
        mock_embedding = [0.1] * 384  # 384-dim embedding
        with patch("src.services.embeddings.trade_embedding.EmbeddingService") as mock_emb_svc:
            mock_service = MagicMock()
            mock_service.embed_text = AsyncMock(return_value=mock_embedding)
            mock_emb_svc.return_value = mock_service

            # Mock database session
            mock_session = MagicMock()
            mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: None))

            service = TradeEmbeddingService(embedding_service=mock_service)
            result = await service.embed_trade(trade, signal, decision, mock_session)

            assert result is not None
            assert result.trade_id == trade.id
            assert result.embedding == mock_embedding
            assert "AAPL" in result.context_text
            assert "BUY" in result.context_text
            assert "RSI" in result.context_text
            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_trade_skips_existing(self):
        """Test that existing embeddings are not recreated."""
        trade = Trade(
            id=uuid4(),
            symbol="AAPL",
            action="BUY",
            quantity=5,
            price=150.0,
            total_value=750.0,
            status=TradeStatus.FILLED.value,
        )

        existing_embedding = TradeEmbedding(
            id=uuid4(),
            trade_id=trade.id,
            embedding=[0.1] * 384,
            context_text="Existing",
        )

        mock_session = MagicMock()
        mock_result = MagicMock()
        # The code calls scalar_one_or_none() to check, then scalar_one() if it exists
        mock_result.scalar_one_or_none.return_value = existing_embedding
        mock_result.scalar_one.return_value = existing_embedding
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = TradeEmbeddingService()
        result = await service.embed_trade(trade, session=mock_session)

        assert result is existing_embedding
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_trade_no_session(self):
        """Test that None is returned when no session provided."""
        trade = Trade(
            id=uuid4(),
            symbol="AAPL",
            action="BUY",
            quantity=5,
            price=150.0,
            total_value=750.0,
            status=TradeStatus.FILLED.value,
        )

        service = TradeEmbeddingService()
        result = await service.embed_trade(trade, session=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_format_trade_context(self):
        """Test that trade context is formatted correctly."""
        trade = Trade(
            id=uuid4(),
            symbol="AAPL",
            action="BUY",
            quantity=5,
            price=150.0,
            total_value=750.0,
            status=TradeStatus.FILLED.value,
            signal_confidence=0.75,
            risk_score=0.5,
            outcome="WIN",
            pnl=50.0,
            pnl_pct=6.67,
        )

        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.75,
            proposed_quantity=5,
            reasoning="RSI oversold",
            rsi_14=28.0,
            sma_20=148.0,
            volume_ratio=2.0,
        )

        decision = FinalDecision(
            signal_id=signal.id,
            decision="EXECUTE",
            final_quantity=5,
            reasoning="Strong signal",
            confidence=0.8,
        )

        service = TradeEmbeddingService()
        context = service._format_trade_context(trade, signal, decision)

        assert "AAPL" in context
        assert "BUY" in context
        assert "RSI" in context
        assert "SMA_20" in context
        assert "Volume ratio" in context
        assert "WIN" in context
        assert "P&L" in context

    @pytest.mark.asyncio
    async def test_find_similar_trades(self):
        """Test similarity search for trades."""
        query_text = "AAPL BUY RSI 28 oversold"

        mock_embedding = [0.1] * 384
        with patch("src.services.embeddings.trade_embedding.EmbeddingService") as mock_emb_svc:
            mock_service = MagicMock()
            mock_service.embed_text = AsyncMock(return_value=mock_embedding)
            mock_emb_svc.return_value = mock_service

            # Mock database query
            similar_trade = TradeEmbedding(
                id=uuid4(),
                trade_id=uuid4(),
                embedding=[0.2] * 384,
                context_text="AAPL BUY RSI 30",
            )

            mock_session = MagicMock()
            # The new code uses result.all() and extracts row[0] for each row
            # Mock the query to return results that pass the filter
            mock_result = MagicMock()
            # result.all() returns list of rows, each row is a tuple with TradeEmbedding as first element
            mock_result.all.return_value = [(similar_trade,)]  # Tuple with TradeEmbedding as first element
            mock_session.execute = AsyncMock(return_value=mock_result)

            service = TradeEmbeddingService(embedding_service=mock_service)
            # Use lower min_similarity to ensure mock results pass
            results = await service.find_similar_trades(
                query_text, 
                session=mock_session,
                min_similarity=0.0  # Accept all results in test
            )

            assert len(results) == 1
            assert results[0] == similar_trade

