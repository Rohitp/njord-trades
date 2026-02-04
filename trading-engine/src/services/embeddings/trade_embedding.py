"""
Trade embedding service.

Generates and stores vector embeddings for completed trades.
Used for similarity search to find similar failed setups.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Trade, TradeEmbedding
from src.services.embeddings.service import EmbeddingService
from src.utils.logging import get_logger
from src.workflows.state import FinalDecision, Signal

log = get_logger(__name__)


class TradeEmbeddingService:
    """
    Service for generating and storing trade embeddings.

    Creates embeddings from trade context (symbol, action, reasoning, outcome)
    for similarity search in the Validator.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
    ):
        """
        Initialize trade embedding service.

        Args:
            embedding_service: EmbeddingService instance (creates default if None)
        """
        self.embedding_service = embedding_service or EmbeddingService()

    async def embed_trade(
        self,
        trade: Trade,
        signal: Signal | None = None,
        decision: FinalDecision | None = None,
        session: AsyncSession | None = None,
    ) -> TradeEmbedding | None:
        """
        Generate and store embedding for a completed trade.

        Args:
            trade: Trade database record
            signal: Original signal that generated the trade (optional)
            decision: Final decision that executed the trade (optional)
            session: Database session (required for persistence)

        Returns:
            TradeEmbedding record if successful, None on error
        """
        if session is None:
            log.warning("trade_embedding_no_session", trade_id=str(trade.id))
            return None

        try:
            # Check if embedding already exists
            existing = await session.execute(
                select(TradeEmbedding).where(TradeEmbedding.trade_id == trade.id)
            )
            if existing.scalar_one_or_none():
                log.debug("trade_embedding_exists", trade_id=str(trade.id))
                return existing.scalar_one()

            # Format context text
            context_text = self._format_trade_context(trade, signal, decision)

            # Generate embedding
            embedding = await self.embedding_service.embed_text(context_text)

            # Create TradeEmbedding record
            trade_embedding = TradeEmbedding(
                trade_id=trade.id,
                embedding=embedding,
                context_text=context_text,
            )

            session.add(trade_embedding)
            # Note: session commit should be handled by caller

            log.info(
                "trade_embedding_created",
                trade_id=str(trade.id),
                symbol=trade.symbol,
                action=trade.action,
            )

            return trade_embedding

        except Exception as e:
            log.error(
                "trade_embedding_error",
                trade_id=str(trade.id),
                error=str(e),
                exc_info=True,
            )
            return None

    def _format_trade_context(
        self,
        trade: Trade,
        signal: Signal | None = None,
        decision: FinalDecision | None = None,
    ) -> str:
        """
        Format trade context into text for embedding.

        Includes:
        - Symbol and action
        - Signal reasoning (if available)
        - Decision reasoning (if available)
        - Technical indicators (if available from signal)
        - Outcome (if available)

        Args:
            trade: Trade record
            signal: Original signal (optional)
            decision: Final decision (optional)

        Returns:
            Formatted context text
        """
        parts = []

        # Basic trade info
        parts.append(f"Symbol: {trade.symbol}")
        parts.append(f"Action: {trade.action}")
        parts.append(f"Quantity: {trade.quantity}")
        parts.append(f"Price: ${trade.price:.2f}")

        # Signal reasoning
        if signal and signal.reasoning:
            parts.append(f"Signal reasoning: {signal.reasoning}")
        if signal and signal.confidence:
            parts.append(f"Signal confidence: {signal.confidence:.2f}")

        # Technical indicators from signal
        if signal:
            indicators = []
            if signal.rsi_14 is not None:
                indicators.append(f"RSI: {signal.rsi_14:.1f}")
            if signal.sma_20 is not None:
                indicators.append(f"SMA_20: ${signal.sma_20:.2f}")
            if signal.sma_50 is not None:
                indicators.append(f"SMA_50: ${signal.sma_50:.2f}")
            if signal.sma_200 is not None:
                indicators.append(f"SMA_200: ${signal.sma_200:.2f}")
            if signal.volume_ratio is not None:
                indicators.append(f"Volume ratio: {signal.volume_ratio:.2f}x")
            if indicators:
                parts.append(f"Technical indicators: {', '.join(indicators)}")

        # Decision reasoning
        if decision and decision.reasoning:
            parts.append(f"Decision reasoning: {decision.reasoning}")
        if decision and decision.confidence:
            parts.append(f"Decision confidence: {decision.confidence:.2f}")

        # Risk score
        if trade.risk_score:
            parts.append(f"Risk score: {trade.risk_score:.2f}")

        # Outcome (if available)
        if trade.outcome:
            parts.append(f"Outcome: {trade.outcome}")
        if trade.pnl is not None:
            parts.append(f"P&L: ${trade.pnl:.2f}")
        if trade.pnl_pct is not None:
            parts.append(f"P&L %: {trade.pnl_pct:.2f}%")

        return " | ".join(parts)

    async def find_similar_trades(
        self,
        context_text: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        session: AsyncSession | None = None,
    ) -> list[TradeEmbedding]:
        """
        Find similar trades using vector similarity search.

        Args:
            context_text: Text to search for (e.g., current signal context)
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity threshold (0.0-1.0)
            session: Database session

        Returns:
            List of TradeEmbedding records sorted by similarity (highest first)
        """
        if session is None:
            log.warning("similarity_search_no_session")
            return []

        try:
            # Generate embedding for query text
            query_embedding = await self.embedding_service.embed_text(context_text)

            # Query using pgvector cosine similarity
            # For normalized embeddings, cosine_distance = 1 - cosine_similarity
            # So: similarity = 1 - cosine_distance
            # To filter by min_similarity, we need: 1 - distance >= min_similarity
            # Which means: distance <= 1 - min_similarity
            from pgvector.sqlalchemy import Vector
            from sqlalchemy import func

            # Calculate max_distance threshold from min_similarity
            # similarity = 1 - distance, so distance = 1 - similarity
            max_distance = 1.0 - min_similarity
            cosine_dist = TradeEmbedding.embedding.cosine_distance(query_embedding)

            stmt = (
                select(TradeEmbedding)
                .where(cosine_dist <= max_distance)
                .order_by(cosine_dist.asc())  # Lower distance = higher similarity
                .limit(limit)
            )

            result = await session.execute(stmt)
            rows = result.all()

            # Extract TradeEmbedding objects (first column)
            similar_trades = [row[0] for row in rows]

            log.info(
                "similarity_search_complete",
                query_length=len(context_text),
                min_similarity=min_similarity,
                results=len(similar_trades),
            )

            return similar_trades

        except Exception as e:
            log.error("similarity_search_error", error=str(e), exc_info=True)
            return []

