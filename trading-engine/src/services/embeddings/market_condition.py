"""
Market condition embedding service.

Generates and stores vector embeddings for market conditions (VIX, SPY trend, sector performance).
Used for finding similar market regimes in LLMPicker.
"""

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import MarketConditionEmbedding
from src.services.embeddings.service import EmbeddingService
from src.services.market_data.service import MarketDataService
from src.utils.logging import get_logger

log = get_logger(__name__)


# Sector ETFs for performance tracking
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrial",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


class MarketConditionService:
    """
    Service for generating and storing market condition embeddings.

    Collects VIX, SPY trend, and sector performance data,
    formats it as context text, and generates embeddings.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        market_data_service: MarketDataService | None = None,
    ):
        """
        Initialize market condition service.

        Args:
            embedding_service: EmbeddingService instance (creates default if None)
            market_data_service: MarketDataService instance (creates default if None)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.market_data = market_data_service or MarketDataService()

    async def embed_market_condition(
        self,
        timestamp: datetime | None = None,
        session: AsyncSession | None = None,
    ) -> MarketConditionEmbedding | None:
        """
        Collect market condition data and generate embedding.

        Args:
            timestamp: Timestamp for the market condition (default: now)
            session: Database session (required for persistence)

        Returns:
            MarketConditionEmbedding record if successful, None on error
        """
        if session is None:
            log.warning("market_condition_no_session")
            return None

        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Check if embedding already exists for this timestamp (within 1 hour)
            # This prevents duplicate embeddings for the same market session
            one_hour_ago = timestamp - timedelta(hours=1)
            existing = await session.execute(
                select(MarketConditionEmbedding)
                .where(MarketConditionEmbedding.timestamp >= one_hour_ago)
                .where(MarketConditionEmbedding.timestamp <= timestamp)
                .order_by(MarketConditionEmbedding.timestamp.desc())
                .limit(1)
            )
            existing_record = existing.scalar_one_or_none()
            if existing_record:
                log.debug("market_condition_exists", timestamp=timestamp)
                return existing_record

            # Collect market condition data
            condition_data = await self._collect_market_data()

            # Format context text
            context_text = self._format_context(condition_data)

            # Generate embedding
            embedding = await self.embedding_service.embed_text(context_text)

            # Create MarketConditionEmbedding record
            market_embedding = MarketConditionEmbedding(
                timestamp=timestamp,
                embedding=embedding,
                context_text=context_text,
                condition_metadata=condition_data,
            )

            session.add(market_embedding)
            # Note: session commit should be handled by caller

            log.info("market_condition_embedded", timestamp=timestamp)

            return market_embedding

        except Exception as e:
            log.error(
                "market_condition_error",
                timestamp=timestamp,
                error=str(e),
                exc_info=True,
            )
            return None

    async def _collect_market_data(self) -> dict:
        """
        Collect market condition data (VIX, SPY trend, sector performance).

        Returns:
            Dictionary with market condition data
        """
        condition_data = {}

        try:
            # Get VIX (volatility index)
            vix_quote = await self.market_data.get_quote("VIX")
            condition_data["vix"] = float(vix_quote.price) if vix_quote.price else None
        except Exception as e:
            log.warning("vix_fetch_failed", error=str(e))
            condition_data["vix"] = None

        try:
            # Get SPY trend (5-day return, position relative to SMA_200)
            spy_indicators = await self.market_data.get_technical_indicators("SPY")
            condition_data["spy_price"] = float(spy_indicators.price) if spy_indicators.price else None
            condition_data["spy_sma_200"] = float(spy_indicators.sma_200) if spy_indicators.sma_200 else None
            
            # Calculate 5-day return (approximate from current price)
            # For more accuracy, we'd need to fetch bars and calculate, but this is a reasonable approximation
            if spy_indicators.price and spy_indicators.sma_200:
                spy_above_sma = spy_indicators.price > spy_indicators.sma_200
                spy_trend_pct = ((spy_indicators.price - spy_indicators.sma_200) / spy_indicators.sma_200) * 100
                condition_data["spy_above_sma_200"] = spy_above_sma
                condition_data["spy_trend_pct"] = float(spy_trend_pct)
            else:
                condition_data["spy_above_sma_200"] = None
                condition_data["spy_trend_pct"] = None
        except Exception as e:
            log.warning("spy_fetch_failed", error=str(e))
            condition_data["spy_price"] = None
            condition_data["spy_sma_200"] = None
            condition_data["spy_above_sma_200"] = None
            condition_data["spy_trend_pct"] = None

        # Get sector performance (parallel fetch)
        sector_performance = {}
        sector_symbols = list(SECTOR_ETFS.keys())
        
        try:
            sector_quotes = await self.market_data.get_quotes(sector_symbols)
            
            # Calculate daily change (we'd need previous day's close for accurate calculation)
            # For now, we'll just record current prices and let the LLM interpret
            for quote in sector_quotes:
                if quote.symbol in SECTOR_ETFS:
                    sector_performance[quote.symbol] = {
                        "name": SECTOR_ETFS[quote.symbol],
                        "price": float(quote.price) if quote.price else None,
                    }
        except Exception as e:
            log.warning("sector_fetch_failed", error=str(e))
            # Continue with empty sector data

        condition_data["sector_performance"] = sector_performance

        return condition_data

    def _format_context(self, condition_data: dict) -> str:
        """
        Format market condition data into context text.

        Args:
            condition_data: Dictionary with market condition data

        Returns:
            Formatted context text
        """
        parts = []

        # VIX
        if condition_data.get("vix") is not None:
            vix = condition_data["vix"]
            vix_level = "High" if vix > 25 else "Moderate" if vix > 15 else "Low"
            parts.append(f"VIX: {vix:.2f} ({vix_level} volatility)")

        # SPY trend
        if condition_data.get("spy_price") is not None:
            spy_price = condition_data["spy_price"]
            if condition_data.get("spy_above_sma_200") is not None:
                trend = "Bullish" if condition_data["spy_above_sma_200"] else "Bearish"
                trend_pct = condition_data.get("spy_trend_pct", 0)
                parts.append(f"SPY: ${spy_price:.2f}, {trend} trend ({trend_pct:+.2f}% vs SMA_200)")
            else:
                parts.append(f"SPY: ${spy_price:.2f}")

        # Sector performance
        sector_perf = condition_data.get("sector_performance", {})
        if sector_perf:
            sector_parts = []
            for symbol, data in sector_perf.items():
                if data.get("price") is not None:
                    sector_parts.append(f"{data['name']}: ${data['price']:.2f}")
            if sector_parts:
                parts.append(f"Sectors: {', '.join(sector_parts)}")

        if not parts:
            return "Market condition data unavailable"

        return " | ".join(parts)

    async def find_similar_conditions(
        self,
        context_text: str,
        limit: int = 5,
        session: AsyncSession | None = None,
    ) -> list[MarketConditionEmbedding]:
        """
        Find similar market conditions using vector similarity search.

        Args:
            context_text: Text describing current market conditions
            limit: Maximum number of results
            session: Database session

        Returns:
            List of MarketConditionEmbedding records sorted by similarity
        """
        if session is None:
            log.warning("similarity_search_no_session")
            return []

        try:
            # Generate embedding for query text
            query_embedding = await self.embedding_service.embed_text(context_text)

            # Query using pgvector cosine similarity
            from pgvector.sqlalchemy import Vector

            stmt = (
                select(MarketConditionEmbedding)
                .order_by(
                    MarketConditionEmbedding.embedding.cosine_distance(query_embedding)
                )
                .limit(limit)
            )

            result = await session.execute(stmt)
            similar_conditions = result.scalars().all()

            log.info(
                "similarity_search_complete",
                query_length=len(context_text),
                results=len(similar_conditions),
            )

            return similar_conditions

        except Exception as e:
            log.error("similarity_search_error", error=str(e), exc_info=True)
            return []

