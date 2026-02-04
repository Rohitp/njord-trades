"""
FuzzyPicker - Weighted multi-factor scoring for symbol discovery.

Uses liquidity, volatility, momentum, and sector balance to score symbols.
Returns ranked list (0.0-1.0 scores) rather than binary pass/fail.
"""

from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Trade
from src.services.discovery.pickers.base import PickerResult, SymbolPicker
from src.services.discovery.scoring import (
    liquidity_score,
    momentum_score,
    volatility_score,
)
from src.services.discovery.sources.alpaca import AlpacaAssetSource
from src.services.embeddings.trade_embedding import TradeEmbeddingService
from src.services.market_data.fundamentals import (
    AlpacaFundamentalsProvider,
    CachedFundamentalsProvider,
    FundamentalsProvider,
)
from src.services.market_data.service import MarketDataService
from src.utils.logging import get_logger

log = get_logger(__name__)


class FuzzyPicker(SymbolPicker):
    """
    Weighted multi-factor scoring picker.

    Scores symbols based on:
    - Liquidity: Higher volume relative to average = better
    - Volatility: Moderate volatility preferred (not too low, not too high)
    - Momentum: Positive momentum preferred (but not extreme/overbought)
    - Sector balance: Penalize over-concentration (if context provided)

    Returns ranked list sorted by composite score (highest first).
    """

    def __init__(
        self,
        liquidity_weight: float = 0.3,
        volatility_weight: float = 0.25,
        momentum_weight: float = 0.35,
        sector_weight: float = 0.10,
        similarity_weight: float = 0.15,  # Weight for similarity-based adjustment
        min_score_threshold: float = 0.3,  # Only return symbols above this
        db_session: AsyncSession | None = None,  # Optional DB session for similarity search
        fundamentals_provider: FundamentalsProvider | None = None,
    ):
        """
        Initialize FuzzyPicker with scoring weights.

        Args:
            liquidity_weight: Weight for liquidity score (default: 0.3)
            volatility_weight: Weight for volatility score (default: 0.25)
            momentum_weight: Weight for momentum score (default: 0.35)
            sector_weight: Weight for sector balance score (default: 0.10)
            similarity_weight: Weight for similarity-based adjustment (default: 0.15)
            min_score_threshold: Minimum composite score to include (default: 0.3)
            db_session: Database session for similarity search (optional)
            fundamentals_provider: Fundamentals provider for sector data (optional)
        """
        # Normalize base weights (excluding similarity which is applied as adjustment)
        base_total = liquidity_weight + volatility_weight + momentum_weight + sector_weight
        if base_total > 0:
            self.liquidity_weight = liquidity_weight / base_total
            self.volatility_weight = volatility_weight / base_total
            self.momentum_weight = momentum_weight / base_total
            self.sector_weight = sector_weight / base_total
        else:
            # Default equal weights if all zero
            self.liquidity_weight = 0.25
            self.volatility_weight = 0.25
            self.momentum_weight = 0.25
            self.sector_weight = 0.25

        self.similarity_weight = similarity_weight
        self.min_score_threshold = min_score_threshold
        self.db_session = db_session

        self.asset_source = AlpacaAssetSource()
        self.market_data = MarketDataService()
        self.trade_embedding_service = TradeEmbeddingService() if db_session else None
        # Use cached provider with Alpaca fallback
        self.fundamentals = fundamentals_provider or CachedFundamentalsProvider(
            fallback_provider=AlpacaFundamentalsProvider()
        )

    @property
    def name(self) -> str:
        """Picker name."""
        return "fuzzy"

    async def pick(self, context: dict | None = None) -> List[PickerResult]:
        """
        Pick symbols using weighted multi-factor scoring.

        Args:
            context: Optional context containing:
                - portfolio_positions: List of current positions (for sector balance)
                - sector_limits: Dict of sector -> max allocation (for sector balance)

        Returns:
            List of PickerResult objects, sorted by score (highest first)
        """
        try:
            # Get all tradable stocks
            symbols = await self.asset_source.get_stocks()
            log.info("fuzzy_picker_starting", symbol_count=len(symbols))
        except ValueError as e:
            log.warning("fuzzy_picker_no_alpaca", error=str(e))
            return []  # Can't fetch symbols without Alpaca

        # Extract context for sector balance
        portfolio_positions = context.get("portfolio_positions", []) if context else []
        sector_limits = context.get("sector_limits", {}) if context else {}

        results = []
        for symbol in symbols:
            try:
                score, metadata = await self._calculate_composite_score(
                    symbol, portfolio_positions, sector_limits
                )

                if score >= self.min_score_threshold:
                    results.append(
                        PickerResult(
                            symbol=symbol,
                            score=score,
                            reason=self._build_reason(score, metadata),
                            metadata=metadata,
                        )
                    )
            except Exception as e:
                log.debug("fuzzy_picker_symbol_error", symbol=symbol, error=str(e))
                continue  # Skip symbols that error

        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)

        log.info(
            "fuzzy_picker_complete",
            passed=len(results),
            total=len(symbols),
            min_score=self.min_score_threshold,
        )
        return results

    async def _calculate_composite_score(
        self,
        symbol: str,
        portfolio_positions: List[dict],
        sector_limits: dict,
    ) -> tuple[float, dict]:
        """
        Calculate composite score for a symbol.

        Args:
            symbol: Symbol to score
            portfolio_positions: Current portfolio positions (for sector balance)
            sector_limits: Sector allocation limits (for sector balance)

        Returns:
            Tuple of (composite_score, metadata_dict)
        """
        metadata = {}

        # Get market data
        quote = await self.market_data.get_quote(symbol)
        indicators = await self.market_data.get_technical_indicators(symbol)

        # 1. Liquidity score
        liquidity = 0.0
        if quote.volume and indicators.volume_avg_20:
            liquidity = liquidity_score(quote.volume, indicators.volume_avg_20)
        elif quote.volume:
            # Fallback: use volume alone (normalize to reasonable range)
            liquidity = min(1.0, quote.volume / 10_000_000)  # 10M = max score
        metadata["liquidity_score"] = liquidity

        # 2. Volatility score (using price range or RSI as proxy)
        volatility = 0.5  # Default neutral
        if indicators.rsi_14 is not None:
            # RSI 30-70 range is moderate volatility (good)
            # RSI <30 or >70 is extreme (bad)
            if 30 <= indicators.rsi_14 <= 70:
                volatility = 1.0 - abs(indicators.rsi_14 - 50) / 20  # Peak at 50
            else:
                volatility = 0.3  # Penalize extreme RSI
        metadata["volatility_score"] = volatility

        # 3. Momentum score (using price change or SMA position)
        momentum = 0.5  # Default neutral
        if indicators.price and indicators.sma_20:
            # Price above SMA = positive momentum
            price_change_pct = (indicators.price - indicators.sma_20) / indicators.sma_20
            momentum = momentum_score(price_change_pct, min_change=0.02, max_change=0.10)
        elif indicators.price and indicators.sma_50:
            # Fallback to SMA_50
            price_change_pct = (indicators.price - indicators.sma_50) / indicators.sma_50
            momentum = momentum_score(price_change_pct, min_change=0.02, max_change=0.10)
        metadata["momentum_score"] = momentum

        # 4. Sector balance score (penalize over-concentration)
        sector_balance = 1.0  # Default: no penalty
        if portfolio_positions:
            # Get symbol's sector from fundamentals
            fundamentals = await self.fundamentals.get_fundamentals(symbol)
            if fundamentals and fundamentals.sector:
                # Count current positions in same sector
                sector_count = sum(
                    1
                    for pos in portfolio_positions
                    if pos.get("sector") == fundamentals.sector
                )
                total_positions = len(portfolio_positions)
                
                # Penalize if sector is over-represented (>30% of portfolio)
                sector_pct = sector_count / total_positions if total_positions > 0 else 0.0
                if sector_pct > 0.3:
                    # Reduce score proportionally to over-concentration
                    sector_balance = 1.0 - (sector_pct - 0.3) / 0.7  # 0.3 -> 1.0, 1.0 -> 0.0
                    sector_balance = max(0.0, sector_balance)
                metadata["sector"] = fundamentals.sector
                metadata["sector_pct"] = sector_pct
        metadata["sector_balance_score"] = sector_balance

        # Calculate weighted composite score (base score)
        base_score = (
            liquidity * self.liquidity_weight
            + volatility * self.volatility_weight
            + momentum * self.momentum_weight
            + sector_balance * self.sector_weight
        )

        # Apply similarity-based adjustment if DB session available
        similarity_adjustment = 0.0
        if self.db_session and self.trade_embedding_service:
            try:
                similarity_adjustment = await self._calculate_similarity_adjustment(
                    symbol, quote, indicators
                )
                metadata["similarity_adjustment"] = similarity_adjustment
            except Exception as e:
                log.debug("similarity_adjustment_failed", symbol=symbol, error=str(e))
                # Continue without adjustment on error

        # Apply adjustment: boost if positive, reduce if negative
        # Adjustment is weighted by similarity_weight
        composite_score = base_score + (similarity_adjustment * self.similarity_weight)
        composite_score = max(0.0, min(1.0, composite_score))  # Clamp to [0, 1]

        metadata["base_score"] = base_score
        metadata["composite_score"] = composite_score
        metadata["picker"] = "fuzzy"

        return composite_score, metadata

    async def _calculate_similarity_adjustment(
        self,
        symbol: str,
        quote,
        indicators,
    ) -> float:
        """
        Calculate score adjustment based on similar trade outcomes.

        Args:
            symbol: Symbol being scored
            quote: Current quote
            indicators: Technical indicators

        Returns:
            Adjustment factor (-1.0 to +1.0):
            - Positive = similar trades were winners (boost score)
            - Negative = similar trades were losers (reduce score)
            - 0.0 = no similar trades or mixed outcomes
        """
        # Build context text similar to trade embedding format
        context_parts = [f"Symbol: {symbol}"]

        if quote.price and isinstance(quote.price, (int, float)):
            context_parts.append(f"Price: ${float(quote.price):.2f}")

        if indicators.rsi_14 is not None and isinstance(indicators.rsi_14, (int, float)):
            context_parts.append(f"RSI: {float(indicators.rsi_14):.1f}")
        if indicators.sma_20 is not None and isinstance(indicators.sma_20, (int, float)):
            context_parts.append(f"SMA_20: ${float(indicators.sma_20):.2f}")
        if indicators.sma_50 is not None and isinstance(indicators.sma_50, (int, float)):
            context_parts.append(f"SMA_50: ${float(indicators.sma_50):.2f}")
        if indicators.sma_200 is not None and isinstance(indicators.sma_200, (int, float)):
            context_parts.append(f"SMA_200: ${float(indicators.sma_200):.2f}")
        if indicators.volume_ratio is not None and isinstance(indicators.volume_ratio, (int, float)):
            context_parts.append(f"Volume ratio: {float(indicators.volume_ratio):.2f}x")

        context_text = " | ".join(context_parts)

        # Find similar trades
        similar_trades = await self.trade_embedding_service.find_similar_trades(
            context_text=context_text,
            limit=5,
            min_similarity=0.7,
            session=self.db_session,
        )

        if not similar_trades:
            return 0.0  # No similar trades found

        # Get trade outcomes from database
        from sqlalchemy import select

        trade_ids = [te.trade_id for te in similar_trades]
        stmt = select(Trade).where(Trade.id.in_(trade_ids))
        result = await self.db_session.execute(stmt)
        trades = result.scalars().all()

        # Calculate win rate
        wins = sum(1 for t in trades if t.outcome == "WIN")
        losses = sum(1 for t in trades if t.outcome == "LOSS")
        total_with_outcome = wins + losses

        if total_with_outcome == 0:
            return 0.0  # No trades with outcomes yet

        win_rate = wins / total_with_outcome

        # Convert win rate to adjustment factor
        # 1.0 win rate = +1.0 adjustment (strong boost)
        # 0.0 win rate = -1.0 adjustment (strong penalty)
        # 0.5 win rate = 0.0 adjustment (neutral)
        adjustment = (win_rate - 0.5) * 2.0  # Scale to [-1, +1]

        log.debug(
            "similarity_adjustment_calculated",
            symbol=symbol,
            similar_trades=len(similar_trades),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            adjustment=adjustment,
        )

        return adjustment

    def _build_reason(self, score: float, metadata: dict) -> str:
        """Build reason string explaining the score."""
        parts = []
        if "liquidity_score" in metadata:
            parts.append(f"liquidity={metadata['liquidity_score']:.2f}")
        if "volatility_score" in metadata:
            parts.append(f"volatility={metadata['volatility_score']:.2f}")
        if "momentum_score" in metadata:
            parts.append(f"momentum={metadata['momentum_score']:.2f}")
        if "sector_balance_score" in metadata:
            parts.append(f"sector={metadata['sector_balance_score']:.2f}")
        if "similarity_adjustment" in metadata:
            adj = metadata["similarity_adjustment"]
            parts.append(f"similarity={adj:+.2f}")

        reason = f"Composite score {score:.3f} ({', '.join(parts)})"
        return reason

