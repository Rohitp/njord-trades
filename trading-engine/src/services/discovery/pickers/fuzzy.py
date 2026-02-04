"""
FuzzyPicker - Weighted multi-factor scoring for symbol discovery.

Uses liquidity, volatility, momentum, and sector balance to score symbols.
Returns ranked list (0.0-1.0 scores) rather than binary pass/fail.
"""

from typing import List

from src.services.discovery.pickers.base import PickerResult, SymbolPicker
from src.services.discovery.scoring import (
    liquidity_score,
    momentum_score,
    volatility_score,
)
from src.services.discovery.sources.alpaca import AlpacaAssetSource
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
        min_score_threshold: float = 0.3,  # Only return symbols above this
    ):
        """
        Initialize FuzzyPicker with scoring weights.

        Args:
            liquidity_weight: Weight for liquidity score (default: 0.3)
            volatility_weight: Weight for volatility score (default: 0.25)
            momentum_weight: Weight for momentum score (default: 0.35)
            sector_weight: Weight for sector balance score (default: 0.10)
            min_score_threshold: Minimum composite score to include (default: 0.3)
        """
        # Normalize weights to sum to 1.0
        total_weight = liquidity_weight + volatility_weight + momentum_weight + sector_weight
        if total_weight > 0:
            self.liquidity_weight = liquidity_weight / total_weight
            self.volatility_weight = volatility_weight / total_weight
            self.momentum_weight = momentum_weight / total_weight
            self.sector_weight = sector_weight / total_weight
        else:
            # Default equal weights if all zero
            self.liquidity_weight = 0.25
            self.volatility_weight = 0.25
            self.momentum_weight = 0.25
            self.sector_weight = 0.25

        self.min_score_threshold = min_score_threshold

        self.asset_source = AlpacaAssetSource()
        self.market_data = MarketDataService()

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
        # TODO: Implement sector balance when sector data is available
        # For now, this is a placeholder
        metadata["sector_balance_score"] = sector_balance

        # Calculate weighted composite score
        composite_score = (
            liquidity * self.liquidity_weight
            + volatility * self.volatility_weight
            + momentum * self.momentum_weight
            + sector_balance * self.sector_weight
        )

        metadata["composite_score"] = composite_score
        metadata["picker"] = "fuzzy"

        return composite_score, metadata

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

        reason = f"Composite score {score:.3f} ({', '.join(parts)})"
        return reason

