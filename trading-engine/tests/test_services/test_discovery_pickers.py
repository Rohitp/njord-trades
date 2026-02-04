"""
Tests for symbol discovery pickers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.discovery.pickers.fuzzy import FuzzyPicker
from src.services.discovery.pickers.llm import LLMPicker
from src.services.discovery.pickers.metric import MetricPicker
from src.services.discovery.pickers.base import PickerResult


class TestMetricPicker:
    """Tests for MetricPicker."""

    @pytest.mark.asyncio
    async def test_metric_picker_name(self):
        """Test that picker has correct name."""
        picker = MetricPicker()
        assert picker.name == "metric"

    @pytest.mark.asyncio
    async def test_metric_picker_no_alpaca(self):
        """Test that picker returns empty list when Alpaca not configured."""
        with patch("src.services.discovery.pickers.metric.AlpacaAssetSource") as mock_source:
            mock_source.return_value.get_stocks.side_effect = ValueError("Alpaca not configured")
            picker = MetricPicker()
            results = await picker.pick()
            assert results == []

    @pytest.mark.asyncio
    async def test_metric_picker_volume_filter(self):
        """Test that volume filter works correctly."""
        with patch("src.services.discovery.pickers.metric.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.metric.MarketDataService") as mock_market:
            
            # Setup mocks
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL", "MSFT"])
            
            # Create mock quote responses
            quote_high_volume = MagicMock()
            quote_high_volume.volume = 10_000_000  # Passes filter
            quote_high_volume.bid = 150.0
            quote_high_volume.ask = 150.1
            quote_high_volume.price = 150.05
            
            quote_low_volume = MagicMock()
            quote_low_volume.volume = 100_000  # Fails filter (< 1M)
            quote_low_volume.bid = None
            quote_low_volume.ask = None
            quote_low_volume.price = 100.0
            
            mock_market.return_value.get_quote = AsyncMock(side_effect=[
                quote_high_volume,  # AAPL passes
                quote_low_volume,   # MSFT fails
            ])
            
            picker = MetricPicker(min_volume=1_000_000)
            results = await picker.pick()
            
            assert len(results) == 1
            assert results[0].symbol == "AAPL"
            assert results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_metric_picker_spread_filter(self):
        """Test that spread filter works correctly."""
        with patch("src.services.discovery.pickers.metric.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.metric.MarketDataService") as mock_market:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            # Wide spread (fails)
            quote_wide_spread = MagicMock()
            quote_wide_spread.volume = 10_000_000
            quote_wide_spread.bid = 150.0
            quote_wide_spread.ask = 152.0  # 1.3% spread (fails 1% threshold)
            quote_wide_spread.price = 151.0
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote_wide_spread)
            
            picker = MetricPicker(max_spread_pct=0.01)  # 1% max
            results = await picker.pick()
            
            assert len(results) == 0  # Should fail spread filter


class TestFuzzyPicker:
    """Tests for FuzzyPicker."""

    @pytest.mark.asyncio
    async def test_fuzzy_picker_name(self):
        """Test that picker has correct name."""
        picker = FuzzyPicker()
        assert picker.name == "fuzzy"

    @pytest.mark.asyncio
    async def test_fuzzy_picker_no_alpaca(self):
        """Test that picker returns empty list when Alpaca not configured."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source:
            mock_source.return_value.get_stocks.side_effect = ValueError("Alpaca not configured")
            picker = FuzzyPicker()
            results = await picker.pick()
            assert results == []

    @pytest.mark.asyncio
    async def test_fuzzy_picker_scoring(self):
        """Test that scoring works correctly."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            # Create mock quote
            quote = MagicMock()
            quote.volume = 5_000_000
            quote.price = 150.0
            
            # Create mock indicators (good scores)
            indicators = MagicMock()
            indicators.price = 155.0  # Above SMA (positive momentum)
            indicators.sma_20 = 150.0
            indicators.sma_50 = 145.0
            indicators.rsi_14 = 55.0  # Moderate (good volatility)
            indicators.volume_avg_20 = 3_000_000  # Current volume > avg (good liquidity)
            indicators.volume_ratio = 1.67
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            picker = FuzzyPicker(min_score_threshold=0.0)  # Include all
            results = await picker.pick()
            
            assert len(results) == 1
            assert results[0].symbol == "AAPL"
            assert 0.0 <= results[0].score <= 1.0
            assert "liquidity" in results[0].reason.lower()
            assert "momentum" in results[0].reason.lower()

    @pytest.mark.asyncio
    async def test_fuzzy_picker_min_threshold(self):
        """Test that min_score_threshold filters low-scoring symbols."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            # Create mock quote (low volume)
            quote = MagicMock()
            quote.volume = 100_000  # Low volume
            
            # Create mock indicators (poor scores)
            indicators = MagicMock()
            indicators.price = 140.0
            indicators.sma_20 = 150.0  # Price below SMA (negative momentum)
            indicators.sma_50 = 145.0
            indicators.rsi_14 = 25.0  # Oversold (extreme)
            indicators.volume_avg_20 = 1_000_000
            indicators.volume_ratio = 0.1  # Very low volume
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            picker = FuzzyPicker(min_score_threshold=0.5)  # High threshold
            results = await picker.pick()
            
            # Should be filtered out due to low score
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_fuzzy_picker_weight_normalization(self):
        """Test that weights are normalized correctly."""
        # Weights that don't sum to 1.0 should be normalized
        picker = FuzzyPicker(
            liquidity_weight=0.6,
            volatility_weight=0.4,
            momentum_weight=0.0,
            sector_weight=0.0,
        )
        
        # Should normalize to sum to 1.0
        total = picker.liquidity_weight + picker.volatility_weight + picker.momentum_weight + picker.sector_weight
        assert abs(total - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_fuzzy_picker_sorted_results(self):
        """Test that results are sorted by score (highest first)."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL", "MSFT"])
            
            # Create mock quotes
            quote1 = MagicMock()
            quote1.volume = 5_000_000
            quote1.price = 150.0
            
            quote2 = MagicMock()
            quote2.volume = 8_000_000
            quote2.price = 300.0
            
            # Create mock indicators (AAPL has lower score)
            indicators1 = MagicMock()
            indicators1.price = 150.0
            indicators1.sma_20 = 150.0
            indicators1.sma_50 = 145.0
            indicators1.rsi_14 = 50.0
            indicators1.volume_avg_20 = 3_000_000
            indicators1.volume_ratio = 1.67
            
            # MSFT has higher score
            indicators2 = MagicMock()
            indicators2.price = 310.0  # Stronger momentum
            indicators2.sma_20 = 300.0
            indicators2.sma_50 = 295.0
            indicators2.rsi_14 = 60.0  # Better RSI
            indicators2.volume_avg_20 = 4_000_000
            indicators2.volume_ratio = 2.0  # Better volume
            
            mock_market.return_value.get_quote = AsyncMock(side_effect=[quote1, quote2])
            mock_market.return_value.get_technical_indicators = AsyncMock(side_effect=[indicators1, indicators2])
            
            picker = FuzzyPicker(min_score_threshold=0.0)
            results = await picker.pick()
            
            assert len(results) == 2
            # Results should be sorted by score (highest first)
            assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_fuzzy_picker_similarity_adjustment_win_trades(self):
        """Test that WIN trades boost the score."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market, \
             patch("src.services.discovery.pickers.fuzzy.TradeEmbeddingService") as mock_embedding:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            from src.services.market_data.provider import Quote, TechnicalIndicators
            
            quote = Quote(
                symbol="AAPL",
                price=150.0,
                volume=5_000_000,
            )
            
            indicators = TechnicalIndicators(
                symbol="AAPL",
                price=155.0,
                sma_20=150.0,
                sma_50=None,
                sma_200=None,
                rsi_14=55.0,
                volume_avg_20=3_000_000,
                volume_ratio=1.67,
            )
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            # Mock similarity search - return trades with WIN outcomes
            from uuid import uuid4
            
            mock_trade_embedding1 = MagicMock()
            mock_trade_embedding1.trade_id = uuid4()
            mock_trade_embedding2 = MagicMock()
            mock_trade_embedding2.trade_id = uuid4()
            
            mock_trade1 = MagicMock()
            mock_trade1.id = mock_trade_embedding1.trade_id
            mock_trade1.outcome = "WIN"
            mock_trade2 = MagicMock()
            mock_trade2.id = mock_trade_embedding2.trade_id
            mock_trade2.outcome = "WIN"
            
            mock_embedding_service = MagicMock()
            mock_embedding_service.find_similar_trades = AsyncMock(
                return_value=[mock_trade_embedding1, mock_trade_embedding2]
            )
            mock_embedding.return_value = mock_embedding_service
            
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_trade1, mock_trade2]
            mock_session.execute = AsyncMock(return_value=mock_result)
            
            picker = FuzzyPicker(
                min_score_threshold=0.0,
                similarity_weight=0.2,  # 20% weight for similarity
                db_session=mock_session,
            )
            results = await picker.pick()
            
            assert len(results) == 1
            assert results[0].symbol == "AAPL"
            # Should have similarity adjustment in metadata
            assert "similarity_adjustment" in results[0].metadata
            # 100% win rate should give +1.0 adjustment
            assert results[0].metadata["similarity_adjustment"] > 0.9
            # Score should be boosted
            assert results[0].score > results[0].metadata.get("base_score", 0.0)

    @pytest.mark.asyncio
    async def test_fuzzy_picker_similarity_adjustment_loss_trades(self):
        """Test that LOSS trades reduce the score."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market, \
             patch("src.services.discovery.pickers.fuzzy.TradeEmbeddingService") as mock_embedding:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            from src.services.market_data.provider import Quote, TechnicalIndicators
            
            quote = Quote(
                symbol="AAPL",
                price=150.0,
                volume=5_000_000,
            )
            
            indicators = TechnicalIndicators(
                symbol="AAPL",
                price=155.0,
                sma_20=150.0,
                sma_50=None,
                sma_200=None,
                rsi_14=55.0,
                volume_avg_20=3_000_000,
                volume_ratio=1.67,
            )
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            from uuid import uuid4
            
            mock_trade_embedding = MagicMock()
            mock_trade_embedding.trade_id = uuid4()
            
            mock_trade = MagicMock()
            mock_trade.id = mock_trade_embedding.trade_id
            mock_trade.outcome = "LOSS"
            
            mock_embedding_service = MagicMock()
            mock_embedding_service.find_similar_trades = AsyncMock(return_value=[mock_trade_embedding])
            mock_embedding.return_value = mock_embedding_service
            
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_trade]
            mock_session.execute = AsyncMock(return_value=mock_result)
            
            picker = FuzzyPicker(
                min_score_threshold=0.0,
                similarity_weight=0.2,
                db_session=mock_session,
            )
            results = await picker.pick()
            
            assert len(results) == 1
            # 0% win rate should give -1.0 adjustment
            assert results[0].metadata["similarity_adjustment"] < -0.9
            # Score should be reduced
            assert results[0].score < results[0].metadata.get("base_score", 1.0)

    @pytest.mark.asyncio
    async def test_fuzzy_picker_similarity_adjustment_no_similar_trades(self):
        """Test that no similar trades results in no adjustment."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market, \
             patch("src.services.discovery.pickers.fuzzy.TradeEmbeddingService") as mock_embedding:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            from src.services.market_data.provider import Quote, TechnicalIndicators
            
            quote = Quote(
                symbol="AAPL",
                price=150.0,
                volume=5_000_000,
            )
            
            indicators = TechnicalIndicators(
                symbol="AAPL",
                price=155.0,
                sma_20=150.0,
                sma_50=None,
                sma_200=None,
                rsi_14=55.0,
                volume_avg_20=3_000_000,
                volume_ratio=1.67,
            )
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            mock_embedding_service = MagicMock()
            mock_embedding_service.find_similar_trades = AsyncMock(return_value=[])  # No similar trades
            mock_embedding.return_value = mock_embedding_service
            
            mock_session = MagicMock()
            
            picker = FuzzyPicker(
                min_score_threshold=0.0,
                db_session=mock_session,
            )
            results = await picker.pick()
            
            assert len(results) == 1
            # No similarity adjustment should be applied
            assert "similarity_adjustment" not in results[0].metadata or results[0].metadata.get("similarity_adjustment") == 0.0

    @pytest.mark.asyncio
    async def test_fuzzy_picker_similarity_adjustment_mixed_outcomes(self):
        """Test that mixed outcomes result in neutral adjustment."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market, \
             patch("src.services.discovery.pickers.fuzzy.TradeEmbeddingService") as mock_embedding:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            from src.services.market_data.provider import Quote, TechnicalIndicators
            
            quote = Quote(
                symbol="AAPL",
                price=150.0,
                volume=5_000_000,
            )
            
            indicators = TechnicalIndicators(
                symbol="AAPL",
                price=155.0,
                sma_20=150.0,
                sma_50=None,
                sma_200=None,
                rsi_14=55.0,
                volume_avg_20=3_000_000,
                volume_ratio=1.67,
            )
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            from uuid import uuid4
            
            mock_trade_embedding1 = MagicMock()
            mock_trade_embedding1.trade_id = uuid4()
            mock_trade_embedding2 = MagicMock()
            mock_trade_embedding2.trade_id = uuid4()
            
            mock_trade1 = MagicMock()
            mock_trade1.id = mock_trade_embedding1.trade_id
            mock_trade1.outcome = "WIN"
            mock_trade2 = MagicMock()
            mock_trade2.id = mock_trade_embedding2.trade_id
            mock_trade2.outcome = "LOSS"
            
            mock_embedding_service = MagicMock()
            mock_embedding_service.find_similar_trades = AsyncMock(
                return_value=[mock_trade_embedding1, mock_trade_embedding2]
            )
            mock_embedding.return_value = mock_embedding_service
            
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_trade1, mock_trade2]
            mock_session.execute = AsyncMock(return_value=mock_result)
            
            picker = FuzzyPicker(
                min_score_threshold=0.0,
                db_session=mock_session,
            )
            results = await picker.pick()
            
            assert len(results) == 1
            # 50% win rate should give ~0.0 adjustment
            adjustment = results[0].metadata.get("similarity_adjustment", 0.0)
            assert abs(adjustment) < 0.1  # Close to zero

    @pytest.mark.asyncio
    async def test_fuzzy_picker_similarity_adjustment_no_outcomes(self):
        """Test that trades without outcomes result in no adjustment."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market, \
             patch("src.services.discovery.pickers.fuzzy.TradeEmbeddingService") as mock_embedding:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            from src.services.market_data.provider import Quote, TechnicalIndicators
            
            quote = Quote(
                symbol="AAPL",
                price=150.0,
                volume=5_000_000,
            )
            
            indicators = TechnicalIndicators(
                symbol="AAPL",
                price=155.0,
                sma_20=150.0,
                sma_50=None,
                sma_200=None,
                rsi_14=55.0,
                volume_avg_20=3_000_000,
                volume_ratio=1.67,
            )
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            from uuid import uuid4
            
            mock_trade_embedding = MagicMock()
            mock_trade_embedding.trade_id = uuid4()
            
            mock_trade = MagicMock()
            mock_trade.id = mock_trade_embedding.trade_id
            mock_trade.outcome = None  # No outcome yet
            
            mock_embedding_service = MagicMock()
            mock_embedding_service.find_similar_trades = AsyncMock(return_value=[mock_trade_embedding])
            mock_embedding.return_value = mock_embedding_service
            
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_trade]
            mock_session.execute = AsyncMock(return_value=mock_result)
            
            picker = FuzzyPicker(
                min_score_threshold=0.0,
                db_session=mock_session,
            )
            results = await picker.pick()
            
            assert len(results) == 1
            # No outcomes should result in 0.0 adjustment
            assert "similarity_adjustment" not in results[0].metadata or results[0].metadata.get("similarity_adjustment") == 0.0

    @pytest.mark.asyncio
    async def test_fuzzy_picker_similarity_adjustment_error_handling(self):
        """Test that similarity search errors don't break the picker."""
        with patch("src.services.discovery.pickers.fuzzy.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.fuzzy.MarketDataService") as mock_market, \
             patch("src.services.discovery.pickers.fuzzy.TradeEmbeddingService") as mock_embedding:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            from src.services.market_data.provider import Quote, TechnicalIndicators
            
            quote = Quote(
                symbol="AAPL",
                price=150.0,
                volume=5_000_000,
            )
            
            indicators = TechnicalIndicators(
                symbol="AAPL",
                price=155.0,
                sma_20=150.0,
                sma_50=None,
                sma_200=None,
                rsi_14=55.0,
                volume_avg_20=3_000_000,
                volume_ratio=1.67,
            )
            
            mock_market.return_value.get_quote = AsyncMock(return_value=quote)
            mock_market.return_value.get_technical_indicators = AsyncMock(return_value=indicators)
            
            mock_embedding_service = MagicMock()
            mock_embedding_service.find_similar_trades = AsyncMock(side_effect=Exception("DB error"))
            mock_embedding.return_value = mock_embedding_service
            
            mock_session = MagicMock()
            
            picker = FuzzyPicker(
                min_score_threshold=0.0,
                db_session=mock_session,
            )
            # Should not raise exception, should continue with base score
            results = await picker.pick()
            
            assert len(results) == 1
            # Should still return a result even if similarity search fails
            assert results[0].symbol == "AAPL"


class TestLLMPicker:
    """Tests for LLMPicker."""

    @pytest.mark.asyncio
    async def test_llm_picker_name(self):
        """Test that picker has correct name."""
        picker = LLMPicker()
        assert picker.name == "llm"

    @pytest.mark.asyncio
    async def test_llm_picker_no_alpaca(self):
        """Test that picker returns empty list when Alpaca not configured."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source:
            mock_source.return_value.get_stocks.side_effect = ValueError("Alpaca not configured")
            picker = LLMPicker()
            results = await picker.pick()
            assert results == []

    @pytest.mark.asyncio
    async def test_llm_picker_parses_response(self):
        """Test that LLM response is parsed correctly."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL", "MSFT"])
            
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = """```json
[
    {"symbol": "AAPL", "score": 0.85, "reason": "Strong momentum"},
    {"symbol": "MSFT", "score": 0.70, "reason": "Good value"}
]
```"""
            mock_llm.return_value = mock_response
            
            picker = LLMPicker()
            results = await picker.pick()
            
            assert len(results) == 2
            assert results[0].symbol == "AAPL"
            assert results[0].score == 0.85
            assert results[1].symbol == "MSFT"
            assert results[1].score == 0.70
            # Results should be sorted by score
            assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_llm_picker_filters_invalid_symbols(self):
        """Test that symbols not in candidate list are filtered out."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL", "MSFT"])
            
            # Mock LLM response with invalid symbol
            mock_response = MagicMock()
            mock_response.content = """```json
[
    {"symbol": "AAPL", "score": 0.85, "reason": "Valid"},
    {"symbol": "INVALID", "score": 0.90, "reason": "Not in candidate list"}
]
```"""
            mock_llm.return_value = mock_response
            
            picker = LLMPicker()
            results = await picker.pick()
            
            # Should only include AAPL (INVALID filtered out)
            assert len(results) == 1
            assert results[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_llm_picker_handles_errors(self):
        """Test that errors are handled gracefully."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            mock_llm.side_effect = ValueError("LLM error")
            
            picker = LLMPicker()
            results = await picker.pick()
            
            # Should return empty list on error (graceful degradation)
            assert results == []

    @pytest.mark.asyncio
    async def test_llm_picker_uses_context(self):
        """Test that portfolio context is included in prompt."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:
            
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            
            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_llm.return_value = mock_response
            
            picker = LLMPicker()
            context = {
                "portfolio_positions": [
                    {"symbol": "TSLA", "quantity": 10, "current_value": 2000.0, "sector": "Technology"}
                ],
                "market_conditions": {"volatility": "High", "trend": "Bullish"}
            }
            
            results = await picker.pick(context=context)
            
            # Verify context was passed to LLM (check call args)
            assert mock_llm.called
            # The prompt should include portfolio and market conditions

    @pytest.mark.asyncio
    async def test_llm_picker_clamps_scores(self):
        """Test that scores are clamped to [0, 1] range."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:
            
            # Include both symbols in candidate list so they don't get filtered
            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL", "MSFT"])
            
            # Mock LLM response with out-of-range scores
            mock_response = MagicMock()
            mock_response.content = """```json
[
    {"symbol": "AAPL", "score": 1.5, "reason": "Too high"},
    {"symbol": "MSFT", "score": -0.5, "reason": "Too low"}
]
```"""
            mock_llm.return_value = mock_response
            
            picker = LLMPicker()
            results = await picker.pick()
            
            assert len(results) == 2
            assert results[0].score == 1.0  # Clamped from 1.5
            assert results[1].score == 0.0  # Clamped from -0.5

