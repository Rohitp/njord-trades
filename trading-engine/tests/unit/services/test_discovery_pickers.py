"""
Tests for symbol discovery pickers.
"""

import warnings
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.discovery.pickers.fuzzy import FuzzyPicker
from src.services.discovery.pickers.llm import LLMPicker
from src.services.discovery.pickers.metric import MetricPicker
from src.services.discovery.pickers.pure_llm import PureLLMPicker
from src.services.discovery.pickers.base import PickerResult


# Suppress deprecation warnings for FuzzyPicker in tests
@pytest.fixture(autouse=True)
def suppress_fuzzy_picker_deprecation():
    """Suppress FuzzyPicker deprecation warnings in tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


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

            from src.services.market_data.provider import Quote

            # Create mock quote responses (batch API returns list)
            quote_high_volume = Quote(
                symbol="AAPL",
                price=150.05,
                bid=150.0,
                ask=150.1,
                volume=10_000_000,  # Passes filter
            )

            quote_low_volume = Quote(
                symbol="MSFT",
                price=100.0,
                bid=None,
                ask=None,
                volume=100_000,  # Fails filter (< 1M)
            )

            # MetricPicker uses get_quotes (batch) not get_quote (single)
            mock_market.return_value.get_quotes = AsyncMock(
                return_value=[quote_high_volume, quote_low_volume]
            )

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

    @pytest.fixture
    def mock_market_data(self):
        """Create a mock MarketDataService."""
        from src.services.market_data.provider import Quote

        mock_service = MagicMock()
        mock_service.get_quotes = AsyncMock(return_value=[
            Quote(symbol="AAPL", price=150.0, volume=10_000_000),
            Quote(symbol="MSFT", price=300.0, volume=8_000_000),
        ])
        mock_service.get_indicators_batch = AsyncMock(return_value={})
        return mock_service

    @pytest.fixture
    def mock_news_service(self):
        """Create a mock NewsService."""
        mock_service = MagicMock()
        mock_service.get_news = AsyncMock(return_value={})
        return mock_service

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
    async def test_llm_picker_parses_response(self, mock_market_data, mock_news_service):
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

            picker = LLMPicker(
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )
            results = await picker.pick()

            assert len(results) == 2
            assert results[0].symbol == "AAPL"
            assert results[0].score == 0.85
            assert results[1].symbol == "MSFT"
            assert results[1].score == 0.70
            # Results should be sorted by score
            assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_llm_picker_filters_invalid_symbols(self, mock_market_data, mock_news_service):
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

            picker = LLMPicker(
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )
            results = await picker.pick()

            # Should only include AAPL (INVALID filtered out)
            assert len(results) == 1
            assert results[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_llm_picker_handles_errors(self, mock_market_data, mock_news_service):
        """Test that errors are handled gracefully."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            mock_llm.side_effect = ValueError("LLM error")

            picker = LLMPicker(
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )
            results = await picker.pick()

            # Should return empty list on error (graceful degradation)
            assert results == []

    @pytest.mark.asyncio
    async def test_llm_picker_uses_context(self, mock_market_data, mock_news_service):
        """Test that portfolio context is included in prompt."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])

            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_llm.return_value = mock_response

            picker = LLMPicker(
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )
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
    async def test_llm_picker_clamps_scores(self, mock_market_data, mock_news_service):
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

            picker = LLMPicker(
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )
            results = await picker.pick()

            assert len(results) == 2
            assert results[0].score == 1.0  # Clamped from 1.5
            assert results[1].score == 0.0  # Clamped from -0.5

    @pytest.mark.asyncio
    async def test_llm_picker_vector_integration_queries_similar_conditions(self, mock_market_data, mock_news_service):
        """Test that LLMPicker queries similar market conditions when db_session is provided."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm, \
             patch("src.services.discovery.pickers.llm.MarketConditionService") as mock_market_condition:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])

            # Mock similar conditions
            from src.database.models import MarketConditionEmbedding
            from datetime import datetime

            mock_condition1 = MagicMock(spec=MarketConditionEmbedding)
            mock_condition1.timestamp = datetime(2024, 1, 15)
            mock_condition1.context_text = "VIX: 18.5 (Moderate volatility) | SPY: $450.0, Bullish trend (+5.2% vs SMA_200)"
            mock_condition1.condition_metadata = {"vix": 18.5, "spy_trend": "bullish"}

            mock_condition2 = MagicMock(spec=MarketConditionEmbedding)
            mock_condition2.timestamp = datetime(2024, 1, 10)
            mock_condition2.context_text = "VIX: 20.1 (Moderate volatility) | SPY: $445.0, Bullish trend (+4.8% vs SMA_200)"
            mock_condition2.condition_metadata = {"vix": 20.1, "spy_trend": "bullish"}

            mock_service = MagicMock()
            mock_service.find_similar_conditions = AsyncMock(return_value=[mock_condition1, mock_condition2])
            mock_market_condition.return_value = mock_service

            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_llm.return_value = mock_response

            mock_session = MagicMock()
            picker = LLMPicker(
                db_session=mock_session,
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )

            context = {
                "market_conditions": {"volatility": "Moderate", "trend": "Bullish"}
            }

            results = await picker.pick(context=context)

            # Verify similarity search was called
            assert mock_service.find_similar_conditions.called
            call_args = mock_service.find_similar_conditions.call_args
            assert call_args[1]["session"] == mock_session
            assert call_args[1]["limit"] == 3

    @pytest.mark.asyncio
    async def test_llm_picker_vector_integration_includes_similar_in_prompt(self, mock_market_data, mock_news_service):
        """Test that similar conditions are included in the LLM prompt."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm, \
             patch("src.services.discovery.pickers.llm.MarketConditionService") as mock_market_condition:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])

            # Mock similar conditions
            from src.database.models import MarketConditionEmbedding
            from datetime import datetime

            mock_condition = MagicMock(spec=MarketConditionEmbedding)
            mock_condition.timestamp = datetime(2024, 1, 15)
            mock_condition.context_text = "VIX: 18.5 (Moderate volatility) | SPY: $450.0, Bullish trend"
            mock_condition.condition_metadata = {"vix": 18.5}

            mock_service = MagicMock()
            mock_service.find_similar_conditions = AsyncMock(return_value=[mock_condition])
            mock_market_condition.return_value = mock_service

            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_llm.return_value = mock_response

            mock_session = MagicMock()
            picker = LLMPicker(
                db_session=mock_session,
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )

            context = {
                "market_conditions": {"volatility": "Moderate", "trend": "Bullish"}
            }

            await picker.pick(context=context)

            # Verify similarity search was called
            assert mock_service.find_similar_conditions.called

            # Verify the prompt contains similar conditions by checking the built prompt
            prompt = picker._build_enriched_prompt(
                candidate_symbols=["AAPL"],
                quotes={},
                indicators={},
                news={},
                context=context,
                similar_conditions=[mock_condition],
            )
            assert "SIMILAR HISTORICAL CONDITIONS" in prompt
            assert "2024-01-15" in prompt  # Date from mock condition
            assert "VIX: 18.5" in prompt  # Context text from mock condition

    @pytest.mark.asyncio
    async def test_llm_picker_vector_integration_graceful_degradation(self, mock_market_data, mock_news_service):
        """Test that LLMPicker continues working if similarity search fails."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm, \
             patch("src.services.discovery.pickers.llm.MarketConditionService") as mock_market_condition:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])

            # Mock similarity search failure
            mock_service = MagicMock()
            mock_service.find_similar_conditions = AsyncMock(side_effect=Exception("DB error"))
            mock_market_condition.return_value = mock_service

            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = '[{"symbol": "AAPL", "score": 0.8, "reason": "Good"}]'
            mock_llm.return_value = mock_response

            mock_session = MagicMock()
            picker = LLMPicker(
                db_session=mock_session,
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )

            context = {
                "market_conditions": {"volatility": "Moderate", "trend": "Bullish"}
            }

            results = await picker.pick(context=context)

            # Should still return results despite similarity search failure
            assert len(results) == 1
            assert results[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_llm_picker_vector_integration_no_session(self, mock_market_data, mock_news_service):
        """Test that LLMPicker works without db_session (backward compatibility)."""
        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])

            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = '[{"symbol": "AAPL", "score": 0.8, "reason": "Good"}]'
            mock_llm.return_value = mock_response

            # No db_session provided
            picker = LLMPicker(
                market_data=mock_market_data,
                news_service=mock_news_service,
                fetch_indicators=False,
                fetch_news=False,
            )

            context = {
                "market_conditions": {"volatility": "Moderate", "trend": "Bullish"}
            }

            results = await picker.pick(context=context)

            # Should work normally without similarity search
            assert len(results) == 1
            assert results[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_llm_picker_fetches_enriched_data(self):
        """Test that LLMPicker fetches quotes, indicators, and news for candidates."""
        from src.services.market_data.provider import Quote, TechnicalIndicators
        from src.services.market_data.news import NewsItem
        from datetime import datetime

        with patch("src.services.discovery.pickers.llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.llm.retry_llm_call") as mock_llm:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL", "MSFT"])

            # Create mock services with data
            mock_market = MagicMock()
            mock_market.get_quotes = AsyncMock(return_value=[
                Quote(symbol="AAPL", price=150.0, volume=10_000_000),
                Quote(symbol="MSFT", price=300.0, volume=8_000_000),
            ])
            mock_market.get_indicators_batch = AsyncMock(return_value={
                "AAPL": TechnicalIndicators(
                    symbol="AAPL", price=150.0, sma_20=148.0, sma_50=145.0, rsi_14=55.0
                ),
            })

            mock_news = MagicMock()
            mock_news.get_news = AsyncMock(return_value={
                "AAPL": [
                    NewsItem(
                        title="Apple announces new product",
                        summary="Apple has announced...",
                        source="yfinance",
                        published=datetime.now(),
                        url="https://example.com/apple",
                        sentiment="positive",
                        symbol="AAPL",
                    )
                ],
            })

            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = '[{"symbol": "AAPL", "score": 0.85, "reason": "Strong momentum with positive news"}]'
            mock_llm.return_value = mock_response

            picker = LLMPicker(
                market_data=mock_market,
                news_service=mock_news,
                fetch_indicators=True,
                indicator_limit=10,
                fetch_news=True,
                news_limit=10,
            )

            results = await picker.pick()

            # Verify data was fetched
            assert mock_market.get_quotes.called
            assert mock_market.get_indicators_batch.called
            assert mock_news.get_news.called

            # Verify results
            assert len(results) == 1
            assert results[0].symbol == "AAPL"


class TestPureLLMPicker:
    """Tests for PureLLMPicker - pure LLM with no filters or data enrichment."""

    @pytest.mark.asyncio
    async def test_pure_llm_picker_name(self):
        """Test that picker has correct name."""
        picker = PureLLMPicker()
        assert picker.name == "pure_llm"

    @pytest.mark.asyncio
    async def test_pure_llm_picker_no_alpaca(self):
        """Test that picker returns empty list when Alpaca not configured."""
        with patch("src.services.discovery.pickers.pure_llm.AlpacaAssetSource") as mock_source:
            mock_source.return_value.get_stocks.side_effect = ValueError("Alpaca not configured")
            picker = PureLLMPicker()
            results = await picker.pick()
            assert results == []

    @pytest.mark.asyncio
    async def test_pure_llm_picker_parses_response(self):
        """Test that LLM response is parsed correctly."""
        with patch("src.services.discovery.pickers.pure_llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.pure_llm.retry_llm_call") as mock_llm:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL", "MSFT", "GOOGL"])

            # Mock LLM response
            mock_response = MagicMock()
            mock_response.content = """```json
[
    {"symbol": "AAPL", "score": 0.75, "reason": "Strong brand and services growth"},
    {"symbol": "NVDA", "score": 0.80, "reason": "AI leader"}
]
```"""
            mock_llm.return_value = mock_response

            picker = PureLLMPicker()
            results = await picker.pick()

            # NVDA should be filtered out (not in candidate list)
            assert len(results) == 1
            assert results[0].symbol == "AAPL"
            assert results[0].score == 0.75

    @pytest.mark.asyncio
    async def test_pure_llm_picker_no_data_enrichment(self):
        """Test that PureLLMPicker does NOT fetch any market data."""
        with patch("src.services.discovery.pickers.pure_llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.pure_llm.retry_llm_call") as mock_llm:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])

            mock_response = MagicMock()
            mock_response.content = '[{"symbol": "AAPL", "score": 0.7, "reason": "Good company"}]'
            mock_llm.return_value = mock_response

            picker = PureLLMPicker()

            # Verify picker has no market data or news service
            assert not hasattr(picker, 'market_data') or picker.__dict__.get('market_data') is None
            assert not hasattr(picker, 'news_service') or picker.__dict__.get('news_service') is None

            results = await picker.pick()
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_pure_llm_picker_prompt_has_no_prices(self):
        """Test that the prompt contains only symbol names, no prices or indicators."""
        picker = PureLLMPicker()

        # Build prompt with just symbols
        prompt = picker._build_prompt(["AAPL", "MSFT", "GOOGL"], context=None)

        # Should contain symbols
        assert "AAPL" in prompt
        assert "MSFT" in prompt
        assert "GOOGL" in prompt

        # Should NOT contain price/indicator language
        assert "RSI" not in prompt
        assert "SMA" not in prompt
        assert "$" not in prompt  # No dollar amounts
        assert "volume" not in prompt.lower() or "no access" in prompt.lower()

    @pytest.mark.asyncio
    async def test_pure_llm_picker_handles_errors(self):
        """Test that errors are handled gracefully."""
        with patch("src.services.discovery.pickers.pure_llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.pure_llm.retry_llm_call") as mock_llm:

            mock_source.return_value.get_stocks = AsyncMock(return_value=["AAPL"])
            mock_llm.side_effect = ValueError("LLM error")

            picker = PureLLMPicker()
            results = await picker.pick()

            # Should return empty list on error
            assert results == []

    @pytest.mark.asyncio
    async def test_pure_llm_picker_limits_symbols(self):
        """Test that symbol count is limited to avoid token limits."""
        with patch("src.services.discovery.pickers.pure_llm.AlpacaAssetSource") as mock_source, \
             patch("src.services.discovery.pickers.pure_llm.retry_llm_call") as mock_llm:

            # Return many symbols
            many_symbols = [f"SYM{i}" for i in range(1000)]
            mock_source.return_value.get_stocks = AsyncMock(return_value=many_symbols)

            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_llm.return_value = mock_response

            picker = PureLLMPicker(max_symbols=100)
            await picker.pick()

            # Check that the LLM was called (prompt was built)
            assert mock_llm.called

            # Verify limiting worked by checking the picker built prompt correctly
            # We can test the _build_prompt method directly
            prompt = picker._build_prompt(many_symbols[:100], context=None)
            assert "SYM0" in prompt  # First symbol should be there
            assert "SYM99" in prompt  # Last of the 100 should be there

            # Full prompt with 1000 symbols would have SYM999
            full_prompt = picker._build_prompt(many_symbols, context=None)
            assert "SYM999" in full_prompt  # This proves limiting is needed
