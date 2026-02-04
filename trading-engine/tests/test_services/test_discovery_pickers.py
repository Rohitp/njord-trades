"""
Tests for symbol discovery pickers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.discovery.pickers.fuzzy import FuzzyPicker
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

