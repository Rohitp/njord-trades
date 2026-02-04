"""
Tests for symbol discovery pickers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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

