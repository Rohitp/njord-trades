"""
Tests for discovery data sources.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.services.discovery.sources.alpaca import AlpacaAssetSource


class TestAlpacaAssetSource:
    """Tests for AlpacaAssetSource."""

    def test_init_without_api_keys(self):
        """Test initialization without API keys."""
        with patch("src.services.discovery.sources.alpaca.settings") as mock_settings:
            mock_settings.alpaca.api_key = ""
            mock_settings.alpaca.secret_key = ""
            source = AlpacaAssetSource()
            assert source.client is None

    def test_init_with_api_keys(self):
        """Test initialization with API keys."""
        with patch("src.services.discovery.sources.alpaca.settings") as mock_settings, \
             patch("src.services.discovery.sources.alpaca.TradingClient") as mock_client:
            mock_settings.alpaca.api_key = "test_key"
            mock_settings.alpaca.secret_key = "test_secret"
            mock_settings.alpaca.paper = True
            
            source = AlpacaAssetSource()
            assert source.client is not None
            mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tradable_symbols_success(self):
        """Test successful fetch of tradable symbols."""
        with patch("src.services.discovery.sources.alpaca.settings") as mock_settings, \
             patch("src.services.discovery.sources.alpaca.TradingClient") as mock_client_class:
            
            mock_settings.alpaca.api_key = "test_key"
            mock_settings.alpaca.secret_key = "test_secret"
            mock_settings.alpaca.paper = True
            
            # Mock asset objects
            mock_asset1 = MagicMock()
            mock_asset1.symbol = "AAPL"
            mock_asset2 = MagicMock()
            mock_asset2.symbol = "MSFT"
            
            mock_client = MagicMock()
            # get_all_assets returns an iterable/list of assets
            mock_client.get_all_assets.return_value = [mock_asset1, mock_asset2]
            mock_client_class.return_value = mock_client
            
            source = AlpacaAssetSource()
            
            from alpaca.trading.enums import AssetClass, AssetStatus
            
            symbols = await source.get_tradable_symbols(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            
            assert len(symbols) == 2
            assert "AAPL" in symbols
            assert "MSFT" in symbols

    @pytest.mark.asyncio
    async def test_get_tradable_symbols_no_client(self):
        """Test that ValueError is raised when Alpaca is not configured."""
        with patch("src.services.discovery.sources.alpaca.settings") as mock_settings:
            mock_settings.alpaca.api_key = ""
            mock_settings.alpaca.secret_key = ""
            
            source = AlpacaAssetSource()
            
            from alpaca.trading.enums import AssetClass, AssetStatus
            
            with pytest.raises(ValueError, match="Alpaca API keys not configured"):
                await source.get_tradable_symbols(
                    asset_class=AssetClass.US_EQUITY,
                    status=AssetStatus.ACTIVE,
                )

    @pytest.mark.asyncio
    async def test_get_stocks(self):
        """Test get_stocks convenience method."""
        with patch("src.services.discovery.sources.alpaca.settings") as mock_settings, \
             patch("src.services.discovery.sources.alpaca.TradingClient") as mock_client_class:
            
            mock_settings.alpaca.api_key = "test_key"
            mock_settings.alpaca.secret_key = "test_secret"
            mock_settings.alpaca.paper = True
            
            mock_asset = MagicMock()
            mock_asset.symbol = "AAPL"
            
            mock_client = MagicMock()
            mock_client.get_all_assets.return_value = [mock_asset]
            mock_client_class.return_value = mock_client
            
            source = AlpacaAssetSource()
            symbols = await source.get_stocks()
            
            assert len(symbols) == 1
            assert "AAPL" in symbols
            # Verify it called with correct defaults
            mock_client.get_all_assets.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tradable_symbols_retry_on_error(self):
        """Test that retry logic is applied on errors."""
        with patch("src.services.discovery.sources.alpaca.settings") as mock_settings, \
             patch("src.services.discovery.sources.alpaca.TradingClient") as mock_client_class, \
             patch("src.services.discovery.sources.alpaca.retry_with_backoff") as mock_retry:
            
            mock_settings.alpaca.api_key = "test_key"
            mock_settings.alpaca.secret_key = "test_secret"
            mock_settings.alpaca.paper = True
            
            # Mock retry decorator to just call the function
            def passthrough(func):
                return func
            
            mock_retry.return_value = passthrough
            
            mock_client = MagicMock()
            mock_client.get_all_assets.side_effect = Exception("API error")
            mock_client_class.return_value = mock_client
            
            source = AlpacaAssetSource()
            
            from alpaca.trading.enums import AssetClass, AssetStatus
            
            with pytest.raises(Exception, match="API error"):
                await source.get_tradable_symbols(
                    asset_class=AssetClass.US_EQUITY,
                    status=AssetStatus.ACTIVE,
                )

