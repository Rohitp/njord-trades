"""
Unit tests for fundamentals providers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.market_data.fundamentals import (
    AlpacaFundamentalsProvider,
    CachedFundamentalsProvider,
    Fundamentals,
    FundamentalsProvider,
)


class TestFundamentals:
    """Test Fundamentals dataclass."""

    def test_fundamentals_creation(self):
        """Test creating Fundamentals object."""
        fund = Fundamentals(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
            beta=1.2,
            pe_ratio=30.5,
            dividend_yield=0.5,
        )

        assert fund.symbol == "AAPL"
        assert fund.sector == "Technology"
        assert fund.market_cap == 3_000_000_000_000
        assert fund.beta == 1.2

    def test_fundamentals_optional_fields(self):
        """Test Fundamentals with optional fields."""
        fund = Fundamentals(symbol="AAPL")

        assert fund.symbol == "AAPL"
        assert fund.sector is None
        assert fund.market_cap is None


class TestAlpacaFundamentalsProvider:
    """Test AlpacaFundamentalsProvider."""

    @pytest.fixture
    def provider(self):
        """Create AlpacaFundamentalsProvider."""
        # Mock AlpacaAssetSource where it's imported in __init__
        with patch("src.services.discovery.sources.alpaca.AlpacaAssetSource") as mock_source:
            mock_instance = MagicMock()
            mock_instance.client = None  # No client configured
            mock_source.return_value = mock_instance
            return AlpacaFundamentalsProvider()

    @pytest.mark.asyncio
    async def test_get_fundamentals_not_implemented(self, provider):
        """Test that Alpaca fundamentals returns None (not yet implemented)."""
        result = await provider.get_fundamentals("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_fundamentals_batch(self, provider):
        """Test batch fundamentals fetch."""
        results = await provider.get_fundamentals_batch(["AAPL", "MSFT"])
        assert isinstance(results, dict)
        assert len(results) == 0  # All return None currently


class TestCachedFundamentalsProvider:
    """Test CachedFundamentalsProvider."""

    @pytest.fixture
    def mock_fallback(self):
        """Mock fallback provider."""
        fallback = AsyncMock()
        fallback.get_fundamentals = AsyncMock(return_value=None)
        return fallback

    @pytest.fixture
    def provider(self, mock_fallback):
        """Create CachedFundamentalsProvider with mocked fallback."""
        return CachedFundamentalsProvider(fallback_provider=mock_fallback)

    @pytest.mark.asyncio
    async def test_get_fundamentals_cache_miss(self, provider, mock_fallback):
        """Test cache miss triggers fallback."""
        # First call - cache miss
        result = await provider.get_fundamentals("AAPL")
        assert result is None
        mock_fallback.get_fundamentals.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_get_fundamentals_cache_hit(self, provider, mock_fallback):
        """Test cache hit returns cached value."""
        # Create a fundamentals object
        fund = Fundamentals(symbol="AAPL", sector="Technology", market_cap=3_000_000_000_000)

        # Mock fallback to return it
        mock_fallback.get_fundamentals = AsyncMock(return_value=fund)

        # First call - cache miss, stores in cache
        result1 = await provider.get_fundamentals("AAPL")
        assert result1 == fund
        assert mock_fallback.get_fundamentals.call_count == 1

        # Second call - cache hit, doesn't call fallback
        result2 = await provider.get_fundamentals("AAPL")
        assert result2 == fund
        assert mock_fallback.get_fundamentals.call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_get_fundamentals_batch(self, provider, mock_fallback):
        """Test batch fundamentals fetch."""
        fund1 = Fundamentals(symbol="AAPL", sector="Technology")
        fund2 = Fundamentals(symbol="MSFT", sector="Technology")

        mock_fallback.get_fundamentals = AsyncMock(side_effect=[fund1, fund2])

        results = await provider.get_fundamentals_batch(["AAPL", "MSFT"])

        assert len(results) == 2
        assert "AAPL" in results
        assert "MSFT" in results
        assert results["AAPL"].sector == "Technology"

