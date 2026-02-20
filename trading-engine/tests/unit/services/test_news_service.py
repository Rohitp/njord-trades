"""
Tests for NewsService.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.services.market_data.news import (
    NewsItem,
    NewsService,
    RedditSentiment,
    classify_sentiment,
    NewsCache,
)


class TestSentimentClassification:
    """Tests for sentiment classification."""

    def test_positive_sentiment(self):
        """Test classification of positive text."""
        text = "Stock soars on strong earnings beat"
        assert classify_sentiment(text) == "positive"

    def test_negative_sentiment(self):
        """Test classification of negative text."""
        text = "Stock plunges on disappointing earnings miss"
        assert classify_sentiment(text) == "negative"

    def test_neutral_sentiment(self):
        """Test classification of neutral text."""
        text = "Company announces quarterly results"
        assert classify_sentiment(text) == "neutral"

    def test_mixed_sentiment_positive_wins(self):
        """Test that more positive words win."""
        text = "Stock rises and gains momentum despite minor decline"
        assert classify_sentiment(text) == "positive"

    def test_mixed_sentiment_negative_wins(self):
        """Test that more negative words win."""
        text = "Stock plunges and crashes, despite small gain"
        assert classify_sentiment(text) == "negative"


class TestNewsCache:
    """Tests for NewsCache."""

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = NewsCache(ttl_minutes=30)
        cache.set(["AAPL"], "yfinance", {"test": "data"})
        result = cache.get(["AAPL"], "yfinance")
        assert result == {"test": "data"}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = NewsCache(ttl_minutes=30)
        result = cache.get(["AAPL"], "yfinance")
        assert result is None

    def test_cache_different_sources(self):
        """Test that different sources have separate cache entries."""
        cache = NewsCache(ttl_minutes=30)
        cache.set(["AAPL"], "yfinance", {"source": "yfinance"})
        cache.set(["AAPL"], "google", {"source": "google"})

        assert cache.get(["AAPL"], "yfinance") == {"source": "yfinance"}
        assert cache.get(["AAPL"], "google") == {"source": "google"}

    def test_cache_different_symbols(self):
        """Test that different symbols have separate cache entries."""
        cache = NewsCache(ttl_minutes=30)
        cache.set(["AAPL"], "yfinance", {"symbol": "AAPL"})
        cache.set(["MSFT"], "yfinance", {"symbol": "MSFT"})

        assert cache.get(["AAPL"], "yfinance") == {"symbol": "AAPL"}
        assert cache.get(["MSFT"], "yfinance") == {"symbol": "MSFT"}

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = NewsCache(ttl_minutes=30)
        cache.set(["AAPL"], "yfinance", {"test": "data"})
        cache.clear()
        assert cache.get(["AAPL"], "yfinance") is None


class TestNewsItem:
    """Tests for NewsItem dataclass."""

    def test_news_item_creation(self):
        """Test creating a NewsItem."""
        item = NewsItem(
            title="Test headline",
            summary="Test summary",
            source="yfinance",
            published=datetime.now(),
            url="https://example.com",
            sentiment="positive",
            symbol="AAPL",
        )
        assert item.title == "Test headline"
        assert item.source == "yfinance"
        assert item.sentiment == "positive"

    def test_news_item_hash(self):
        """Test that NewsItem can be hashed."""
        item1 = NewsItem(
            title="Test",
            summary="",
            source="yfinance",
            published=None,
            url="https://example.com/1",
        )
        item2 = NewsItem(
            title="Different",
            summary="",
            source="google",
            published=None,
            url="https://example.com/1",  # Same URL
        )
        # Same URL = same hash
        assert hash(item1) == hash(item2)

    def test_news_item_equality(self):
        """Test NewsItem equality based on URL."""
        item1 = NewsItem(
            title="Test",
            summary="",
            source="yfinance",
            published=None,
            url="https://example.com/1",
        )
        item2 = NewsItem(
            title="Different",
            summary="",
            source="google",
            published=None,
            url="https://example.com/1",  # Same URL
        )
        assert item1 == item2


class TestNewsService:
    """Tests for NewsService."""

    @pytest.mark.asyncio
    async def test_get_news_empty_symbols(self):
        """Test that empty symbols list returns empty dict."""
        service = NewsService()
        result = await service.get_news([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_news_yfinance(self):
        """Test fetching news from yfinance."""
        service = NewsService(enabled_sources=["yfinance"])

        # Mock _fetch_yfinance_news directly since yfinance import is inside the method
        async def mock_fetch(*args, **kwargs):
            return {
                "AAPL": [
                    NewsItem(
                        title="Apple announces new product",
                        summary="Apple Inc announced...",
                        source="yfinance",
                        published=datetime.now(),
                        url="https://example.com/apple",
                        sentiment="positive",
                        symbol="AAPL",
                    )
                ]
            }

        service._fetch_yfinance_news = mock_fetch
        result = await service.get_news(["AAPL"])

        assert "AAPL" in result
        assert len(result["AAPL"]) > 0
        assert result["AAPL"][0].source == "yfinance"

    @pytest.mark.asyncio
    async def test_get_news_deduplication(self):
        """Test that duplicate URLs are filtered out."""
        service = NewsService(enabled_sources=["yfinance"])

        # Mock _fetch_yfinance_news to return duplicates
        async def mock_fetch(*args, **kwargs):
            return {
                "AAPL": [
                    NewsItem(
                        title="News 1",
                        summary="",
                        source="yfinance",
                        published=datetime.now(),
                        url="https://example.com/same",
                    ),
                    NewsItem(
                        title="News 2",
                        summary="",
                        source="yfinance",
                        published=datetime.now(),
                        url="https://example.com/same",  # Duplicate
                    ),
                ]
            }

        service._fetch_yfinance_news = mock_fetch
        result = await service.get_news(["AAPL"])

        # Should only have one item (deduplicated)
        assert len(result["AAPL"]) == 1

    @pytest.mark.asyncio
    async def test_get_news_limit_per_symbol(self):
        """Test that results are limited per symbol."""
        service = NewsService(enabled_sources=["yfinance"], max_items_per_symbol=2)

        # Mock to return many items
        async def mock_fetch(*args, **kwargs):
            return {
                "AAPL": [
                    NewsItem(
                        title=f"News {i}",
                        summary="",
                        source="yfinance",
                        published=datetime.now(),
                        url=f"https://example.com/{i}",
                    )
                    for i in range(10)
                ]
            }

        service._fetch_yfinance_news = mock_fetch
        result = await service.get_news(["AAPL"], limit_per_symbol=2)

        assert len(result["AAPL"]) == 2

    @pytest.mark.asyncio
    async def test_get_news_handles_errors(self):
        """Test that errors from sources don't crash the service."""
        service = NewsService(enabled_sources=["yfinance", "google"])

        # Mock yfinance to raise error
        async def mock_yfinance_error(*args, **kwargs):
            raise Exception("yfinance error")

        async def mock_google_success(*args, **kwargs):
            return {
                "AAPL": [
                    NewsItem(
                        title="Google news",
                        summary="",
                        source="google",
                        published=None,
                        url="https://example.com/google",
                    )
                ]
            }

        service._fetch_yfinance_news = mock_yfinance_error
        service._fetch_google_news = mock_google_success

        # Should still return results from working source
        result = await service.get_news(["AAPL"])
        assert "AAPL" in result
        assert len(result["AAPL"]) > 0


class TestRedditSentiment:
    """Tests for RedditSentiment."""

    def test_reddit_sentiment_defaults(self):
        """Test RedditSentiment default values."""
        sentiment = RedditSentiment(symbol="AAPL")
        assert sentiment.symbol == "AAPL"
        assert sentiment.mentions == 0
        assert sentiment.sentiment_score == 0.0
        assert sentiment.sentiment_label == "neutral"
        assert sentiment.top_posts == []

    def test_reddit_sentiment_with_data(self):
        """Test RedditSentiment with actual data."""
        sentiment = RedditSentiment(
            symbol="AAPL",
            mentions=50,
            sentiment_score=0.7,
            sentiment_label="positive",
            top_posts=["Post 1", "Post 2"],
        )
        assert sentiment.mentions == 50
        assert sentiment.sentiment_score == 0.7
        assert len(sentiment.top_posts) == 2


class TestNewsServiceReddit:
    """Tests for Reddit integration in NewsService."""

    @pytest.mark.asyncio
    async def test_get_reddit_sentiment_no_credentials(self):
        """Test that missing credentials return default sentiment."""
        service = NewsService(
            reddit_client_id="",
            reddit_client_secret="",
        )
        result = await service.get_reddit_sentiment(["AAPL"])

        assert "AAPL" in result
        assert result["AAPL"].mentions == 0
        assert result["AAPL"].sentiment_label == "neutral"

    @pytest.mark.asyncio
    async def test_get_reddit_sentiment_empty_symbols(self):
        """Test that empty symbols list returns empty dict."""
        service = NewsService()
        result = await service.get_reddit_sentiment([])
        assert result == {}
