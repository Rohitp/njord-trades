"""Multi-source news service for market data enrichment.

This module provides a NewsService that aggregates news from multiple sources:
- yfinance: Company-specific news from Yahoo Finance
- Google Search: Broader news coverage via web search
- Reddit: Sentiment from r/wallstreetbets, r/stocks, r/investing

Features:
- Parallel fetch across all sources
- 30-minute caching to reduce API calls
- Simple keyword-based sentiment classification
- Deduplication by URL
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class NewsItem:
    """A single news item from any source.

    Attributes:
        title: News headline
        summary: Brief summary or first paragraph
        source: Origin of the news ("yfinance", "google", "reddit")
        published: Publication timestamp (may be None for some sources)
        url: Link to the full article
        sentiment: Simple classification ("positive", "negative", "neutral")
        symbol: Associated stock symbol
    """

    title: str
    summary: str
    source: str
    published: datetime | None
    url: str
    sentiment: str | None = None
    symbol: str | None = None

    def __hash__(self) -> int:
        """Hash by URL for deduplication."""
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NewsItem):
            return False
        return self.url == other.url


@dataclass
class RedditSentiment:
    """Reddit sentiment data for a symbol.

    Attributes:
        symbol: Stock symbol
        mentions: Number of mentions across subreddits
        sentiment_score: -1.0 to 1.0 scale (negative to positive)
        sentiment_label: "positive", "negative", or "neutral"
        top_posts: List of top post titles mentioning the symbol
    """

    symbol: str
    mentions: int = 0
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    top_posts: list[str] = field(default_factory=list)


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

# Keywords for simple sentiment classification
POSITIVE_KEYWORDS = {
    "surge",
    "soar",
    "jump",
    "rally",
    "gain",
    "rise",
    "up",
    "high",
    "beat",
    "exceed",
    "outperform",
    "bullish",
    "growth",
    "profit",
    "record",
    "strong",
    "upgrade",
    "buy",
    "breakthrough",
    "innovation",
    "success",
    "boost",
    "accelerate",
    "expand",
    "positive",
    "optimistic",
    "moon",
    "rocket",
    "tendies",
    "calls",
    "yolo",
    "diamond hands",
}

NEGATIVE_KEYWORDS = {
    "drop",
    "fall",
    "decline",
    "plunge",
    "crash",
    "sink",
    "down",
    "low",
    "miss",
    "fail",
    "underperform",
    "bearish",
    "loss",
    "warning",
    "weak",
    "downgrade",
    "sell",
    "concern",
    "risk",
    "layoff",
    "cut",
    "negative",
    "pessimistic",
    "puts",
    "short",
    "bag holder",
    "rug pull",
    "dump",
}


def classify_sentiment(text: str) -> str:
    """Classify text sentiment using keyword matching.

    Args:
        text: Text to analyze (title + summary)

    Returns:
        "positive", "negative", or "neutral"
    """
    text_lower = text.lower()
    positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    return "neutral"


# =============================================================================
# NEWS CACHE
# =============================================================================


class NewsCache:
    """Simple in-memory cache with TTL for news items."""

    def __init__(self, ttl_minutes: int = 30):
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)

    def _make_key(self, symbols: list[str], source: str) -> str:
        """Create cache key from symbols and source."""
        sorted_symbols = sorted(symbols)
        key_str = f"{source}:{','.join(sorted_symbols)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, symbols: list[str], source: str) -> Any | None:
        """Get cached value if not expired."""
        key = self._make_key(symbols, source)
        if key in self._cache:
            cached_at, value = self._cache[key]
            if datetime.now() - cached_at < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, symbols: list[str], source: str, value: Any) -> None:
        """Cache a value."""
        key = self._make_key(symbols, source)
        self._cache[key] = (datetime.now(), value)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


# =============================================================================
# NEWS SERVICE
# =============================================================================


class NewsService:
    """Multi-source news aggregation service.

    Fetches news from yfinance, Google Search, and Reddit in parallel,
    then aggregates and deduplicates the results.

    Example:
        service = NewsService()
        news = await service.get_news(["AAPL", "MSFT"], limit_per_symbol=5)
        # news = {"AAPL": [NewsItem(...), ...], "MSFT": [...]}

        sentiment = await service.get_reddit_sentiment(["AAPL", "TSLA"])
        # sentiment = {"AAPL": RedditSentiment(...), ...}
    """

    def __init__(
        self,
        cache_ttl_minutes: int | None = None,
        max_items_per_symbol: int | None = None,
        enabled_sources: list[str] | None = None,
        reddit_client_id: str | None = None,
        reddit_client_secret: str | None = None,
        reddit_user_agent: str | None = None,
    ):
        """Initialize NewsService.

        Args:
            cache_ttl_minutes: Cache TTL in minutes (default: from config or 30)
            max_items_per_symbol: Max news items per symbol (default: from config or 5)
            enabled_sources: List of sources to use (default: from config)
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
        """
        # Load from config with fallbacks
        news_config = getattr(settings, "news", None)

        self._cache_ttl = cache_ttl_minutes or (
            getattr(news_config, "cache_ttl_minutes", 30) if news_config else 30
        )
        self._max_items = max_items_per_symbol or (
            getattr(news_config, "max_items_per_symbol", 5) if news_config else 5
        )
        self._enabled_sources = enabled_sources or (
            getattr(news_config, "enabled_sources", ["yfinance", "google", "reddit"])
            if news_config
            else ["yfinance", "google", "reddit"]
        )

        # Reddit credentials
        self._reddit_client_id = reddit_client_id or (
            getattr(news_config, "reddit_client_id", "") if news_config else ""
        )
        self._reddit_client_secret = reddit_client_secret or (
            getattr(news_config, "reddit_client_secret", "") if news_config else ""
        )
        self._reddit_user_agent = reddit_user_agent or (
            getattr(news_config, "reddit_user_agent", "trading-engine/1.0")
            if news_config
            else "trading-engine/1.0"
        )

        self._cache = NewsCache(ttl_minutes=self._cache_ttl)
        self._reddit_client: Any = None

    async def get_news(
        self,
        symbols: list[str],
        limit_per_symbol: int | None = None,
    ) -> dict[str, list[NewsItem]]:
        """Fetch news for multiple symbols from all enabled sources.

        Args:
            symbols: List of stock symbols
            limit_per_symbol: Max items per symbol (overrides default)

        Returns:
            Dict mapping symbols to lists of NewsItem
        """
        if not symbols:
            return {}

        limit = limit_per_symbol or self._max_items
        results: dict[str, list[NewsItem]] = {s: [] for s in symbols}

        # Fetch from all enabled sources in parallel
        tasks = []
        if "yfinance" in self._enabled_sources:
            tasks.append(self._fetch_yfinance_news(symbols))
        if "google" in self._enabled_sources:
            tasks.append(self._fetch_google_news(symbols))
        if "reddit" in self._enabled_sources:
            tasks.append(self._fetch_reddit_news(symbols))

        if not tasks:
            return results

        source_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results from all sources
        for source_result in source_results:
            if isinstance(source_result, Exception):
                log.warning("news_source_error", error=str(source_result))
                continue
            if isinstance(source_result, dict):
                for symbol, items in source_result.items():
                    if symbol in results:
                        results[symbol].extend(items)

        # Deduplicate by URL and limit per symbol
        for symbol in results:
            seen_urls: set[str] = set()
            unique_items: list[NewsItem] = []
            # Sort by published date (newest first) if available
            sorted_items = sorted(
                results[symbol],
                key=lambda x: x.published or datetime.min,
                reverse=True,
            )
            for item in sorted_items:
                if item.url not in seen_urls and len(unique_items) < limit:
                    seen_urls.add(item.url)
                    unique_items.append(item)
            results[symbol] = unique_items

        log.info(
            "news_fetched",
            symbols=len(symbols),
            total_items=sum(len(items) for items in results.values()),
        )
        return results

    async def get_reddit_sentiment(
        self,
        symbols: list[str],
    ) -> dict[str, RedditSentiment]:
        """Get Reddit sentiment analysis for symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbols to RedditSentiment
        """
        if not symbols:
            return {}

        # Check cache
        cached = self._cache.get(symbols, "reddit_sentiment")
        if cached is not None:
            log.debug("reddit_sentiment_cache_hit", symbols=len(symbols))
            return cached

        results: dict[str, RedditSentiment] = {}

        try:
            reddit = await self._get_reddit_client()
            if reddit is None:
                log.debug("reddit_client_unavailable")
                return {s: RedditSentiment(symbol=s) for s in symbols}

            # Search across trading subreddits
            subreddits = ["wallstreetbets", "stocks", "investing"]

            for symbol in symbols:
                mentions = 0
                positive_count = 0
                negative_count = 0
                top_posts: list[str] = []

                for subreddit_name in subreddits:
                    try:
                        subreddit = await asyncio.to_thread(
                            lambda name=subreddit_name: reddit.subreddit(name)
                        )
                        # Search for symbol mentions
                        search_results = await asyncio.to_thread(
                            lambda sub=subreddit, sym=symbol: list(
                                sub.search(f"${sym} OR {sym}", limit=10, time_filter="week")
                            )
                        )

                        for post in search_results:
                            mentions += 1
                            text = f"{post.title} {post.selftext}"
                            sentiment = classify_sentiment(text)
                            if sentiment == "positive":
                                positive_count += 1
                            elif sentiment == "negative":
                                negative_count += 1
                            if len(top_posts) < 3:
                                top_posts.append(post.title[:100])

                    except Exception as e:
                        log.debug(
                            "reddit_subreddit_error",
                            subreddit=subreddit_name,
                            symbol=symbol,
                            error=str(e),
                        )

                # Calculate sentiment score
                total = positive_count + negative_count
                if total > 0:
                    score = (positive_count - negative_count) / total
                else:
                    score = 0.0

                if score > 0.2:
                    label = "positive"
                elif score < -0.2:
                    label = "negative"
                else:
                    label = "neutral"

                results[symbol] = RedditSentiment(
                    symbol=symbol,
                    mentions=mentions,
                    sentiment_score=score,
                    sentiment_label=label,
                    top_posts=top_posts,
                )

        except Exception as e:
            log.warning("reddit_sentiment_error", error=str(e))
            results = {s: RedditSentiment(symbol=s) for s in symbols}

        # Cache results
        self._cache.set(symbols, "reddit_sentiment", results)
        return results

    # =========================================================================
    # PRIVATE METHODS - SOURCE FETCHERS
    # =========================================================================

    async def _fetch_yfinance_news(
        self,
        symbols: list[str],
    ) -> dict[str, list[NewsItem]]:
        """Fetch news from yfinance.

        Uses ticker.news to get company-specific news.
        """
        # Check cache
        cached = self._cache.get(symbols, "yfinance")
        if cached is not None:
            log.debug("yfinance_news_cache_hit", symbols=len(symbols))
            return cached

        results: dict[str, list[NewsItem]] = {s: [] for s in symbols}

        try:
            import yfinance as yf

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    # yfinance news is a list of dicts
                    news_data = await asyncio.to_thread(lambda t=ticker: t.news or [])

                    for item in news_data[: self._max_items]:
                        # Parse yfinance news item
                        title = item.get("title", "")
                        summary = item.get("summary", "") or item.get("description", "")
                        url = item.get("link", "") or item.get("url", "")
                        published = None

                        # Parse timestamp
                        pub_time = item.get("providerPublishTime")
                        if pub_time:
                            try:
                                published = datetime.fromtimestamp(pub_time)
                            except (ValueError, TypeError, OSError):
                                pass

                        if title and url:
                            sentiment = classify_sentiment(f"{title} {summary}")
                            results[symbol].append(
                                NewsItem(
                                    title=title,
                                    summary=summary[:500] if summary else "",
                                    source="yfinance",
                                    published=published,
                                    url=url,
                                    sentiment=sentiment,
                                    symbol=symbol,
                                )
                            )

                except Exception as e:
                    log.debug("yfinance_symbol_error", symbol=symbol, error=str(e))

        except ImportError:
            log.warning("yfinance_not_installed")
        except Exception as e:
            log.warning("yfinance_news_error", error=str(e))

        # Cache results
        self._cache.set(symbols, "yfinance", results)
        log.debug(
            "yfinance_news_fetched",
            symbols=len(symbols),
            items=sum(len(v) for v in results.values()),
        )
        return results

    async def _fetch_google_news(
        self,
        symbols: list[str],
    ) -> dict[str, list[NewsItem]]:
        """Fetch news using Google Search.

        Uses googlesearch-python for broader news coverage.
        """
        # Check cache
        cached = self._cache.get(symbols, "google")
        if cached is not None:
            log.debug("google_news_cache_hit", symbols=len(symbols))
            return cached

        results: dict[str, list[NewsItem]] = {s: [] for s in symbols}

        try:
            from googlesearch import search as google_search

            for symbol in symbols:
                try:
                    # Search for recent stock news
                    query = f"{symbol} stock news"
                    search_results = await asyncio.to_thread(
                        lambda q=query: list(google_search(q, num_results=self._max_items))
                    )

                    for url in search_results:
                        # Extract domain as pseudo-title
                        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
                        domain = domain_match.group(1) if domain_match else "Unknown"

                        # Create news item with URL as identifier
                        results[symbol].append(
                            NewsItem(
                                title=f"{symbol} news from {domain}",
                                summary=f"Stock news article from {domain}",
                                source="google",
                                published=None,
                                url=url,
                                sentiment="neutral",  # Can't classify without fetching content
                                symbol=symbol,
                            )
                        )

                except Exception as e:
                    log.debug("google_symbol_error", symbol=symbol, error=str(e))

        except ImportError:
            log.debug("googlesearch_not_installed")
        except Exception as e:
            log.warning("google_news_error", error=str(e))

        # Cache results
        self._cache.set(symbols, "google", results)
        log.debug(
            "google_news_fetched",
            symbols=len(symbols),
            items=sum(len(v) for v in results.values()),
        )
        return results

    async def _fetch_reddit_news(
        self,
        symbols: list[str],
    ) -> dict[str, list[NewsItem]]:
        """Fetch news/posts from Reddit.

        Searches r/wallstreetbets, r/stocks, r/investing for symbol mentions.
        """
        # Check cache
        cached = self._cache.get(symbols, "reddit")
        if cached is not None:
            log.debug("reddit_news_cache_hit", symbols=len(symbols))
            return cached

        results: dict[str, list[NewsItem]] = {s: [] for s in symbols}

        try:
            reddit = await self._get_reddit_client()
            if reddit is None:
                return results

            subreddits = ["wallstreetbets", "stocks", "investing"]

            for symbol in symbols:
                seen_urls: set[str] = set()

                for subreddit_name in subreddits:
                    if len(results[symbol]) >= self._max_items:
                        break

                    try:
                        subreddit = await asyncio.to_thread(
                            lambda name=subreddit_name: reddit.subreddit(name)
                        )
                        search_results = await asyncio.to_thread(
                            lambda sub=subreddit, sym=symbol: list(
                                sub.search(f"${sym}", limit=5, time_filter="week")
                            )
                        )

                        for post in search_results:
                            url = f"https://reddit.com{post.permalink}"
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)

                            # Parse timestamp
                            published = None
                            try:
                                published = datetime.fromtimestamp(post.created_utc)
                            except (ValueError, TypeError, OSError):
                                pass

                            title = post.title[:200]
                            summary = (post.selftext[:500] if post.selftext else "")

                            sentiment = classify_sentiment(f"{title} {summary}")

                            results[symbol].append(
                                NewsItem(
                                    title=title,
                                    summary=summary,
                                    source="reddit",
                                    published=published,
                                    url=url,
                                    sentiment=sentiment,
                                    symbol=symbol,
                                )
                            )

                            if len(results[symbol]) >= self._max_items:
                                break

                    except Exception as e:
                        log.debug(
                            "reddit_subreddit_error",
                            subreddit=subreddit_name,
                            symbol=symbol,
                            error=str(e),
                        )

        except Exception as e:
            log.warning("reddit_news_error", error=str(e))

        # Cache results
        self._cache.set(symbols, "reddit", results)
        log.debug(
            "reddit_news_fetched",
            symbols=len(symbols),
            items=sum(len(v) for v in results.values()),
        )
        return results

    async def _get_reddit_client(self) -> Any:
        """Get or create Reddit client (PRAW).

        Returns None if credentials are not configured or PRAW is not installed.
        """
        if self._reddit_client is not None:
            return self._reddit_client

        if not self._reddit_client_id or not self._reddit_client_secret:
            log.debug("reddit_credentials_not_configured")
            return None

        try:
            import praw

            self._reddit_client = praw.Reddit(
                client_id=self._reddit_client_id,
                client_secret=self._reddit_client_secret,
                user_agent=self._reddit_user_agent,
            )
            log.info("reddit_client_initialized")
            return self._reddit_client

        except ImportError:
            log.debug("praw_not_installed")
            return None
        except Exception as e:
            log.warning("reddit_client_error", error=str(e))
            return None


# Singleton instance
news_service = NewsService()
