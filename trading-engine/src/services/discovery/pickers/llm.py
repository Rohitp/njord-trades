"""
LLMPicker - LLM-powered context-aware symbol discovery with enriched market data.

Uses LLM reasoning to select symbols based on portfolio context,
market conditions, technical indicators, and news - with actual market data.

This is Stage 2 of the two-stage discovery architecture:
1. MetricPicker filters to ~100-200 liquid, tradable candidates
2. LLMPicker analyzes candidates with rich data (quotes, indicators, news)
"""

import asyncio
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.services.discovery.pickers.base import PickerResult, SymbolPicker
from src.services.discovery.pickers.metric import MetricPicker
from src.services.discovery.sources.alpaca import AlpacaAssetSource
from src.services.embeddings.market_condition import MarketConditionService
from src.services.market_data.news import NewsItem, NewsService
from src.services.market_data.provider import Quote, TechnicalIndicators
from src.services.market_data.service import MarketDataService
from src.utils.llm import parse_json_list
from src.utils.logging import get_logger
from src.utils.retry import retry_llm_call

log = get_logger(__name__)

# Try to import optional providers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


# System prompt for LLM symbol picker with enriched data
LLM_PICKER_SYSTEM_PROMPT = """You are a symbol discovery assistant for a quantitative trading system.

ROLE:
- Analyze real market data (prices, indicators, news) for candidate symbols
- Identify promising trading symbols that complement the current portfolio
- Consider diversification, momentum, and risk-adjusted opportunities
- Output ranked list of symbols with scores and reasoning

YOU HAVE ACCESS TO:
- Current prices and daily changes for all candidates
- Technical indicators (RSI, SMAs, volume) for top candidates
- Recent news headlines and sentiment for top candidates
- Portfolio context and sector exposure
- Similar historical market conditions

ANALYSIS FRAMEWORK:
1. MOMENTUM: Look for stocks with price above SMAs, healthy RSI (40-70), strong volume
2. VALUE: Consider oversold stocks (RSI < 30) that may be bottoming
3. NEWS: Factor in recent news sentiment - positive news with momentum is bullish
4. DIVERSIFICATION: Avoid over-concentration in sectors already heavily represented
5. RISK: Prefer stocks with clear technical setups over uncertain situations

CONSTRAINTS:
- Only recommend symbols from the provided candidate list
- Score should reflect opportunity quality (0.0 = poor, 1.0 = excellent)
- Consider portfolio diversification (avoid over-concentration)
- Weight news sentiment appropriately - not all news is equally important
- Prefer liquid symbols with clear momentum or value opportunities

OUTPUT FORMAT:
Respond with a JSON array of symbol recommendations. Each object must have:
{
    "symbol": "AAPL",
    "score": 0.0 to 1.0,
    "reason": "Brief explanation referencing actual data (price, RSI, news, etc.)"
}

Example response:
```json
[
    {"symbol": "AAPL", "score": 0.85, "reason": "$185.50 (+2.3%), RSI 58 healthy uptrend, above all SMAs, positive AI news"},
    {"symbol": "JPM", "score": 0.70, "reason": "$195.20 (+1.1%), financials underweight in portfolio, strong volume 1.5x avg"}
]
```

IMPORTANT:
- Return only symbols from the candidate list
- Reference actual data in your reasons (prices, RSI, SMAs, news)
- Scores should be realistic (most symbols should be 0.3-0.7 range)
- Only include symbols you genuinely recommend (skip weak candidates)
"""


class LLMPicker(SymbolPicker):
    """
    LLM-powered context-aware symbol picker with enriched market data.

    Two-stage discovery architecture:
    - Stage 1 (MetricPicker): Efficient quantitative pre-filtering
    - Stage 2 (LLMPicker): LLM analysis with real market data

    The LLM receives:
    - Current quotes (price, change %) for all candidates
    - Technical indicators (RSI, SMAs, volume) for top 30-50 symbols
    - News headlines and sentiment for top 20-30 symbols
    - Portfolio context and sector exposure
    - Similar historical market conditions

    Returns ranked list with scores and reasoning based on actual data.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_candidates: int | None = None,
        prefilter_with_metric: bool | None = None,
        metric_prefilter_limit: int | None = None,
        db_session: AsyncSession | None = None,
        provider: str | None = None,
        market_data: MarketDataService | None = None,
        news_service: NewsService | None = None,
        # Enrichment settings
        fetch_indicators: bool | None = None,
        indicator_limit: int | None = None,
        fetch_news: bool | None = None,
        news_limit: int | None = None,
    ):
        """
        Initialize LLMPicker.

        Args:
            model_name: LLM model to use (default: from config)
            max_candidates: Maximum number of candidate symbols to evaluate
            prefilter_with_metric: If True, run MetricPicker first to filter candidates
            metric_prefilter_limit: Top N symbols from MetricPicker to send to LLM
            db_session: Optional database session for vector similarity search
            provider: Override provider ("openai", "anthropic", "google", "deepseek", "auto")
            market_data: MarketDataService instance (default: creates new one)
            news_service: NewsService instance (default: creates new one)
            fetch_indicators: Whether to fetch technical indicators
            indicator_limit: Max symbols to fetch indicators for
            fetch_news: Whether to fetch news
            news_limit: Max symbols to fetch news for
        """
        # Load settings with defaults from config
        self.model_name = model_name or settings.discovery.llm_picker_model
        self.max_candidates = max_candidates or settings.discovery.llm_picker_max_candidates
        self.prefilter_with_metric = (
            prefilter_with_metric
            if prefilter_with_metric is not None
            else settings.discovery.llm_picker_prefilter
        )
        self.metric_prefilter_limit = (
            metric_prefilter_limit or settings.discovery.llm_picker_prefilter_limit
        )
        self.db_session = db_session
        self.provider_override = provider

        # Enrichment settings
        self.fetch_indicators = (
            fetch_indicators
            if fetch_indicators is not None
            else settings.discovery.llm_picker_fetch_indicators
        )
        self.indicator_limit = indicator_limit or settings.discovery.llm_picker_indicator_limit
        self.fetch_news = (
            fetch_news if fetch_news is not None else settings.discovery.llm_picker_fetch_news
        )
        self.news_limit = news_limit or settings.discovery.llm_picker_news_limit

        # Create LLM client
        self.llm = self._create_llm()

        # Services
        self.asset_source = AlpacaAssetSource()
        self.market_data = market_data or MarketDataService()
        self.news_service = news_service or NewsService()
        self.market_condition_service = MarketConditionService()
        self.metric_picker = MetricPicker() if self.prefilter_with_metric else None

    @property
    def name(self) -> str:
        """Picker name."""
        return "llm"

    def _create_llm(self) -> BaseChatModel:
        """Create the LangChain LLM client with fallback support."""
        provider = self._get_provider()

        try:
            return self._create_llm_for_provider(provider)
        except (ValueError, AttributeError) as e:
            if provider != settings.llm.fallback_provider:
                log.warning(
                    "llm_picker_provider_fallback",
                    primary_provider=provider,
                    fallback_provider=settings.llm.fallback_provider,
                    error=str(e),
                )
                return self._create_llm_for_provider(settings.llm.fallback_provider)
            raise

    def _get_provider(self) -> str:
        """Get the provider for LLMPicker."""
        if self.provider_override and self.provider_override != "auto":
            return self.provider_override.lower()

        if settings.llm.llm_picker_provider != "auto":
            return settings.llm.llm_picker_provider.lower()

        inferred = self._infer_provider(self.model_name)
        if inferred:
            return inferred

        return settings.llm.default_provider.lower()

    def _create_llm_for_provider(self, provider: str) -> BaseChatModel:
        """Create LLM client for a specific provider."""
        provider = provider.lower()
        temperature = 0.3

        if provider == "openai":
            if not settings.llm.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.llm.openai_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=temperature,
            )
        elif provider == "anthropic":
            if not settings.llm.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            return ChatAnthropic(
                model=self.model_name,
                api_key=settings.llm.anthropic_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=temperature,
            )
        elif provider == "google":
            if ChatGoogleGenerativeAI is None:
                raise ValueError(
                    "langchain-google-genai not installed. Run: uv sync --extra google"
                )
            if not settings.llm.google_api_key:
                raise ValueError("Google API key not configured")
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=settings.llm.google_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=temperature,
            )
        elif provider == "deepseek":
            if not settings.llm.deepseek_api_key:
                raise ValueError("DeepSeek API key not configured")
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.llm.deepseek_api_key,
                openai_api_base="https://api.deepseek.com/v1",
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _infer_provider(self, model_name: str) -> str | None:
        """Infer LLM provider from model name."""
        model_lower = model_name.lower()

        if model_lower.startswith(("gpt-", "o1-", "o3-")):
            return "openai"
        if model_lower.startswith("claude"):
            return "anthropic"
        if model_lower.startswith(("gemini-", "gemini-pro", "gemini-ultra")):
            return "google"
        if model_lower.startswith("deepseek"):
            return "deepseek"

        return None

    async def pick(self, context: dict | None = None) -> List[PickerResult]:
        """
        Pick symbols using LLM reasoning with enriched market data.

        Args:
            context: Optional context containing:
                - portfolio_positions: List of current positions
                - market_conditions: Dict describing market state
                - candidate_symbols: Optional list of symbols to evaluate

        Returns:
            List of PickerResult objects, sorted by score (highest first)
        """
        try:
            # Get candidate symbols
            if context and "candidate_symbols" in context:
                candidate_symbols = context["candidate_symbols"]
            else:
                candidate_symbols = await self.asset_source.get_stocks()

            if not candidate_symbols:
                log.warning("llm_picker_no_candidates")
                return []

            # Pre-filter with MetricPicker to reduce candidates
            user_supplied_candidates = context and "candidate_symbols" in context
            should_prefilter = (
                self.prefilter_with_metric
                and self.metric_picker
                and len(candidate_symbols) > self.metric_prefilter_limit
                and not user_supplied_candidates
            )

            if should_prefilter:
                log.info(
                    "llm_picker_prefiltering",
                    initial_count=len(candidate_symbols),
                    prefilter_limit=self.metric_prefilter_limit,
                )
                try:
                    metric_context = {"candidate_symbols": candidate_symbols}
                    metric_results = await self.metric_picker.pick(context=metric_context)
                    prefiltered_symbols = [
                        r.symbol for r in metric_results[: self.metric_prefilter_limit]
                    ]
                    candidate_symbols = [s for s in candidate_symbols if s in prefiltered_symbols]
                    log.info("llm_picker_prefiltered", prefiltered_count=len(candidate_symbols))
                except Exception as e:
                    log.warning("llm_picker_prefilter_failed", error=str(e))
                    candidate_symbols = candidate_symbols[: self.max_candidates]

            # Final limit to max_candidates
            candidate_symbols = candidate_symbols[: self.max_candidates]

            if not candidate_symbols:
                log.warning("llm_picker_no_candidates_after_filtering")
                return []

            log.info("llm_picker_starting", candidate_count=len(candidate_symbols))

        except ValueError as e:
            log.warning("llm_picker_no_alpaca", error=str(e))
            return []

        # Fetch enriched market data in parallel
        quotes, indicators, news, similar_conditions = await self._fetch_enriched_data(
            candidate_symbols, context
        )

        # Build prompt with enriched data
        user_prompt = self._build_enriched_prompt(
            candidate_symbols=candidate_symbols,
            quotes=quotes,
            indicators=indicators,
            news=news,
            context=context,
            similar_conditions=similar_conditions,
        )

        try:
            # Call LLM with retry logic
            response = await retry_llm_call(
                self.llm.ainvoke,
                [SystemMessage(content=LLM_PICKER_SYSTEM_PROMPT), HumanMessage(content=user_prompt)],
            )

            # Parse response
            response_text = response.content if hasattr(response, "content") else str(response)
            parsed_list = parse_json_list(response_text, context="LLMPicker response")

            # Convert to PickerResult objects
            results = []
            for item in parsed_list:
                symbol = item.get("symbol", "").upper()
                if symbol not in candidate_symbols:
                    log.debug("llm_picker_invalid_symbol", symbol=symbol)
                    continue

                score = float(item.get("score", 0.0))
                reason = item.get("reason", "No reason provided")

                results.append(
                    PickerResult(
                        symbol=symbol,
                        score=max(0.0, min(1.0, score)),
                        reason=reason,
                        metadata={"picker": "llm", "model": self.model_name},
                    )
                )

            results.sort(key=lambda x: x.score, reverse=True)

            log.info(
                "llm_picker_complete",
                results_count=len(results),
                candidates=len(candidate_symbols),
                quotes_fetched=len(quotes),
                indicators_fetched=len(indicators),
                news_symbols=len(news),
            )
            return results

        except Exception as e:
            log.error("llm_picker_error", error=str(e), exc_info=True)
            return []

    async def _fetch_enriched_data(
        self,
        candidate_symbols: List[str],
        context: dict | None = None,
    ) -> tuple[dict[str, Quote], dict[str, TechnicalIndicators], dict[str, list[NewsItem]], list]:
        """
        Fetch enriched market data for candidates in parallel.

        Returns:
            Tuple of (quotes, indicators, news, similar_conditions)
        """
        quotes: dict[str, Quote] = {}
        indicators: dict[str, TechnicalIndicators] = {}
        news: dict[str, list[NewsItem]] = {}
        similar_conditions: list = []

        # Define async tasks
        async def fetch_quotes():
            nonlocal quotes
            try:
                quote_list = await self.market_data.get_quotes(candidate_symbols)
                quotes = {q.symbol: q for q in quote_list}
                log.debug("llm_picker_quotes_fetched", count=len(quotes))
            except Exception as e:
                log.warning("llm_picker_quotes_failed", error=str(e))

        async def fetch_indicators():
            nonlocal indicators
            if not self.fetch_indicators:
                return
            try:
                # Only fetch for top N candidates
                symbols_for_indicators = candidate_symbols[: self.indicator_limit]
                indicators = await self.market_data.get_indicators_batch(symbols_for_indicators)
                log.debug("llm_picker_indicators_fetched", count=len(indicators))
            except Exception as e:
                log.warning("llm_picker_indicators_failed", error=str(e))

        async def fetch_news():
            nonlocal news
            if not self.fetch_news:
                return
            try:
                # Only fetch for top N candidates
                symbols_for_news = candidate_symbols[: self.news_limit]
                news = await self.news_service.get_news(symbols_for_news, limit_per_symbol=3)
                log.debug(
                    "llm_picker_news_fetched",
                    symbols=len(news),
                    total_items=sum(len(items) for items in news.values()),
                )
            except Exception as e:
                log.warning("llm_picker_news_failed", error=str(e))

        async def fetch_similar_conditions():
            nonlocal similar_conditions
            if not self.db_session:
                return
            try:
                current_context = self._build_market_context_text(context)
                if current_context:
                    similar_conditions = await self.market_condition_service.find_similar_conditions(
                        context_text=current_context,
                        limit=3,
                        min_similarity=0.7,
                        session=self.db_session,
                    )
                    log.debug("llm_picker_similar_conditions", count=len(similar_conditions))
            except Exception as e:
                log.warning("llm_picker_similarity_search_failed", error=str(e))

        # Run all fetches in parallel
        await asyncio.gather(
            fetch_quotes(),
            fetch_indicators(),
            fetch_news(),
            fetch_similar_conditions(),
            return_exceptions=True,
        )

        return quotes, indicators, news, similar_conditions

    def _build_enriched_prompt(
        self,
        candidate_symbols: List[str],
        quotes: dict[str, Quote],
        indicators: dict[str, TechnicalIndicators],
        news: dict[str, list[NewsItem]],
        context: dict | None = None,
        similar_conditions: list | None = None,
    ) -> str:
        """
        Build prompt with enriched market data.

        Args:
            candidate_symbols: List of symbols to evaluate
            quotes: Dict of symbol -> Quote
            indicators: Dict of symbol -> TechnicalIndicators
            news: Dict of symbol -> list of NewsItem
            context: Optional context (portfolio, market conditions)
            similar_conditions: List of similar historical conditions

        Returns:
            Formatted prompt with real market data
        """
        lines = []

        # Portfolio context
        if context and "portfolio_positions" in context:
            positions = context["portfolio_positions"]
            if positions:
                lines.append("CURRENT PORTFOLIO:")
                total_value = sum(pos.get("current_value", 0.0) for pos in positions)
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    quantity = pos.get("quantity", 0)
                    value = pos.get("current_value", 0.0)
                    sector = pos.get("sector", "Unknown")
                    pct = (value / total_value * 100) if total_value > 0 else 0
                    lines.append(
                        f"  - {symbol}: {quantity} shares (${value:,.2f}, {pct:.1f}%), Sector: {sector}"
                    )

                # Calculate sector exposure
                sector_values: dict[str, float] = {}
                for pos in positions:
                    sector = pos.get("sector", "Unknown")
                    sector_values[sector] = sector_values.get(sector, 0) + pos.get(
                        "current_value", 0
                    )
                if sector_values and total_value > 0:
                    lines.append("\n  Sector Exposure:")
                    for sector, value in sorted(
                        sector_values.items(), key=lambda x: x[1], reverse=True
                    ):
                        pct = value / total_value * 100
                        lines.append(f"    - {sector}: {pct:.1f}%")
            else:
                lines.append("CURRENT PORTFOLIO: Empty (starting fresh)")
        else:
            lines.append("CURRENT PORTFOLIO: Not provided")

        # Current market conditions
        if context and "market_conditions" in context:
            market = context["market_conditions"]
            lines.append("\nMARKET CONDITIONS:")
            if "volatility" in market:
                lines.append(f"  - Volatility: {market['volatility']}")
            if "trend" in market:
                lines.append(f"  - Overall Trend: {market['trend']}")
            if "sector_rotation" in market:
                lines.append(f"  - Sector Rotation: {market['sector_rotation']}")

        # Similar historical conditions
        if similar_conditions:
            lines.append("\nSIMILAR HISTORICAL CONDITIONS:")
            lines.append(
                "(Market conditions from the past similar to now - use for pattern recognition)"
            )
            for i, condition in enumerate(similar_conditions[:3], 1):
                timestamp = (
                    condition.timestamp.strftime("%Y-%m-%d")
                    if condition.timestamp
                    else "Unknown"
                )
                context_summary = (
                    condition.context_text[:150] + "..."
                    if len(condition.context_text) > 150
                    else condition.context_text
                )
                lines.append(f"  {i}. {timestamp}: {context_summary}")

        # Candidate symbols with enriched data
        lines.append(f"\n{'='*60}")
        lines.append(f"CANDIDATE SYMBOLS WITH MARKET DATA ({len(candidate_symbols)} total):")
        lines.append("=" * 60)

        for i, symbol in enumerate(candidate_symbols, 1):
            symbol_lines = []

            # Quote data
            quote = quotes.get(symbol)
            if quote:
                price = quote.price
                # Calculate price change if we have previous close (not available in Quote)
                # For now, just show price
                symbol_lines.append(f"{i}. {symbol} - ${price:.2f}")

                if quote.volume:
                    volume_str = self._format_volume(quote.volume)
                    symbol_lines[0] += f" | Vol: {volume_str}"
            else:
                symbol_lines.append(f"{i}. {symbol} - (no quote data)")

            # Technical indicators
            ind = indicators.get(symbol)
            if ind:
                ind_parts = []

                # RSI
                if ind.rsi_14 is not None:
                    rsi_status = self._interpret_rsi(ind.rsi_14)
                    ind_parts.append(f"RSI: {ind.rsi_14:.1f} ({rsi_status})")

                # SMA positioning
                if ind.price and any([ind.sma_20, ind.sma_50, ind.sma_200]):
                    sma_status = self._interpret_sma_position(
                        ind.price, ind.sma_20, ind.sma_50, ind.sma_200
                    )
                    ind_parts.append(sma_status)

                # Volume ratio
                if ind.volume_ratio is not None:
                    vol_status = "high" if ind.volume_ratio > 1.5 else (
                        "low" if ind.volume_ratio < 0.5 else "normal"
                    )
                    ind_parts.append(f"Vol ratio: {ind.volume_ratio:.1f}x ({vol_status})")

                if ind_parts:
                    symbol_lines.append(f"   Indicators: {' | '.join(ind_parts)}")

            # News
            symbol_news = news.get(symbol, [])
            if symbol_news:
                # Show top 2 news items
                for news_item in symbol_news[:2]:
                    sentiment_emoji = {
                        "positive": "+",
                        "negative": "-",
                        "neutral": "~",
                    }.get(news_item.sentiment or "neutral", "~")
                    title = news_item.title[:80] + "..." if len(news_item.title) > 80 else news_item.title
                    symbol_lines.append(f"   News [{sentiment_emoji}]: {title}")

            lines.extend(symbol_lines)
            lines.append("")  # Blank line between symbols

        # Task instructions
        lines.append("=" * 60)
        lines.append("TASK:")
        lines.append(
            "Analyze the market data above and select the most promising trading opportunities."
        )
        lines.append("Consider:")
        lines.append("  1. Technical setup (RSI, SMAs, volume)")
        lines.append("  2. News sentiment and catalysts")
        lines.append("  3. Portfolio diversification needs")
        lines.append("  4. Similar historical market conditions")
        lines.append("")
        lines.append("Return a JSON array with your top recommendations (score 0.0-1.0).")
        lines.append("Reference actual data in your reasons.")

        return "\n".join(lines)

    def _format_volume(self, volume: int) -> str:
        """Format volume with K/M/B suffix."""
        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.1f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}K"
        return str(volume)

    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value."""
        if rsi < 30:
            return "oversold"
        elif rsi < 40:
            return "weak"
        elif rsi <= 60:
            return "neutral"
        elif rsi <= 70:
            return "strong"
        else:
            return "overbought"

    def _interpret_sma_position(
        self,
        price: float,
        sma_20: float | None,
        sma_50: float | None,
        sma_200: float | None,
    ) -> str:
        """Interpret price position relative to SMAs."""
        above = []
        below = []

        if sma_20 is not None:
            if price > sma_20:
                above.append("SMA20")
            else:
                below.append("SMA20")

        if sma_50 is not None:
            if price > sma_50:
                above.append("SMA50")
            else:
                below.append("SMA50")

        if sma_200 is not None:
            if price > sma_200:
                above.append("SMA200")
            else:
                below.append("SMA200")

        if above and not below:
            return f"Above {', '.join(above)}"
        elif below and not above:
            return f"Below {', '.join(below)}"
        elif above and below:
            return f"Above {', '.join(above)}, Below {', '.join(below)}"
        return "No SMA data"

    def _build_market_context_text(self, context: dict | None = None) -> str:
        """
        Build text representation of current market conditions for similarity search.

        Args:
            context: Optional context containing market conditions

        Returns:
            Formatted text describing current market conditions
        """
        parts = []

        if context and "market_conditions" in context:
            market = context["market_conditions"]
            if "volatility" in market:
                parts.append(f"Volatility: {market['volatility']}")
            if "trend" in market:
                parts.append(f"Trend: {market['trend']}")
            if "sector_rotation" in market:
                parts.append(f"Sector Rotation: {market['sector_rotation']}")

        if not parts:
            return ""

        return " | ".join(parts)

    # Keep the old method for backwards compatibility
    def _build_user_prompt(
        self,
        candidate_symbols: List[str],
        context: dict | None = None,
        similar_conditions: list | None = None,
    ) -> str:
        """
        Build user prompt (legacy method - uses enriched prompt internally).

        Deprecated: Use _build_enriched_prompt instead.
        """
        # For backwards compatibility, just build a simple prompt
        return self._build_enriched_prompt(
            candidate_symbols=candidate_symbols,
            quotes={},
            indicators={},
            news={},
            context=context,
            similar_conditions=similar_conditions,
        )
