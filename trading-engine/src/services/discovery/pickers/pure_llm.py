"""
PureLLMPicker - LLM-only symbol discovery with no quantitative filters.

The LLM receives only symbol names and must rely on its training knowledge
to select promising stocks. No market data, no indicators, no news.

Use case: Testing LLM's inherent stock-picking ability without data enrichment.
"""

from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.services.discovery.pickers.base import PickerResult, SymbolPicker
from src.services.discovery.sources.alpaca import AlpacaAssetSource
from src.utils.llm import parse_json_list
from src.utils.logging import get_logger
from src.utils.retry import retry_llm_call

log = get_logger(__name__)

# Try to import optional providers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


PURE_LLM_SYSTEM_PROMPT = """You are a stock picker for a trading system.

You will receive a list of stock symbols. Using ONLY your knowledge of these companies,
select the most promising ones for trading opportunities.

You have NO access to:
- Current prices or price changes
- Technical indicators (RSI, moving averages)
- Recent news
- Real-time market data

You must rely on:
- Your knowledge of the companies (business model, sector, competitive position)
- General market dynamics and sector trends
- Company fundamentals you know from training data

CONSTRAINTS:
- Only recommend symbols from the provided list
- Score 0.0 to 1.0 (higher = more promising)
- Be selective - only pick stocks you genuinely believe have potential
- Scores should be realistic (most 0.3-0.7 range)

OUTPUT FORMAT:
Return a JSON array:
```json
[
    {"symbol": "AAPL", "score": 0.75, "reason": "Strong brand, services growth, large cash position"},
    {"symbol": "NVDA", "score": 0.80, "reason": "AI/GPU leader, data center demand"}
]
```

IMPORTANT:
- You are picking based on company quality, not real-time data
- Be honest about uncertainty - your knowledge has a cutoff date
- Focus on fundamentally strong businesses
"""


class PureLLMPicker(SymbolPicker):
    """
    Pure LLM symbol picker with no quantitative filters or data enrichment.

    The LLM receives only:
    - List of tradable symbol names
    - Optional portfolio context

    The LLM does NOT receive:
    - Current prices or quotes
    - Technical indicators
    - News headlines
    - Volume data

    This picker tests the LLM's ability to pick stocks based purely on
    its training knowledge of companies.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_symbols: int = 500,  # Limit symbols sent to LLM (token limits)
        db_session: AsyncSession | None = None,
        provider: str | None = None,
    ):
        """
        Initialize PureLLMPicker.

        Args:
            model_name: LLM model to use (default: from config)
            max_symbols: Maximum symbols to send to LLM (default: 500)
            db_session: Optional database session (not used, for interface compatibility)
            provider: Override provider ("openai", "anthropic", "google", "auto")
        """
        self.model_name = model_name or settings.discovery.llm_picker_model
        self.max_symbols = max_symbols
        self.db_session = db_session
        self.provider_override = provider
        self.llm = self._create_llm()
        self.asset_source = AlpacaAssetSource()

    @property
    def name(self) -> str:
        """Picker name."""
        return "pure_llm"

    def _create_llm(self) -> BaseChatModel:
        """Create the LangChain LLM client."""
        provider = self._get_provider()

        try:
            return self._create_llm_for_provider(provider)
        except (ValueError, AttributeError) as e:
            if provider != settings.llm.fallback_provider:
                log.warning(
                    "pure_llm_picker_provider_fallback",
                    primary_provider=provider,
                    fallback_provider=settings.llm.fallback_provider,
                    error=str(e),
                )
                return self._create_llm_for_provider(settings.llm.fallback_provider)
            raise

    def _get_provider(self) -> str:
        """Get the provider for this picker."""
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

        return None

    async def pick(self, context: dict | None = None) -> List[PickerResult]:
        """
        Pick symbols using only LLM knowledge.

        Args:
            context: Optional context containing:
                - candidate_symbols: Optional list of symbols to evaluate
                - portfolio_positions: Current portfolio (for context only)

        Returns:
            List of PickerResult objects, sorted by score (highest first)
        """
        try:
            # Get candidate symbols
            if context and "candidate_symbols" in context:
                symbols = context["candidate_symbols"]
            else:
                symbols = await self.asset_source.get_stocks()

            if not symbols:
                log.warning("pure_llm_picker_no_symbols")
                return []

            # Limit symbols to avoid token limits
            if len(symbols) > self.max_symbols:
                log.info(
                    "pure_llm_picker_limiting_symbols",
                    total=len(symbols),
                    limit=self.max_symbols,
                )
                symbols = symbols[: self.max_symbols]

            log.info("pure_llm_picker_starting", symbol_count=len(symbols))

        except ValueError as e:
            log.warning("pure_llm_picker_no_alpaca", error=str(e))
            return []

        # Build simple prompt with just symbols
        user_prompt = self._build_prompt(symbols, context)

        try:
            # Call LLM
            response = await retry_llm_call(
                self.llm.ainvoke,
                [
                    SystemMessage(content=PURE_LLM_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ],
            )

            # Parse response
            response_text = response.content if hasattr(response, "content") else str(response)
            parsed_list = parse_json_list(response_text, context="PureLLMPicker response")

            # Convert to PickerResult objects
            results = []
            for item in parsed_list:
                symbol = item.get("symbol", "").upper()
                if symbol not in symbols:
                    log.debug("pure_llm_picker_invalid_symbol", symbol=symbol)
                    continue

                score = float(item.get("score", 0.0))
                reason = item.get("reason", "No reason provided")

                results.append(
                    PickerResult(
                        symbol=symbol,
                        score=max(0.0, min(1.0, score)),
                        reason=reason,
                        metadata={"picker": "pure_llm", "model": self.model_name},
                    )
                )

            results.sort(key=lambda x: x.score, reverse=True)

            log.info(
                "pure_llm_picker_complete",
                results_count=len(results),
                symbols_provided=len(symbols),
            )
            return results

        except Exception as e:
            log.error("pure_llm_picker_error", error=str(e), exc_info=True)
            return []

    def _build_prompt(self, symbols: List[str], context: dict | None = None) -> str:
        """
        Build prompt with just symbols and optional portfolio context.

        Args:
            symbols: List of symbol names
            context: Optional context

        Returns:
            Simple prompt string
        """
        lines = []

        # Portfolio context (if provided)
        if context and "portfolio_positions" in context:
            positions = context["portfolio_positions"]
            if positions:
                lines.append("CURRENT PORTFOLIO (for diversification context):")
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    sector = pos.get("sector", "Unknown")
                    lines.append(f"  - {symbol} ({sector})")
                lines.append("")

        # Symbol list
        lines.append(f"AVAILABLE SYMBOLS ({len(symbols)} total):")
        lines.append("")

        # Group symbols for readability (20 per line)
        for i in range(0, len(symbols), 20):
            chunk = symbols[i : i + 20]
            lines.append(", ".join(chunk))

        lines.append("")
        lines.append("TASK:")
        lines.append("Select the most promising symbols based on your knowledge of these companies.")
        lines.append("You have NO access to current prices, news, or technical data.")
        lines.append("Rely only on your understanding of company fundamentals and market position.")
        lines.append("")
        lines.append("Return a JSON array with your picks (include score and reason for each).")

        return "\n".join(lines)
