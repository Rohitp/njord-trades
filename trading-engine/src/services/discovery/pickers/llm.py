"""
LLMPicker - LLM-powered context-aware symbol discovery.

Uses LLM reasoning to select symbols based on portfolio context,
market conditions, and trading opportunities.
"""

from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.services.discovery.pickers.base import PickerResult, SymbolPicker
from src.services.discovery.sources.alpaca import AlpacaAssetSource
from src.services.market_data.service import MarketDataService
from src.utils.llm import parse_json_list
from src.utils.logging import get_logger
from src.utils.retry import retry_llm_call

log = get_logger(__name__)


# System prompt for LLM symbol picker
LLM_PICKER_SYSTEM_PROMPT = """You are a symbol discovery assistant for a quantitative trading system.

ROLE:
- Analyze market conditions and portfolio context
- Identify promising trading symbols that complement the current portfolio
- Consider diversification, momentum, and risk-adjusted opportunities
- Output ranked list of symbols with scores and reasoning

CONSTRAINTS:
- Only recommend symbols from the provided candidate list
- Score should reflect opportunity quality (0.0 = poor, 1.0 = excellent)
- Consider portfolio diversification (avoid over-concentration)
- Consider market conditions (volatility, trends, sector rotation)
- Prefer liquid symbols with clear momentum or value opportunities

OUTPUT FORMAT:
Respond with a JSON array of symbol recommendations. Each object must have:
{
    "symbol": "AAPL",
    "score": 0.0 to 1.0,
    "reason": "Brief explanation of why this symbol is promising"
}

Example response:
```json
[
    {"symbol": "AAPL", "score": 0.85, "reason": "Strong momentum above SMA_200, RSI 65 indicates healthy uptrend, complements tech exposure"},
    {"symbol": "JPM", "score": 0.70, "reason": "Financials underrepresented in portfolio, positive momentum, good liquidity"}
]
```

IMPORTANT:
- Return only symbols from the candidate list
- Scores should be realistic (most symbols should be 0.3-0.7 range)
- Only include symbols you genuinely recommend (skip weak candidates)
"""


class LLMPicker(SymbolPicker):
    """
    LLM-powered context-aware symbol picker.

    Uses LLM reasoning to select symbols based on:
    - Portfolio context (current positions, sector exposure)
    - Market conditions (volatility, trends, sector rotation)
    - Trading opportunities (momentum, value, diversification)

    Returns ranked list with scores and reasoning.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_candidates: int = 50,  # Limit candidates to avoid token limits
    ):
        """
        Initialize LLMPicker.

        Args:
            model_name: LLM model to use (default: from config)
            max_candidates: Maximum number of candidate symbols to evaluate
        """
        self.model_name = model_name or settings.discovery.llm_picker_model
        self.max_candidates = max_candidates
        self.llm = self._create_llm()

        self.asset_source = AlpacaAssetSource()
        self.market_data = MarketDataService()

    @property
    def name(self) -> str:
        """Picker name."""
        return "llm"

    def _create_llm(self) -> BaseChatModel:
        """Create the LangChain LLM client."""
        provider = self._infer_provider(self.model_name)

        if provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.llm.openai_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.3,  # Slightly higher for creative selection
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                api_key=settings.llm.anthropic_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.3,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _infer_provider(self, model_name: str) -> str:
        """Infer LLM provider from model name."""
        model_lower = model_name.lower()

        if model_lower.startswith(("gpt-", "o1-", "o3-")):
            return "openai"
        elif model_lower.startswith("claude-"):
            return "anthropic"
        else:
            # Default to configured provider
            return settings.llm.default_provider

    async def pick(self, context: dict | None = None) -> List[PickerResult]:
        """
        Pick symbols using LLM reasoning.

        Args:
            context: Optional context containing:
                - portfolio_positions: List of current positions
                - market_conditions: Dict describing market state
                - candidate_symbols: Optional list of symbols to evaluate (if None, fetches all)

        Returns:
            List of PickerResult objects, sorted by score (highest first)
        """
        try:
            # Get candidate symbols
            if context and "candidate_symbols" in context:
                candidate_symbols = context["candidate_symbols"][: self.max_candidates]
            else:
                # Fetch all tradable stocks
                candidate_symbols = await self.asset_source.get_stocks()
                candidate_symbols = candidate_symbols[: self.max_candidates]

            if not candidate_symbols:
                log.warning("llm_picker_no_candidates")
                return []

            log.info("llm_picker_starting", candidate_count=len(candidate_symbols))
        except ValueError as e:
            log.warning("llm_picker_no_alpaca", error=str(e))
            return []  # Can't fetch symbols without Alpaca

        # Build prompt with context
        user_prompt = self._build_user_prompt(candidate_symbols, context)

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
                    continue  # Skip symbols not in candidate list

                score = float(item.get("score", 0.0))
                reason = item.get("reason", "No reason provided")

                results.append(
                    PickerResult(
                        symbol=symbol,
                        score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                        reason=reason,
                        metadata={"picker": "llm", "model": self.model_name},
                    )
                )

            # Sort by score (highest first)
            results.sort(key=lambda x: x.score, reverse=True)

            log.info("llm_picker_complete", results_count=len(results), candidates=len(candidate_symbols))
            return results

        except Exception as e:
            log.error("llm_picker_error", error=str(e), exc_info=True)
            return []  # Return empty on error (graceful degradation)

    def _build_user_prompt(self, candidate_symbols: List[str], context: dict | None = None) -> str:
        """
        Build user prompt with portfolio context and market conditions.

        Args:
            candidate_symbols: List of symbols to evaluate
            context: Optional context (portfolio, market conditions)

        Returns:
            Formatted prompt string
        """
        lines = []

        # Portfolio context
        if context and "portfolio_positions" in context:
            positions = context["portfolio_positions"]
            if positions:
                lines.append("CURRENT PORTFOLIO:")
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    quantity = pos.get("quantity", 0)
                    value = pos.get("current_value", 0.0)
                    sector = pos.get("sector", "Unknown")
                    lines.append(f"  - {symbol}: {quantity} shares (${value:,.2f}), Sector: {sector}")
            else:
                lines.append("CURRENT PORTFOLIO: Empty (starting fresh)")
        else:
            lines.append("CURRENT PORTFOLIO: Not provided")

        # Market conditions
        if context and "market_conditions" in context:
            market = context["market_conditions"]
            lines.append("\nMARKET CONDITIONS:")
            if "volatility" in market:
                lines.append(f"  - Volatility: {market['volatility']}")
            if "trend" in market:
                lines.append(f"  - Overall Trend: {market['trend']}")
            if "sector_rotation" in market:
                lines.append(f"  - Sector Rotation: {market['sector_rotation']}")

        # Candidate symbols
        lines.append(f"\nCANDIDATE SYMBOLS ({len(candidate_symbols)} total):")
        # Group symbols for readability (10 per line)
        symbol_lines = []
        for i in range(0, len(candidate_symbols), 10):
            chunk = candidate_symbols[i : i + 10]
            symbol_lines.append(", ".join(chunk))
        lines.append("  " + "\n  ".join(symbol_lines))

        # Instructions
        lines.append("\nTASK:")
        lines.append("Analyze the portfolio context and market conditions above.")
        lines.append("Select the most promising symbols from the candidate list.")
        lines.append("Consider diversification, momentum, and risk-adjusted opportunities.")
        lines.append("Return a JSON array with your top recommendations (score 0.0-1.0).")

        return "\n".join(lines)

