"""
Base agent protocol and shared utilities.

All trading agents implement the BaseAgent protocol, which defines a single
`run` method that takes TradingState and returns modified TradingState.

This allows the LangGraph workflow to treat all agents uniformly while
each agent has its own specialized logic and LLM prompts.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.utils.langfuse import langfuse_trace, langfuse_generation
from src.utils.logging import get_logger
from src.utils.retry import retry_llm_call
from src.workflows.state import TradingState

log = get_logger(__name__)

# Try to import optional providers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.

    Each agent:
    1. Receives the full TradingState
    2. Reads the inputs it needs (e.g., signals, portfolio snapshot)
    3. Calls the LLM with a specialized prompt
    4. Parses the response into typed output (e.g., RiskAssessment)
    5. Appends output to the appropriate state list
    6. Returns the modified state
    """

    # Subclasses set these
    name: str = "BaseAgent"
    model_name: str = settings.llm.data_agent_model  # Default model

    def __init__(self, model_name: str | None = None, provider: str | None = None):
        """
        Initialize the agent with an LLM client.

        Args:
            model_name: Override the default model for this agent
            provider: Override the provider ("openai", "anthropic", "google", "deepseek", or "auto")
        """
        self.model_name = model_name or self.model_name
        self.provider_override = provider
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseChatModel:
        """
        Create the LangChain LLM client for this agent with fallback support.

        Provider selection priority:
        1. Explicit provider override (if provided)
        2. Agent-specific provider from config (e.g., data_agent_provider)
        3. Infer from model name
        4. Default provider from config

        Falls back to fallback_provider if primary fails.
        """
        # Determine which provider to use
        provider = self._get_provider()
        
        # Try to create LLM with primary provider
        try:
            return self._create_llm_for_provider(provider)
        except (ValueError, AttributeError) as e:
            # If primary provider fails, try fallback
            if provider != settings.llm.fallback_provider:
                log.warning(
                    "llm_provider_fallback",
                    agent=self.name,
                    primary_provider=provider,
                    fallback_provider=settings.llm.fallback_provider,
                    error=str(e),
                )
                return self._create_llm_for_provider(settings.llm.fallback_provider)
            raise

    def _get_provider(self) -> str:
        """Get the provider for this agent."""
        # 1. Explicit override
        if self.provider_override and self.provider_override != "auto":
            return self.provider_override.lower()
        
        # 2. Agent-specific provider from config
        agent_provider_map = {
            "DataAgent": settings.llm.data_agent_provider,
            "RiskManager": settings.llm.risk_agent_provider,
            "Validator": settings.llm.validator_provider,
            "MetaAgent": settings.llm.meta_agent_provider,
        }
        
        agent_provider = agent_provider_map.get(self.name, "auto")
        if agent_provider != "auto":
            return agent_provider.lower()
        
        # 3. Infer from model name
        inferred = self._infer_provider(self.model_name)
        if inferred:
            return inferred
        
        # 4. Default provider
        return settings.llm.default_provider.lower()

    def _create_llm_for_provider(self, provider: str) -> BaseChatModel:
        """Create LLM client for a specific provider."""
        provider = provider.lower()
        
        if provider == "openai":
            if not settings.llm.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.llm.openai_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )
        elif provider == "anthropic":
            if not settings.llm.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            return ChatAnthropic(
                model=self.model_name,
                api_key=settings.llm.anthropic_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )
        elif provider == "google":
            if ChatGoogleGenerativeAI is None:
                raise ValueError("langchain-google-genai not installed. Run: uv sync --extra google")
            if not settings.llm.google_api_key:
                raise ValueError("Google API key not configured")
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=settings.llm.google_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )
        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            if not settings.llm.deepseek_api_key:
                raise ValueError("DeepSeek API key not configured")
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.llm.deepseek_api_key,
                openai_api_base="https://api.deepseek.com/v1",  # DeepSeek API endpoint
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _infer_provider(self, model_name: str) -> str | None:
        """
        Infer the LLM provider from the model name.

        Args:
            model_name: The model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")

        Returns:
            Provider name or None if ambiguous
        """
        model_lower = model_name.lower()

        # OpenAI models
        if model_lower.startswith(("gpt-", "o1-", "o3-")):
            return "openai"

        # Anthropic models
        if model_lower.startswith("claude"):
            return "anthropic"

        # Google/Gemini models
        if model_lower.startswith(("gemini-", "gemini-pro", "gemini-ultra")):
            return "google"

        # DeepSeek models
        if model_lower.startswith("deepseek"):
            return "deepseek"

        # Ambiguous - return None to use default
        return None

    @abstractmethod
    async def run(self, state: TradingState) -> TradingState:
        """
        Execute this agent's logic on the trading state.

        Args:
            state: Current trading state with inputs from previous agents

        Returns:
            Modified state with this agent's outputs appended
        """
        pass

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        state: TradingState | None = None,
    ) -> str:
        """
        Call the LLM with system and user prompts.

        Includes retry logic with exponential backoff for:
        - Rate limit errors (429)
        - Server errors (500, 502, 503, 504)
        - Network timeouts

        Args:
            system_prompt: The agent's role and instructions
            user_prompt: The specific request with context data
            state: Optional TradingState for trace metadata (cycle_id, trace_id)

        Returns:
            The LLM's response text

        Raises:
            Exception: If all retries exhausted or non-retryable error
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Prepare Langfuse metadata
        trace_metadata = {
            "agent": self.name,
            "model": self.model_name,
            "provider": getattr(self.llm, "_llm_type", "unknown"),
        }
        
        # Add cycle context if available
        if state:
            trace_metadata["cycle_id"] = str(state.cycle_id)
            trace_metadata["cycle_type"] = state.cycle_type
            if state.trace_id:
                trace_metadata["trace_id"] = state.trace_id
            if state.symbols:
                trace_metadata["symbols"] = state.symbols

        # Convert messages to dict format for Langfuse
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async def _invoke():
            response = await self.llm.ainvoke(messages)
            return response.content

        # Use Langfuse tracing if enabled
        session_id = str(state.cycle_id) if state else None
        trace_name = f"{self.name}_llm_call"
        
        with langfuse_trace(
            name=trace_name,
            metadata=trace_metadata,
            session_id=session_id,
        ) as trace:
            # Use retry wrapper with context for logging
            result = await retry_llm_call(
                _invoke,
                context=f"{self.name} LLM call",
            )

            # Log generation to Langfuse
            langfuse_generation(
                trace=trace,
                name=f"{self.name}_generation",
                model=self.model_name,
                input_messages=input_messages,
                output=result,
                metadata={
                    **trace_metadata,
                    "response_length": len(result) if result else 0,
                },
            )

        # Log provider used (extract from llm object if possible)
        provider_used = getattr(self.llm, "_llm_type", "unknown")
        log.debug(
            "llm_call_complete",
            agent=self.name,
            model=self.model_name,
            provider=provider_used,
            response_length=len(result) if result else 0,
        )

        return result

    def _format_portfolio_context(self, state: TradingState) -> str:
        """
        Format portfolio snapshot as context string for prompts.

        Provides the LLM with current portfolio state including cash,
        positions, and sector exposure.
        """
        if not state.portfolio_snapshot:
            return "Portfolio snapshot not available"

        snapshot = state.portfolio_snapshot

        # Format numbers with commas and 2 decimal places
        cash_str = f"${snapshot.cash:,.2f}" if snapshot.cash is not None else "N/A"
        total_value_str = f"${snapshot.total_value:,.2f}" if snapshot.total_value is not None else "N/A"
        deployed_capital_str = f"${snapshot.deployed_capital:,.2f}" if snapshot.deployed_capital is not None else "N/A"

        lines = [
            f"Cash Available: {cash_str}",
            f"Total Portfolio Value: {total_value_str}",
            f"Deployed Capital: {deployed_capital_str}",
        ]

        if snapshot.positions:
            lines.append("\nCurrent Positions:")
            for symbol, details in snapshot.positions.items():
                qty = details.get("quantity", 0)
                value = details.get("value", 0)
                sector = details.get("sector", "Unknown")
                value_str = f"${value:,.2f}" if value is not None else "N/A"
                lines.append(f"  {symbol}: {qty} shares, {value_str} ({sector})")

        if snapshot.sector_exposure:
            lines.append("\nSector Exposure:")
            for sector, exposure in snapshot.sector_exposure.items():
                exposure_str = f"${exposure:,.2f}" if exposure is not None else "N/A"
                pct = (exposure / snapshot.total_value * 100) if snapshot.total_value and snapshot.total_value > 0 else 0
                lines.append(f"  {sector}: {exposure_str} ({pct:.1f}%)")

        return "\n".join(lines)
