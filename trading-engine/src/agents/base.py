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
from src.workflows.state import TradingState


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

    def __init__(self, model_name: str | None = None):
        """
        Initialize the agent with an LLM client.

        Args:
            model_name: Override the default model for this agent
        """
        self.model_name = model_name or self.model_name
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseChatModel:
        """
        Create the LangChain LLM client for this agent.

        Infers provider from model name:
        - claude-* → Anthropic
        - gpt-*, o1-*, o3-* → OpenAI

        Falls back to settings.llm.default_provider if model name is ambiguous.
        """
        provider = self._infer_provider(self.model_name)

        if provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.llm.openai_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                api_key=settings.llm.anthropic_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _infer_provider(self, model_name: str) -> str:
        """
        Infer the LLM provider from the model name.

        Args:
            model_name: The model identifier (e.g., "claude-3-5-sonnet-20241022")

        Returns:
            "anthropic" or "openai"
        """
        model_lower = model_name.lower()

        # Check for explicit OpenAI models
        if model_lower.startswith(("gpt-", "o1-", "o3-")):
            return "openai"

        # Check for explicit Anthropic models
        if model_lower.startswith("claude"):
            return "anthropic"

        # Fall back to default provider from config
        return settings.llm.default_provider.lower()

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
    ) -> str:
        """
        Call the LLM with system and user prompts.

        Args:
            system_prompt: The agent's role and instructions
            user_prompt: The specific request with context data

        Returns:
            The LLM's response text
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

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
