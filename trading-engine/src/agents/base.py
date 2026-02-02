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

        Uses settings.llm.default_provider to choose between Anthropic and OpenAI.
        Falls back to Anthropic if provider is unknown.
        """
        provider = settings.llm.default_provider.lower()

        if provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.llm.openai_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )
        else:
            # Default to Anthropic
            return ChatAnthropic(
                model=self.model_name,
                api_key=settings.llm.anthropic_api_key,
                max_retries=settings.llm.max_retries,
                timeout=60.0,
                temperature=0.0,
            )

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
        snapshot = state.portfolio_snapshot

        lines = [
            f"Cash Available: ${snapshot.cash:,.2f}",
            f"Total Portfolio Value: ${snapshot.total_value:,.2f}",
            f"Deployed Capital: ${snapshot.deployed_capital:,.2f}",
        ]

        if snapshot.positions:
            lines.append("\nCurrent Positions:")
            for symbol, details in snapshot.positions.items():
                qty = details.get("quantity", 0)
                value = details.get("value", 0)
                sector = details.get("sector", "Unknown")
                lines.append(f"  {symbol}: {qty} shares, ${value:,.2f} ({sector})")

        if snapshot.sector_exposure:
            lines.append("\nSector Exposure:")
            for sector, exposure in snapshot.sector_exposure.items():
                pct = (exposure / snapshot.total_value * 100) if snapshot.total_value > 0 else 0
                lines.append(f"  {sector}: ${exposure:,.2f} ({pct:.1f}%)")

        return "\n".join(lines)
