"""
Data Agent - Analyzes market data and generates trading signals.

The Data Agent is the first agent in the pipeline. It:
1. Receives market data (prices, technical indicators) for symbols
2. Analyzes the data using LLM reasoning
3. Produces Signal objects with BUY/SELL/HOLD recommendations

The agent focuses purely on technical analysis and does not consider
portfolio constraints (that's the Risk Manager's job).
"""

from typing import Any

from src.agents.base import BaseAgent
from src.config import settings
from src.services.market_data import MarketDataService, Quote, TechnicalIndicators
from src.utils.llm import parse_json_list
from src.workflows.state import Signal, SignalAction, TradingState


# System prompt defining the Data Agent's role and output format
DATA_AGENT_SYSTEM_PROMPT = """You are a trading signal generator. Your job is to analyze market data and generate trading signals.

ROLE:
- Analyze technical indicators (price, SMAs, RSI, volume)
- Identify potential trading opportunities
- Output clear BUY, SELL, or HOLD signals with confidence scores

CONSTRAINTS:
- Only analyze the data provided - do not make up information
- Confidence should reflect signal strength (0.0 = no confidence, 1.0 = very confident)
- HOLD is the default when signals are unclear or conflicting
- Do not consider portfolio constraints (the Risk Manager handles that)

TECHNICAL ANALYSIS GUIDELINES:
- RSI > 70: Overbought (potential SELL signal)
- RSI < 30: Oversold (potential BUY signal)
- Price > SMA_200: Long-term uptrend
- Price < SMA_200: Long-term downtrend
- Price crossing above SMA_20: Short-term bullish
- Price crossing below SMA_20: Short-term bearish
- Volume ratio > 1.5: High interest, confirms moves

OUTPUT FORMAT:
Respond with a JSON array of signal objects. Each object must have:
{
    "symbol": "AAPL",
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0 to 1.0,
    "proposed_quantity": integer (suggest based on signal strength, will be adjusted by risk),
    "reasoning": "Brief explanation of why this signal"
}

Example response:
```json
[
    {"symbol": "AAPL", "action": "BUY", "confidence": 0.75, "proposed_quantity": 5, "reasoning": "RSI oversold at 28, price above SMA_200 indicates uptrend"}
]
```
"""


class DataAgent(BaseAgent):
    """
    Analyzes market data and generates trading signals.

    This agent:
    1. Fetches current market data for requested symbols
    2. Formats data as context for the LLM
    3. Asks LLM to analyze and generate signals
    4. Parses LLM response into Signal objects
    5. Appends signals to state.signals
    """

    name = "DataAgent"
    model_name = settings.llm.data_agent_model

    def __init__(
        self,
        market_data_service: MarketDataService | None = None,
        model_name: str | None = None,
        provider: str | None = None,
    ):
        """
        Initialize the Data Agent.

        Args:
            market_data_service: Service to fetch market data. If None, creates default.
            model_name: Override the default model.
            provider: Override the provider ("openai", "anthropic", "google", "deepseek", or "auto")
        """
        super().__init__(model_name=model_name, provider=provider)
        self.market_data = market_data_service or MarketDataService()

    async def run(self, state: TradingState) -> TradingState:
        """
        Analyze market data for symbols and generate signals.

        Args:
            state: Trading state containing symbols to analyze

        Returns:
            State with signals appended to state.signals
        """
        if not state.symbols:
            # No symbols to analyze
            return state

        # Fetch market data for all symbols
        market_context = await self._gather_market_data(state.symbols)

        # Build the user prompt with market data
        user_prompt = self._build_user_prompt(state, market_context)

        try:
            # Call LLM to analyze data
            response = await self._call_llm(DATA_AGENT_SYSTEM_PROMPT, user_prompt)

            # Parse response into Signal objects
            signals = self._parse_signals(response, market_context)

            # Add signals to state
            state.signals.extend(signals)

        except Exception as e:
            state.add_error(self.name, str(e), {"symbols": state.symbols})

        return state

    async def _gather_market_data(
        self, symbols: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch quotes and technical indicators for all symbols.

        Args:
            symbols: List of stock symbols to fetch data for

        Returns:
            Dict mapping symbol to its market data (quote + indicators)
        """
        market_context = {}

        for symbol in symbols:
            try:
                # Fetch current quote and technical indicators in parallel
                quote = await self.market_data.get_quote(symbol)
                indicators = await self.market_data.get_technical_indicators(symbol)

                market_context[symbol] = {
                    "quote": quote,
                    "indicators": indicators,
                }
            except Exception as e:
                # Log error but continue with other symbols
                market_context[symbol] = {"error": str(e)}

        return market_context

    def _build_user_prompt(
        self,
        state: TradingState,
        market_context: dict[str, dict[str, Any]],
    ) -> str:
        """
        Build the user prompt with market data context.

        Args:
            state: Current trading state
            market_context: Market data for each symbol

        Returns:
            Formatted prompt string
        """
        lines = [
            f"Analyze the following symbols and generate trading signals.",
            f"Cycle type: {state.cycle_type}",
        ]

        if state.trigger_symbol:
            lines.append(f"Triggered by price move in: {state.trigger_symbol}")

        lines.append("\n--- MARKET DATA ---\n")

        for symbol, data in market_context.items():
            lines.append(f"## {symbol}")

            if "error" in data:
                lines.append(f"  Error fetching data: {data['error']}")
                continue

            quote: Quote = data.get("quote")
            indicators: TechnicalIndicators = data.get("indicators")

            if quote:
                lines.append(f"  Price: ${quote.price:.2f}")
                if quote.bid is not None and quote.ask is not None:
                    lines.append(f"  Bid: ${quote.bid:.2f} / Ask: ${quote.ask:.2f}")
                if quote.volume is not None:
                    lines.append(f"  Volume: {quote.volume:,}")

            if indicators:
                lines.append(f"  SMA_20: {indicators.sma_20 or 'N/A'}")
                lines.append(f"  SMA_50: {indicators.sma_50 or 'N/A'}")
                lines.append(f"  SMA_200: {indicators.sma_200 or 'N/A'}")
                lines.append(f"  RSI_14: {indicators.rsi_14 or 'N/A'}")
                lines.append(f"  Volume Ratio: {indicators.volume_ratio or 'N/A'}")

            lines.append("")

        lines.append("Generate signals for each symbol. Respond with JSON array only.")

        return "\n".join(lines)

    def _parse_signals(
        self,
        response: str,
        market_context: dict[str, dict[str, Any]],
    ) -> list[Signal]:
        """
        Parse LLM response into Signal objects.

        Args:
            response: Raw LLM response text
            market_context: Market data to enrich signals with

        Returns:
            List of parsed Signal objects
        """
        # Use shared JSON parsing utility
        parsed_list = parse_json_list(response, context="DataAgent signals")

        signals = []
        for item in parsed_list:
            symbol = item.get("symbol", "")

            # Get market data to enrich signal
            data = market_context.get(symbol, {})
            quote = data.get("quote")
            indicators = data.get("indicators")

            signal = Signal(
                symbol=symbol,
                action=SignalAction(item.get("action", "HOLD")),
                confidence=float(item.get("confidence", 0.0)),
                proposed_quantity=int(item.get("proposed_quantity", 0)),
                reasoning=item.get("reasoning", ""),
                # Enrich with actual market data
                price=quote.price if quote else 0.0,
                sma_20=indicators.sma_20 if indicators else None,
                sma_50=indicators.sma_50 if indicators else None,
                sma_200=indicators.sma_200 if indicators else None,
                rsi_14=indicators.rsi_14 if indicators else None,
                volume_ratio=indicators.volume_ratio if indicators else None,
            )
            signals.append(signal)

        return signals
