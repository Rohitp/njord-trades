"""
Tests for Data Agent.

Tests the market data analysis and signal generation logic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.data_agent import DataAgent
from src.services.market_data import Quote, TechnicalIndicators
from src.workflows.state import SignalAction, TradingState


class TestDataAgentBasic:
    """Tests for basic DataAgent functionality."""

    def test_data_agent_name(self):
        """Test that DataAgent has correct name."""
        agent = DataAgent()
        assert agent.name == "DataAgent"

    @pytest.mark.asyncio
    async def test_empty_symbols_returns_unchanged_state(self):
        """Test that empty symbols list returns state unchanged."""
        agent = DataAgent()
        state = TradingState(symbols=[])

        result = await agent.run(state)

        assert result.signals == []
        assert result.errors == []


class TestDataAgentMarketData:
    """Tests for market data gathering."""

    @pytest.mark.asyncio
    async def test_gather_market_data_success(self):
        """Test successful market data gathering."""
        mock_market_service = MagicMock()
        mock_quote = Quote(symbol="AAPL", price=150.0, bid=149.9, ask=150.1, volume=1000000)
        mock_indicators = TechnicalIndicators(
            symbol="AAPL",
            price=150.0,
            sma_20=148.0,
            sma_50=145.0,
            sma_200=140.0,
            rsi_14=55.0,
            volume_avg_20=800000,
            volume_ratio=1.25,
        )
        mock_market_service.get_quote = AsyncMock(return_value=mock_quote)
        mock_market_service.get_technical_indicators = AsyncMock(return_value=mock_indicators)

        agent = DataAgent(market_data_service=mock_market_service)
        result = await agent._gather_market_data(["AAPL"])

        assert "AAPL" in result
        assert result["AAPL"]["quote"] == mock_quote
        assert result["AAPL"]["indicators"] == mock_indicators

    @pytest.mark.asyncio
    async def test_gather_market_data_error_handling(self):
        """Test that errors for one symbol don't stop others."""
        mock_market_service = MagicMock()
        mock_quote = Quote(symbol="MSFT", price=300.0, volume=500000)

        async def get_quote(symbol):
            if symbol == "AAPL":
                raise Exception("API error")
            return mock_quote

        mock_market_service.get_quote = AsyncMock(side_effect=get_quote)
        mock_market_service.get_technical_indicators = AsyncMock(return_value=None)

        agent = DataAgent(market_data_service=mock_market_service)
        result = await agent._gather_market_data(["AAPL", "MSFT"])

        assert "AAPL" in result
        assert "error" in result["AAPL"]
        assert "API error" in result["AAPL"]["error"]
        assert "MSFT" in result
        assert result["MSFT"]["quote"] == mock_quote


class TestDataAgentPromptBuilding:
    """Tests for prompt building."""

    def test_build_user_prompt_basic(self):
        """Test basic prompt building."""
        agent = DataAgent()
        state = TradingState(symbols=["AAPL"], cycle_type="scheduled")
        market_context = {
            "AAPL": {
                "quote": Quote(symbol="AAPL", price=150.0, bid=149.9, ask=150.1, volume=1000000),
                "indicators": TechnicalIndicators(
                    symbol="AAPL",
                    price=150.0,
                    sma_20=148.0,
                    sma_50=145.0,
                    sma_200=140.0,
                    rsi_14=55.0,
                    volume_avg_20=800000,
                    volume_ratio=1.25,
                ),
            }
        }

        prompt = agent._build_user_prompt(state, market_context)

        assert "AAPL" in prompt
        assert "scheduled" in prompt
        assert "$150.00" in prompt
        assert "SMA_20" in prompt
        assert "RSI_14" in prompt

    def test_build_user_prompt_with_trigger(self):
        """Test prompt building with trigger symbol."""
        agent = DataAgent()
        state = TradingState(
            symbols=["TSLA"],
            cycle_type="event",
            trigger_symbol="TSLA",
        )
        market_context = {
            "TSLA": {
                "quote": Quote(symbol="TSLA", price=250.0, volume=2000000),
                "indicators": None,
            }
        }

        prompt = agent._build_user_prompt(state, market_context)

        assert "event" in prompt
        assert "Triggered by price move in: TSLA" in prompt

    def test_build_user_prompt_with_error(self):
        """Test prompt building when data fetch failed."""
        agent = DataAgent()
        state = TradingState(symbols=["AAPL"])
        market_context = {
            "AAPL": {"error": "API timeout"}
        }

        prompt = agent._build_user_prompt(state, market_context)

        assert "AAPL" in prompt
        assert "Error fetching data: API timeout" in prompt


class TestDataAgentSignalParsing:
    """Tests for signal parsing."""

    def test_parse_signals_basic(self):
        """Test basic signal parsing."""
        agent = DataAgent()
        response = '''```json
[
    {"symbol": "AAPL", "action": "BUY", "confidence": 0.75, "proposed_quantity": 5, "reasoning": "RSI oversold"}
]
```'''
        market_context = {
            "AAPL": {
                "quote": Quote(symbol="AAPL", price=150.0, volume=1000000),
                "indicators": TechnicalIndicators(
                    symbol="AAPL",
                    price=150.0,
                    sma_20=148.0,
                    sma_50=None,
                    sma_200=None,
                    rsi_14=28.0,
                    volume_avg_20=800000,
                    volume_ratio=1.25,
                ),
            }
        }

        signals = agent._parse_signals(response, market_context)

        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
        assert signals[0].action == SignalAction.BUY
        assert signals[0].confidence == 0.75
        assert signals[0].proposed_quantity == 5
        assert signals[0].price == 150.0
        assert signals[0].rsi_14 == 28.0
        assert signals[0].sma_20 == 148.0

    def test_parse_signals_multiple(self):
        """Test parsing multiple signals."""
        agent = DataAgent()
        response = '''[
    {"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "proposed_quantity": 5, "reasoning": "Bullish"},
    {"symbol": "MSFT", "action": "HOLD", "confidence": 0.5, "proposed_quantity": 0, "reasoning": "Neutral"}
]'''
        market_context = {
            "AAPL": {"quote": Quote(symbol="AAPL", price=150.0), "indicators": None},
            "MSFT": {"quote": Quote(symbol="MSFT", price=300.0), "indicators": None},
        }

        signals = agent._parse_signals(response, market_context)

        assert len(signals) == 2
        assert signals[0].symbol == "AAPL"
        assert signals[1].symbol == "MSFT"
        assert signals[1].action == SignalAction.HOLD

    def test_parse_signals_sell(self):
        """Test parsing SELL signal."""
        agent = DataAgent()
        response = '[{"symbol": "AAPL", "action": "SELL", "confidence": 0.9, "proposed_quantity": 10, "reasoning": "Overbought"}]'
        market_context = {
            "AAPL": {"quote": Quote(symbol="AAPL", price=180.0), "indicators": None}
        }

        signals = agent._parse_signals(response, market_context)

        assert len(signals) == 1
        assert signals[0].action == SignalAction.SELL
        assert signals[0].proposed_quantity == 10


class TestDataAgentFullFlow:
    """Tests for full run flow."""

    @pytest.mark.asyncio
    async def test_run_generates_signals(self):
        """Test full run generates signals correctly."""
        mock_market_service = MagicMock()
        mock_quote = Quote(symbol="AAPL", price=150.0, bid=149.9, ask=150.1, volume=1000000)
        mock_indicators = TechnicalIndicators(
            symbol="AAPL",
            price=150.0,
            sma_20=148.0,
            sma_50=145.0,
            sma_200=140.0,
            rsi_14=28.0,
            volume_avg_20=800000,
            volume_ratio=1.5,
        )
        mock_market_service.get_quote = AsyncMock(return_value=mock_quote)
        mock_market_service.get_technical_indicators = AsyncMock(return_value=mock_indicators)

        agent = DataAgent(market_data_service=mock_market_service)
        state = TradingState(symbols=["AAPL"])

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '[{"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "proposed_quantity": 5, "reasoning": "RSI oversold at 28"}]'

            result = await agent.run(state)

            assert len(result.signals) == 1
            assert result.signals[0].symbol == "AAPL"
            assert result.signals[0].action == SignalAction.BUY
            assert result.signals[0].price == 150.0
            assert result.signals[0].rsi_14 == 28.0

    @pytest.mark.asyncio
    async def test_run_handles_llm_error(self):
        """Test that LLM errors are handled gracefully."""
        mock_market_service = MagicMock()
        mock_market_service.get_quote = AsyncMock(return_value=Quote(symbol="AAPL", price=150.0))
        mock_market_service.get_technical_indicators = AsyncMock(return_value=None)

        agent = DataAgent(market_data_service=mock_market_service)
        state = TradingState(symbols=["AAPL"])

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM timeout")

            result = await agent.run(state)

            assert len(result.signals) == 0
            assert len(result.errors) == 1
            assert "LLM timeout" in result.errors[0]["error"]
            assert result.errors[0]["agent"] == "DataAgent"

    @pytest.mark.asyncio
    async def test_run_multiple_symbols(self):
        """Test running with multiple symbols."""
        mock_market_service = MagicMock()

        async def get_quote(symbol):
            prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 140.0}
            return Quote(symbol=symbol, price=prices.get(symbol, 100.0))

        mock_market_service.get_quote = AsyncMock(side_effect=get_quote)
        mock_market_service.get_technical_indicators = AsyncMock(return_value=None)

        agent = DataAgent(market_data_service=mock_market_service)
        state = TradingState(symbols=["AAPL", "MSFT", "GOOGL"])

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '''[
                {"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "proposed_quantity": 5, "reasoning": "Strong"},
                {"symbol": "MSFT", "action": "HOLD", "confidence": 0.5, "proposed_quantity": 0, "reasoning": "Neutral"},
                {"symbol": "GOOGL", "action": "SELL", "confidence": 0.7, "proposed_quantity": 3, "reasoning": "Overbought"}
            ]'''

            result = await agent.run(state)

            assert len(result.signals) == 3
            symbols = {s.symbol for s in result.signals}
            assert symbols == {"AAPL", "MSFT", "GOOGL"}
