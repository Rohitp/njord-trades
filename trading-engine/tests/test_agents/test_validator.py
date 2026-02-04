"""
Tests for Validator agent.

Tests pattern detection, including vector similarity search for similar failed setups.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.agents.validator import Validator
from src.database.models import Trade, TradeEmbedding
from src.workflows.state import (
    PortfolioSnapshot,
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
    Validation,
)


class TestValidatorBasic:
    """Tests for basic Validator functionality."""

    @pytest.mark.asyncio
    async def test_validator_name(self):
        """Test that Validator has correct name."""
        validator = Validator()
        assert validator.name == "Validator"

    @pytest.mark.asyncio
    async def test_validator_no_approved_signals(self):
        """Test that Validator returns state unchanged if no approved signals."""
        validator = Validator()
        state = TradingState(
            symbols=["AAPL"],
            portfolio_snapshot=PortfolioSnapshot(cash=1000.0, total_value=5000.0),
        )
        # No approved signals (no risk assessments)
        result = await validator.run(state)
        assert len(result.validations) == 0

    @pytest.mark.asyncio
    async def test_validator_handles_llm_error(self):
        """Test that Validator handles LLM errors gracefully."""
        validator = Validator()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=150.0,
            proposed_quantity=5,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
        )
        state = TradingState(
            symbols=["AAPL"],
            portfolio_snapshot=PortfolioSnapshot(cash=1000.0, total_value=5000.0),
        )
        state.signals.append(signal)
        state.risk_assessments.append(assessment)

        with patch.object(validator, "_call_llm", side_effect=Exception("LLM error")):
            result = await validator.run(state)
            # Should fail-open: approve with warning
            assert len(result.validations) == 1
            assert result.validations[0].approved is True
            assert "Validation failed" in result.validations[0].concerns[0]


class TestValidatorVectorIntegration:
    """Tests for Validator vector similarity integration."""

    @pytest.mark.asyncio
    async def test_validator_queries_similar_failures(self):
        """Test that Validator queries similar failed setups when db_session provided."""
        validator = Validator()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=150.0,
            proposed_quantity=5,
            rsi_14=65.0,
            sma_20=150.0,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
            risk_score=0.3,
        )
        state = TradingState(
            symbols=["AAPL"],
            portfolio_snapshot=PortfolioSnapshot(cash=1000.0, total_value=5000.0),
        )
        state.signals.append(signal)
        state.risk_assessments.append(assessment)

        # Mock similar failed trades
        mock_trade_embedding1 = MagicMock(spec=TradeEmbedding)
        mock_trade_embedding1.trade_id = uuid4()
        mock_trade_embedding1.context_text = "Symbol: AAPL | Action: BUY | RSI: 65.0 | SMA_20: $150.0"

        mock_trade_embedding2 = MagicMock(spec=TradeEmbedding)
        mock_trade_embedding2.trade_id = uuid4()
        mock_trade_embedding2.context_text = "Symbol: AAPL | Action: BUY | RSI: 68.0 | SMA_20: $152.0"

        mock_trade1 = MagicMock(spec=Trade)
        mock_trade1.id = mock_trade_embedding1.trade_id
        mock_trade1.outcome = "LOSS"

        mock_trade2 = MagicMock(spec=Trade)
        mock_trade2.id = mock_trade_embedding2.trade_id
        mock_trade2.outcome = "LOSS"

        mock_session = MagicMock()
        mock_embedding_result = MagicMock()
        mock_embedding_result.scalars.return_value.all.return_value = [
            mock_trade_embedding1,
            mock_trade_embedding2,
        ]
        mock_trade_result1 = MagicMock()
        mock_trade_result1.scalar_one_or_none.return_value = mock_trade1
        mock_trade_result2 = MagicMock()
        mock_trade_result2.scalar_one_or_none.return_value = mock_trade2

        # Mock session.execute to return different results for embedding query and trade queries
        async def mock_execute(stmt):
            # Check if this is a Trade query (has .where(Trade.id == ...))
            if "Trade" in str(stmt):
                if mock_trade1.id in str(stmt):
                    return mock_trade_result1
                elif mock_trade2.id in str(stmt):
                    return mock_trade_result2
            # Otherwise it's the embedding similarity query
            return mock_embedding_result

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        with patch("src.agents.validator.TradeEmbeddingService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.find_similar_trades = AsyncMock(
                return_value=[mock_trade_embedding1, mock_trade_embedding2]
            )
            mock_service_class.return_value = mock_service

            with patch.object(validator, "_call_llm") as mock_llm:
                mock_response = MagicMock()
                mock_response.content = f'[{{"signal_id": "{signal.id}", "approved": false, "concerns": ["Similar setup failed"], "similar_setup_failures": 2}}]'
                mock_llm.return_value = mock_response

                result = await validator.run(state, db_session=mock_session)

                # Verify similarity search was called
                assert mock_service.find_similar_trades.called
                call_args = mock_service.find_similar_trades.call_args
                assert call_args[1]["session"] == mock_session
                assert call_args[1]["limit"] == 5
                assert call_args[1]["min_similarity"] == 0.7

                # Verify validation includes failure count
                assert len(result.validations) == 1
                assert result.validations[0].similar_setup_failures == 2

    @pytest.mark.asyncio
    async def test_validator_filters_to_failed_trades_only(self):
        """Test that Validator only counts LOSS trades, not WIN trades."""
        validator = Validator()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=150.0,
            proposed_quantity=5,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
        )
        state = TradingState(
            symbols=["AAPL"],
            portfolio_snapshot=PortfolioSnapshot(cash=1000.0, total_value=5000.0),
        )
        state.signals.append(signal)
        state.risk_assessments.append(assessment)

        # Mock similar trades: one LOSS, one WIN
        mock_trade_embedding1 = MagicMock(spec=TradeEmbedding)
        mock_trade_embedding1.trade_id = uuid4()
        mock_trade1 = MagicMock(spec=Trade)
        mock_trade1.id = mock_trade_embedding1.trade_id
        mock_trade1.outcome = "LOSS"

        mock_trade_embedding2 = MagicMock(spec=TradeEmbedding)
        mock_trade_embedding2.trade_id = uuid4()
        mock_trade2 = MagicMock(spec=Trade)
        mock_trade2.id = mock_trade_embedding2.trade_id
        mock_trade2.outcome = "WIN"

        mock_session = MagicMock()
        mock_embedding_result = MagicMock()
        mock_embedding_result.scalars.return_value.all.return_value = [
            mock_trade_embedding1,
            mock_trade_embedding2,
        ]

        async def mock_execute(stmt):
            if "Trade" in str(stmt):
                if mock_trade1.id in str(stmt):
                    result = MagicMock()
                    result.scalar_one_or_none.return_value = mock_trade1
                    return result
                elif mock_trade2.id in str(stmt):
                    result = MagicMock()
                    result.scalar_one_or_none.return_value = mock_trade2
                    return result
            return mock_embedding_result

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        with patch("src.agents.validator.TradeEmbeddingService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.find_similar_trades = AsyncMock(
                return_value=[mock_trade_embedding1, mock_trade_embedding2]
            )
            mock_service_class.return_value = mock_service

            with patch.object(validator, "_call_llm") as mock_llm:
                mock_response = MagicMock()
                mock_response.content = f'[{{"signal_id": "{signal.id}", "approved": true, "similar_setup_failures": 1}}]'
                mock_llm.return_value = mock_response

                result = await validator.run(state, db_session=mock_session)

                # Should only count the LOSS trade (1 failure, not 2)
                assert len(result.validations) == 1
                assert result.validations[0].similar_setup_failures == 1

    @pytest.mark.asyncio
    async def test_validator_graceful_degradation_no_session(self):
        """Test that Validator works without db_session (backward compatible)."""
        validator = Validator()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=150.0,
            proposed_quantity=5,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
        )
        state = TradingState(
            symbols=["AAPL"],
            portfolio_snapshot=PortfolioSnapshot(cash=1000.0, total_value=5000.0),
        )
        state.signals.append(signal)
        state.risk_assessments.append(assessment)

        with patch.object(validator, "_call_llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = f'[{{"signal_id": "{signal.id}", "approved": true, "similar_setup_failures": 0}}]'
            mock_llm.return_value = mock_response

            result = await validator.run(state)  # No db_session

            # Should work normally without similarity search
            assert len(result.validations) == 1
            assert result.validations[0].similar_setup_failures == 0

    @pytest.mark.asyncio
    async def test_validator_graceful_degradation_similarity_error(self):
        """Test that Validator continues if similarity search fails."""
        validator = Validator()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=150.0,
            proposed_quantity=5,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
        )
        state = TradingState(
            symbols=["AAPL"],
            portfolio_snapshot=PortfolioSnapshot(cash=1000.0, total_value=5000.0),
        )
        state.signals.append(signal)
        state.risk_assessments.append(assessment)

        mock_session = MagicMock()

        with patch("src.agents.validator.TradeEmbeddingService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.find_similar_trades = AsyncMock(side_effect=Exception("DB error"))
            mock_service_class.return_value = mock_service

            with patch.object(validator, "_call_llm") as mock_llm:
                mock_response = MagicMock()
                mock_response.content = f'[{{"signal_id": "{signal.id}", "approved": true, "similar_setup_failures": 0}}]'
                mock_llm.return_value = mock_response

                result = await validator.run(state, db_session=mock_session)

                # Should continue working despite similarity search failure
                assert len(result.validations) == 1
                assert result.validations[0].approved is True

    @pytest.mark.asyncio
    async def test_validator_builds_signal_context_text(self):
        """Test that signal context text matches trade embedding format."""
        validator = Validator()
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            price=150.0,
            proposed_quantity=5,
            rsi_14=65.0,
            sma_20=150.0,
            sma_50=145.0,
            volume_ratio=1.5,
            reasoning="Strong momentum",
            confidence=0.8,
        )
        assessment = RiskAssessment(
            signal_id=signal.id,
            approved=True,
            adjusted_quantity=5,
            risk_score=0.3,
        )

        context_text = validator._build_signal_context_text(signal, assessment)

        # Should include all relevant fields
        assert "Symbol: AAPL" in context_text
        assert "Action: BUY" in context_text
        assert "Quantity: 5" in context_text
        assert "Signal reasoning: Strong momentum" in context_text
        assert "Signal confidence: 0.8" in context_text
        assert "RSI: 65.0" in context_text
        assert "SMA_20: $150.0" in context_text
        assert "SMA_50: $145.0" in context_text
        assert "Volume ratio: 1.5x" in context_text
        assert "Risk score: 0.3" in context_text

