"""
Validator Agent - Pattern recognition and quality control.

The Validator is the third agent in the pipeline. It provides a "second opinion"
by looking for problematic patterns:

1. Repetition: Are we making the same trade repeatedly?
2. Sector clustering: Too many trades in same sector this week?
3. Historical failures: Did similar setups fail before?
4. Overtrading: Are we trading too frequently?

This agent can access trade history to identify patterns the other agents miss.
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.base import BaseAgent
from src.config import settings
from src.services.embeddings.trade_embedding import TradeEmbeddingService
from src.utils.llm import parse_json_list
from src.utils.logging import get_logger
from src.workflows.state import (
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
    Validation,
)

log = get_logger(__name__)


VALIDATOR_SYSTEM_PROMPT = """You are a trading validator providing quality control and pattern detection.

ROLE:
- Review signals that passed risk assessment
- Detect problematic patterns from trade history
- Provide a second opinion on trade quality
- Suggest improvements or flag concerns

PATTERNS TO DETECT:

1. REPETITION
   - Same symbol traded multiple times recently
   - Buying back something we just sold
   - Flip-flopping between positions

2. SECTOR CLUSTERING
   - Multiple trades in same sector within short period
   - Concentrated bets on related industries
   - Missing diversification opportunities

3. SETUP FAILURES
   - Similar technical setups that failed before
   - Patterns that historically underperformed
   - Ignoring lessons from past trades

4. OVERTRADING
   - Too many trades relative to portfolio size
   - Trading for activity rather than opportunity
   - Churn that generates costs without returns

OUTPUT FORMAT:
Respond with a JSON array of validation objects:
{
    "signal_id": "uuid-string",
    "approved": true/false,
    "concerns": ["list", "of", "concerns"],
    "suggestions": ["list", "of", "suggestions"],
    "reasoning": "Explanation",
    "repetition_detected": true/false,
    "sector_clustering_detected": true/false,
    "similar_setup_failures": integer (count of similar failures)
}

APPROVAL GUIDELINES:
- Approve if no major patterns detected
- Reject if clear repetition or clustering
- When uncertain, approve with concerns noted
- Similar setup failures > 2 should trigger rejection

Example:
```json
[
    {"signal_id": "abc-123", "approved": true, "concerns": ["Third tech trade this week"], "suggestions": ["Consider diversifying into other sectors"], "reasoning": "Pattern detected but not severe enough to reject", "repetition_detected": false, "sector_clustering_detected": true, "similar_setup_failures": 1}
]
```
"""


class Validator(BaseAgent):
    """
    Provides quality control and pattern detection for trading signals.

    Reviews signals that passed risk assessment, looking for patterns
    that might indicate poor trading decisions (repetition, clustering,
    historical failures).

    Uses vector similarity search to find similar failed trade setups.
    """

    name = "Validator"
    model_name = settings.llm.validator_model

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize Validator.

        Args:
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(**kwargs)
        # db_session will be passed to run() method when available

    async def run(
        self,
        state: TradingState,
        db_session: AsyncSession | None = None,
    ) -> TradingState:
        """
        Validate signals that passed risk assessment.

        Args:
            state: Trading state with signals and risk assessments
            db_session: Optional database session for vector similarity search

        Returns:
            State with validations appended
        """
        # Get signals that were approved by risk manager
        approved_signals = state.get_approved_signals()

        if not approved_signals:
            return state

        # Get corresponding risk assessments for context
        assessment_map = {
            ra.signal_id: ra for ra in state.risk_assessments if ra.approved
        }

        try:
            # Query similar failed setups for each signal (if db_session available)
            similar_failures_map = {}
            if db_session:
                trade_embedding_service = TradeEmbeddingService()
                similar_failures_map = await self._find_similar_failures(
                    approved_signals, assessment_map, db_session, trade_embedding_service
                )

            user_prompt = self._build_user_prompt(
                approved_signals, assessment_map, state, similar_failures_map
            )
            response = await self._call_llm(VALIDATOR_SYSTEM_PROMPT, user_prompt, state)
            validations = self._parse_validations(
                response, approved_signals, similar_failures_map
            )
            state.validations.extend(validations)

        except Exception as e:
            state.add_error(self.name, str(e))
            # On error, approve all with warning
            for signal in approved_signals:
                state.validations.append(
                    Validation(
                        signal_id=signal.id,
                        approved=True,  # Fail-open: don't block on validation error
                        concerns=["Validation failed - proceeding with caution"],
                        reasoning=str(e),
                    )
                )

        return state

    async def _find_similar_failures(
        self,
        signals: list[Signal],
        assessments: dict[UUID, RiskAssessment],
        db_session: AsyncSession,
        trade_embedding_service: TradeEmbeddingService,
    ) -> dict[UUID, list]:
        """
        Find similar failed trade setups for each signal.

        Args:
            signals: List of signals to check
            assessments: Risk assessments for context
            db_session: Database session for queries
            trade_embedding_service: TradeEmbeddingService instance

        Returns:
            Dictionary mapping signal_id -> list of similar failed TradeEmbedding records
        """
        similar_failures_map = {}

        for signal in signals:
            try:
                # Build context text similar to trade embedding format
                context_text = self._build_signal_context_text(signal, assessments.get(signal.id))

                # Find similar trades
                similar_trades = await trade_embedding_service.find_similar_trades(
                    context_text=context_text,
                    limit=5,
                    min_similarity=0.7,
                    session=db_session,
                )

                # Filter to only failed trades (LOSS outcome)
                from src.database.models import Trade
                failed_trades = []
                for trade_embedding in similar_trades:
                    # Fetch the trade to check outcome
                    from sqlalchemy import select
                    trade_result = await db_session.execute(
                        select(Trade).where(Trade.id == trade_embedding.trade_id)
                    )
                    trade = trade_result.scalar_one_or_none()
                    if trade and trade.outcome == "LOSS":
                        failed_trades.append(trade_embedding)

                if failed_trades:
                    similar_failures_map[signal.id] = failed_trades
                    log.debug(
                        "validator_similar_failures_found",
                        signal_id=str(signal.id),
                        symbol=signal.symbol,
                        count=len(failed_trades),
                    )

            except Exception as e:
                log.warning(
                    "validator_similarity_search_failed",
                    signal_id=str(signal.id),
                    error=str(e),
                )
                # Continue without similar failures for this signal (graceful degradation)

        return similar_failures_map

    def _build_signal_context_text(
        self,
        signal: Signal,
        assessment: RiskAssessment | None = None,
    ) -> str:
        """
        Build context text from signal for similarity search.

        Matches the format used in TradeEmbeddingService._format_trade_context()
        so similarity search works correctly.

        Args:
            signal: Signal to build context for
            assessment: Optional risk assessment for additional context

        Returns:
            Formatted context text
        """
        parts = []

        # Basic signal info
        parts.append(f"Symbol: {signal.symbol}")
        parts.append(f"Action: {signal.action.value}")
        parts.append(f"Quantity: {signal.proposed_quantity}")

        # Signal reasoning
        if signal.reasoning:
            parts.append(f"Signal reasoning: {signal.reasoning}")
        if signal.confidence:
            parts.append(f"Signal confidence: {signal.confidence:.2f}")

        # Technical indicators from signal
        indicators = []
        if signal.rsi_14 is not None:
            indicators.append(f"RSI: {signal.rsi_14:.1f}")
        if signal.sma_20 is not None:
            indicators.append(f"SMA_20: ${signal.sma_20:.2f}")
        if signal.sma_50 is not None:
            indicators.append(f"SMA_50: ${signal.sma_50:.2f}")
        if signal.sma_200 is not None:
            indicators.append(f"SMA_200: ${signal.sma_200:.2f}")
        if signal.volume_ratio is not None:
            indicators.append(f"Volume ratio: {signal.volume_ratio:.2f}x")
        if indicators:
            parts.append(f"Technical indicators: {', '.join(indicators)}")

        # Risk score from assessment
        if assessment and assessment.risk_score:
            parts.append(f"Risk score: {assessment.risk_score:.2f}")

        return " | ".join(parts)

    def _build_user_prompt(
        self,
        signals: list[Signal],
        assessments: dict[UUID, RiskAssessment],
        state: TradingState,
        similar_failures_map: dict[UUID, list] | None = None,
    ) -> str:
        """Build prompt with signals, trade history context, and similar failures."""
        lines = [
            "Validate the following signals that passed risk assessment.",
            "",
            "--- PORTFOLIO CONTEXT ---",
            self._format_portfolio_context(state),
            "",
            "--- SIGNALS TO VALIDATE ---",
        ]

        for signal in signals:
            assessment = assessments.get(signal.id)

            lines.append(f"\nSignal ID: {signal.id}")
            lines.append(f"  Symbol: {signal.symbol}")
            lines.append(f"  Action: {signal.action.value}")
            lines.append(f"  Confidence: {signal.confidence:.2f}")
            lines.append(f"  Proposed Quantity: {signal.proposed_quantity}")

            if assessment:
                lines.append(f"  Risk-Adjusted Quantity: {assessment.adjusted_quantity}")
                lines.append(f"  Risk Score: {assessment.risk_score:.2f}")
                if assessment.concerns:
                    lines.append(f"  Risk Concerns: {', '.join(assessment.concerns)}")

            lines.append(f"  Signal Reasoning: {signal.reasoning}")

            # Add similar failed setups if found
            if similar_failures_map and signal.id in similar_failures_map:
                failures = similar_failures_map[signal.id]
                lines.append(f"\n  ⚠️  SIMILAR FAILED SETUPS FOUND: {len(failures)}")
                lines.append("  These are past trades with similar setups that resulted in LOSS:")
                for i, failure in enumerate(failures[:3], 1):  # Show top 3
                    lines.append(f"    {i}. {failure.context_text[:150]}...")
                lines.append(f"  WARNING: {len(failures)} similar setup(s) failed in the past.")

        # Add recent trade history context
        lines.append("\n--- RECENT TRADE HISTORY ---")
        lines.append("(Trade history will be provided by memory system)")

        lines.append("\nProvide validations as JSON array.")
        lines.append("IMPORTANT: Set 'similar_setup_failures' to the count of similar failures found above.")

        return "\n".join(lines)

    def _parse_validations(
        self,
        response: str,
        signals: list[Signal],
        similar_failures_map: dict[UUID, list] | None = None,
    ) -> list[Validation]:
        """
        Parse LLM response into Validation objects.

        Args:
            response: LLM response text
            signals: List of signals being validated
            similar_failures_map: Map of signal_id -> list of similar failed trades

        Returns:
            List of Validation objects
        """
        # Use shared JSON parsing utility
        parsed_list = parse_json_list(response, context="Validator validations")

        signal_map = {str(s.id): s for s in signals}

        validations = []
        for item in parsed_list:
            signal_id_str = item.get("signal_id", "")
            signal = signal_map.get(signal_id_str)
            if not signal:
                continue

            # Get similar failures count (from LLM or fallback to actual count)
            llm_failure_count = int(item.get("similar_setup_failures", 0))
            # Use actual count if available (more reliable than LLM parsing)
            if similar_failures_map and signal.id in similar_failures_map:
                actual_failure_count = len(similar_failures_map[signal.id])
                # Use the maximum to be safe (LLM might miss some)
                failure_count = max(llm_failure_count, actual_failure_count)
            else:
                failure_count = llm_failure_count

            validation = Validation(
                signal_id=signal.id,
                approved=item.get("approved", False),
                concerns=item.get("concerns", []),
                suggestions=item.get("suggestions", []),
                reasoning=item.get("reasoning", ""),
                repetition_detected=item.get("repetition_detected", False),
                sector_clustering_detected=item.get("sector_clustering_detected", False),
                similar_setup_failures=failure_count,
            )
            validations.append(validation)

        return validations
