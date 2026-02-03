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

from src.agents.base import BaseAgent
from src.config import settings
from src.utils.llm import parse_json_list
from src.workflows.state import (
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
    Validation,
)


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
    """

    name = "Validator"
    model_name = settings.llm.validator_model

    async def run(self, state: TradingState) -> TradingState:
        """
        Validate signals that passed risk assessment.

        Args:
            state: Trading state with signals and risk assessments

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
            user_prompt = self._build_user_prompt(
                approved_signals, assessment_map, state
            )
            response = await self._call_llm(VALIDATOR_SYSTEM_PROMPT, user_prompt)
            validations = self._parse_validations(response, approved_signals)
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

    def _build_user_prompt(
        self,
        signals: list[Signal],
        assessments: dict[UUID, RiskAssessment],
        state: TradingState,
    ) -> str:
        """Build prompt with signals and trade history context."""
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

        # Add recent trade history context
        # In production, this would come from database
        lines.append("\n--- RECENT TRADE HISTORY ---")
        lines.append("(Trade history will be provided by memory system)")

        lines.append("\nProvide validations as JSON array.")

        return "\n".join(lines)

    def _parse_validations(
        self,
        response: str,
        signals: list[Signal],
    ) -> list[Validation]:
        """Parse LLM response into Validation objects."""
        # Use shared JSON parsing utility
        parsed_list = parse_json_list(response, context="Validator validations")

        signal_map = {str(s.id): s for s in signals}

        validations = []
        for item in parsed_list:
            signal_id_str = item.get("signal_id", "")
            signal = signal_map.get(signal_id_str)
            if not signal:
                continue

            validation = Validation(
                signal_id=signal.id,
                approved=item.get("approved", False),
                concerns=item.get("concerns", []),
                suggestions=item.get("suggestions", []),
                reasoning=item.get("reasoning", ""),
                repetition_detected=item.get("repetition_detected", False),
                sector_clustering_detected=item.get("sector_clustering_detected", False),
                similar_setup_failures=int(item.get("similar_setup_failures", 0)),
            )
            validations.append(validation)

        return validations
