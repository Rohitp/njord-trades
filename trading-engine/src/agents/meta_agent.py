"""
Meta Agent - Final decision synthesis.

The Meta Agent is the final agent in the pipeline. It:
1. Reviews all previous agent outputs (signals, assessments, validations)
2. Weighs different perspectives
3. Makes the final EXECUTE or DO_NOT_EXECUTE decision

This agent uses a more powerful model (Claude Opus) because it needs to:
- Synthesize complex, sometimes conflicting information
- Make nuanced judgment calls
- Provide clear reasoning for audit trails

The Meta Agent CANNOT override hard constraints (those are enforced in code).
It can only decide between signals that passed all previous checks.
"""

import json
from uuid import UUID

from src.agents.base import BaseAgent
from src.config import settings
from src.workflows.state import (
    Decision,
    FinalDecision,
    RiskAssessment,
    Signal,
    TradingState,
    Validation,
)


META_AGENT_SYSTEM_PROMPT = """You are the Meta Agent - the final decision maker in a trading system.

ROLE:
- Synthesize inputs from Data Agent, Risk Manager, and Validator
- Make final EXECUTE or DO_NOT_EXECUTE decisions
- Provide clear reasoning for audit purposes
- Determine final quantities

IMPORTANT CONSTRAINTS:
- You CANNOT override hard constraint rejections (already filtered out)
- You can only approve/reject signals that passed all previous checks
- Your decisions will result in real trades with real money

DECISION CRITERIA:

1. AGREEMENT
   - All agents agree: Strong signal to execute
   - Risk and Validator agree, Data was uncertain: Consider executing
   - Disagreement between agents: Lean toward caution

2. CONFIDENCE
   - High confidence (>0.7) from multiple agents: Likely execute
   - Low confidence or mixed signals: Likely reject
   - Single high confidence with low risk: Consider executing

3. RISK BALANCE
   - Low risk score + high confidence = Execute
   - High risk score + high confidence = Reduce quantity or reject
   - Low confidence + any risk = Reject

4. PATTERN CONCERNS
   - Repetition detected: Strong reason to reject
   - Sector clustering: Consider reducing quantity
   - Setup failures > 2: Reject

QUANTITY GUIDELINES:
- Start with risk-adjusted quantity
- Reduce further if concerns exist
- Never exceed risk-adjusted quantity
- Zero quantity = DO_NOT_EXECUTE

OUTPUT FORMAT:
Respond with a JSON array of final decisions:
{
    "signal_id": "uuid-string",
    "decision": "EXECUTE" | "DO_NOT_EXECUTE",
    "final_quantity": integer,
    "confidence": 0.0 to 1.0,
    "reasoning": "Clear explanation of decision"
}

Example:
```json
[
    {"signal_id": "abc-123", "decision": "EXECUTE", "final_quantity": 3, "confidence": 0.85, "reasoning": "Strong buy signal with RSI oversold, risk approved with minor concerns, no validation issues. Proceeding with reduced quantity."}
]
```
"""


class MetaAgent(BaseAgent):
    """
    Final decision maker that synthesizes all agent perspectives.

    Uses Claude Opus (the most capable model) for nuanced decision-making.
    Reviews all previous agent outputs and makes final EXECUTE/DO_NOT_EXECUTE
    decisions for each signal.
    """

    name = "MetaAgent"
    # Use the highest quality model for final decisions
    model_name = settings.llm.meta_agent_model

    async def run(self, state: TradingState) -> TradingState:
        """
        Make final decisions on all validated signals.

        Args:
            state: Trading state with all previous agent outputs

        Returns:
            State with final_decisions appended
        """
        # Get signals that passed both risk and validation
        validated_signals = state.get_validated_signals()

        if not validated_signals:
            return state

        # Build context maps
        assessment_map = {ra.signal_id: ra for ra in state.risk_assessments}
        validation_map = {v.signal_id: v for v in state.validations}

        try:
            user_prompt = self._build_user_prompt(
                validated_signals, assessment_map, validation_map, state
            )
            response = await self._call_llm(META_AGENT_SYSTEM_PROMPT, user_prompt)
            decisions = self._parse_decisions(response, validated_signals)
            state.final_decisions.extend(decisions)

        except Exception as e:
            state.add_error(self.name, str(e))
            # On error, reject all - fail-closed for final decisions
            for signal in validated_signals:
                state.final_decisions.append(
                    FinalDecision(
                        signal_id=signal.id,
                        decision=Decision.DO_NOT_EXECUTE,
                        final_quantity=0,
                        confidence=0.0,
                        reasoning=f"Decision failed due to error: {e}",
                    )
                )

        return state

    def _build_user_prompt(
        self,
        signals: list[Signal],
        assessments: dict[UUID, RiskAssessment],
        validations: dict[UUID, Validation],
        state: TradingState,
    ) -> str:
        """Build comprehensive prompt with all agent perspectives."""
        lines = [
            "Make final trading decisions for the following signals.",
            "All signals have passed risk assessment and validation.",
            "",
            "--- PORTFOLIO CONTEXT ---",
            self._format_portfolio_context(state),
            "",
            "--- SIGNALS FOR FINAL DECISION ---",
        ]

        for signal in signals:
            assessment = assessments.get(signal.id)
            validation = validations.get(signal.id)

            lines.append(f"\n{'='*50}")
            lines.append(f"Signal ID: {signal.id}")
            lines.append(f"Symbol: {signal.symbol}")
            lines.append(f"Action: {signal.action.value}")
            lines.append(f"Price: ${signal.price:.2f}")

            # Data Agent perspective
            lines.append("\n[DATA AGENT]")
            lines.append(f"  Confidence: {signal.confidence:.2f}")
            lines.append(f"  Proposed Quantity: {signal.proposed_quantity}")
            lines.append(f"  Reasoning: {signal.reasoning}")
            if signal.rsi_14:
                lines.append(f"  RSI: {signal.rsi_14:.1f}")

            # Risk Manager perspective
            if assessment:
                lines.append("\n[RISK MANAGER]")
                lines.append(f"  Approved: {assessment.approved}")
                lines.append(f"  Adjusted Quantity: {assessment.adjusted_quantity}")
                lines.append(f"  Risk Score: {assessment.risk_score:.2f}")
                if assessment.concerns:
                    lines.append(f"  Concerns: {', '.join(assessment.concerns)}")
                lines.append(f"  Reasoning: {assessment.reasoning}")

            # Validator perspective
            if validation:
                lines.append("\n[VALIDATOR]")
                lines.append(f"  Approved: {validation.approved}")
                if validation.concerns:
                    lines.append(f"  Concerns: {', '.join(validation.concerns)}")
                if validation.suggestions:
                    lines.append(f"  Suggestions: {', '.join(validation.suggestions)}")
                lines.append(f"  Repetition Detected: {validation.repetition_detected}")
                lines.append(f"  Sector Clustering: {validation.sector_clustering_detected}")
                lines.append(f"  Similar Failures: {validation.similar_setup_failures}")
                lines.append(f"  Reasoning: {validation.reasoning}")

        lines.append(f"\n{'='*50}")
        lines.append("\nProvide final decisions as JSON array.")

        return "\n".join(lines)

    def _parse_decisions(
        self,
        response: str,
        signals: list[Signal],
    ) -> list[FinalDecision]:
        """Parse LLM response into FinalDecision objects."""
        decisions = []

        # Extract JSON from response
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        signal_map = {str(s.id): s for s in signals}

        try:
            parsed = json.loads(json_str.strip())
            if isinstance(parsed, dict):
                parsed = [parsed]

            for item in parsed:
                signal_id_str = item.get("signal_id", "")
                signal = signal_map.get(signal_id_str)
                if not signal:
                    continue

                decision = FinalDecision(
                    signal_id=signal.id,
                    decision=Decision(item.get("decision", "DO_NOT_EXECUTE")),
                    final_quantity=int(item.get("final_quantity", 0)),
                    confidence=float(item.get("confidence", 0.0)),
                    reasoning=item.get("reasoning", ""),
                )
                decisions.append(decision)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse decision JSON: {e}")

        return decisions
