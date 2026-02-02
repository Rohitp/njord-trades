"""
Risk Manager Agent - Evaluates signals against portfolio constraints.

The Risk Manager is the second agent in the pipeline. It:
1. Receives signals from the Data Agent
2. Checks each signal against hard constraints (cannot be overridden)
3. Evaluates soft constraints and concerns
4. Produces RiskAssessment objects that approve/reject signals

HARD CONSTRAINTS (enforced by code, not LLM):
- Max 20% of portfolio in a single position
- Max 30% in a single sector
- Must have sufficient cash for purchase
- Max 10 concurrent positions

These are checked programmatically before LLM analysis to ensure
they cannot be bypassed by LLM hallucination or prompt injection.
"""

import json
from uuid import UUID

from src.agents.base import BaseAgent
from src.config import settings
from src.workflows.state import (
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
)


# System prompt for soft constraint analysis
RISK_MANAGER_SYSTEM_PROMPT = """You are a risk manager for a trading system. Your job is to evaluate trading signals and identify risks.

ROLE:
- Evaluate signals that have passed hard constraint checks
- Identify soft concerns (warnings, not blockers)
- Suggest quantity adjustments if needed
- Provide risk scores

HARD CONSTRAINTS (already checked by system - you verify reasoning):
- Max 20% of portfolio in single position
- Max 30% in single sector
- Sufficient cash for purchase
- Max 10 concurrent positions

SOFT CONSTRAINTS (your job to evaluate):
- Concentration risk: Is this adding to an already large position?
- Correlation risk: Multiple positions in correlated assets?
- Volatility: Is the underlying unusually volatile?
- Timing: Any earnings, dividends, or events coming up?
- Market conditions: Overall market sentiment

RISK SCORE GUIDELINES:
- 0.0-0.3: Low risk, proceed normally
- 0.3-0.6: Medium risk, consider reduced position
- 0.6-0.8: High risk, significantly reduce or avoid
- 0.8-1.0: Very high risk, strong recommendation against

OUTPUT FORMAT:
Respond with a JSON array of assessment objects:
{
    "signal_id": "uuid-string",
    "approved": true/false,
    "adjusted_quantity": integer (your recommended quantity),
    "risk_score": 0.0 to 1.0,
    "concerns": ["list", "of", "concerns"],
    "reasoning": "Explanation of assessment"
}

Example:
```json
[
    {"signal_id": "abc-123", "approved": true, "adjusted_quantity": 3, "risk_score": 0.4, "concerns": ["Adding to existing tech exposure"], "reasoning": "Signal is valid but reducing quantity due to sector concentration"}
]
```
"""


class RiskManager(BaseAgent):
    """
    Evaluates trading signals against portfolio constraints.

    This agent enforces two types of constraints:
    1. HARD constraints - Checked programmatically, cannot be overridden
    2. SOFT constraints - Evaluated by LLM, provides warnings and adjustments

    Hard constraints are never sent to the LLM to prevent prompt injection
    from bypassing safety rules.
    """

    name = "RiskManager"
    model_name = settings.llm.risk_agent_model

    # Hard constraint thresholds from settings
    max_position_pct = settings.trading.max_position_pct  # 0.20 = 20%
    max_sector_pct = settings.trading.max_sector_pct  # 0.30 = 30%
    max_positions = settings.trading.max_positions  # 10

    async def run(self, state: TradingState) -> TradingState:
        """
        Evaluate all signals against risk constraints.

        For each signal:
        1. Check hard constraints programmatically
        2. If passed, send to LLM for soft constraint analysis
        3. Create RiskAssessment with results

        Args:
            state: Trading state with signals from Data Agent

        Returns:
            State with risk_assessments appended
        """
        if not state.signals:
            return state

        # Separate signals that pass hard constraints from those that don't
        passed_hard_checks = []
        for signal in state.signals:
            assessment = self._check_hard_constraints(signal, state)
            if assessment:
                # Hard constraint violated - add assessment directly
                state.risk_assessments.append(assessment)
            else:
                # Passed hard checks - needs LLM soft analysis
                passed_hard_checks.append(signal)

        # Send signals that passed hard checks to LLM for soft analysis
        if passed_hard_checks:
            try:
                soft_assessments = await self._analyze_soft_constraints(
                    passed_hard_checks, state
                )
                state.risk_assessments.extend(soft_assessments)
            except Exception as e:
                state.add_error(self.name, str(e))
                # On error, reject all remaining signals
                for signal in passed_hard_checks:
                    state.risk_assessments.append(
                        RiskAssessment(
                            signal_id=signal.id,
                            approved=False,
                            concerns=["Risk analysis failed due to error"],
                            reasoning=str(e),
                        )
                    )

        return state

    def _check_hard_constraints(
        self,
        signal: Signal,
        state: TradingState,
    ) -> RiskAssessment | None:
        """
        Check signal against hard constraints.

        These checks are done in code (not LLM) to ensure they cannot
        be bypassed through prompt injection.

        Args:
            signal: The signal to check
            state: Current trading state with portfolio snapshot

        Returns:
            RiskAssessment if constraint violated, None if passed
        """
        snapshot = state.portfolio_snapshot

        # HOLD signals don't need risk checking
        if signal.action == SignalAction.HOLD:
            return RiskAssessment(
                signal_id=signal.id,
                approved=True,
                adjusted_quantity=0,
                risk_score=0.0,
                reasoning="HOLD signal - no risk assessment needed",
            )

        # Check: Sufficient cash for BUY
        if signal.action == SignalAction.BUY:
            required_cash = signal.price * signal.proposed_quantity
            if required_cash > snapshot.cash:
                return RiskAssessment(
                    signal_id=signal.id,
                    approved=False,
                    hard_constraint_violated=True,
                    hard_constraint_reason=(
                        f"Insufficient cash. Required: ${required_cash:,.2f}, "
                        f"Available: ${snapshot.cash:,.2f}"
                    ),
                    reasoning="Hard constraint: Insufficient cash for purchase",
                )

        # Check: Max position size (20% of portfolio)
        if signal.action == SignalAction.BUY:
            position_value = signal.price * signal.proposed_quantity
            existing = snapshot.positions.get(signal.symbol, {})
            existing_value = existing.get("value", 0)
            total_position = position_value + existing_value

            max_allowed = snapshot.total_value * self.max_position_pct
            if total_position > max_allowed and snapshot.total_value > 0:
                return RiskAssessment(
                    signal_id=signal.id,
                    approved=False,
                    hard_constraint_violated=True,
                    hard_constraint_reason=(
                        f"Position would exceed {self.max_position_pct*100:.0f}% limit. "
                        f"Would be: ${total_position:,.2f}, Max: ${max_allowed:,.2f}"
                    ),
                    reasoning="Hard constraint: Maximum position size exceeded",
                )

        # Check: Max positions count
        if signal.action == SignalAction.BUY:
            current_positions = len(snapshot.positions)
            is_new_position = signal.symbol not in snapshot.positions

            if is_new_position and current_positions >= self.max_positions:
                return RiskAssessment(
                    signal_id=signal.id,
                    approved=False,
                    hard_constraint_violated=True,
                    hard_constraint_reason=(
                        f"Already at max positions ({self.max_positions}). "
                        f"Close a position before opening new ones."
                    ),
                    reasoning="Hard constraint: Maximum number of positions reached",
                )

        # Check: Max sector exposure (30%)
        if signal.action == SignalAction.BUY:
            # Get sector for this symbol (would come from market data in production)
            # For now, check if we have sector info in existing position
            existing = snapshot.positions.get(signal.symbol, {})
            sector = existing.get("sector")

            if sector and sector in snapshot.sector_exposure:
                position_value = signal.price * signal.proposed_quantity
                current_sector_exposure = snapshot.sector_exposure[sector]
                new_exposure = current_sector_exposure + position_value

                max_sector = snapshot.total_value * self.max_sector_pct
                if new_exposure > max_sector and snapshot.total_value > 0:
                    return RiskAssessment(
                        signal_id=signal.id,
                        approved=False,
                        hard_constraint_violated=True,
                        hard_constraint_reason=(
                            f"Sector {sector} would exceed {self.max_sector_pct*100:.0f}% limit. "
                            f"Would be: ${new_exposure:,.2f}, Max: ${max_sector:,.2f}"
                        ),
                        reasoning="Hard constraint: Maximum sector exposure exceeded",
                    )

        # All hard constraints passed
        return None

    async def _analyze_soft_constraints(
        self,
        signals: list[Signal],
        state: TradingState,
    ) -> list[RiskAssessment]:
        """
        Send signals to LLM for soft constraint analysis.

        Args:
            signals: Signals that passed hard constraint checks
            state: Current trading state

        Returns:
            List of RiskAssessment objects from LLM analysis
        """
        user_prompt = self._build_user_prompt(signals, state)
        response = await self._call_llm(RISK_MANAGER_SYSTEM_PROMPT, user_prompt)
        return self._parse_assessments(response, signals)

    def _build_user_prompt(
        self,
        signals: list[Signal],
        state: TradingState,
    ) -> str:
        """Build prompt with signals and portfolio context."""
        lines = [
            "Evaluate the following signals that have passed hard constraint checks.",
            "",
            "--- PORTFOLIO CONTEXT ---",
            self._format_portfolio_context(state),
            "",
            "--- SIGNALS TO EVALUATE ---",
        ]

        for signal in signals:
            lines.append(f"\nSignal ID: {signal.id}")
            lines.append(f"  Symbol: {signal.symbol}")
            lines.append(f"  Action: {signal.action.value}")
            lines.append(f"  Confidence: {signal.confidence:.2f}")
            lines.append(f"  Proposed Quantity: {signal.proposed_quantity}")
            lines.append(f"  Price: ${signal.price:.2f}")
            lines.append(f"  Reasoning: {signal.reasoning}")

            if signal.rsi_14:
                lines.append(f"  RSI: {signal.rsi_14:.1f}")

        lines.append("\nProvide risk assessments as JSON array.")

        return "\n".join(lines)

    def _parse_assessments(
        self,
        response: str,
        signals: list[Signal],
    ) -> list[RiskAssessment]:
        """Parse LLM response into RiskAssessment objects."""
        assessments = []

        # Extract JSON from response
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        # Create lookup for signals
        signal_map = {str(s.id): s for s in signals}

        try:
            parsed = json.loads(json_str.strip())
            if isinstance(parsed, dict):
                parsed = [parsed]

            for item in parsed:
                signal_id_str = item.get("signal_id", "")

                # Find matching signal
                signal = signal_map.get(signal_id_str)
                if not signal:
                    continue

                assessment = RiskAssessment(
                    signal_id=signal.id,
                    approved=item.get("approved", False),
                    adjusted_quantity=int(item.get("adjusted_quantity", 0)),
                    risk_score=float(item.get("risk_score", 0.5)),
                    concerns=item.get("concerns", []),
                    reasoning=item.get("reasoning", ""),
                )
                assessments.append(assessment)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse risk assessment JSON: {e}")

        return assessments
