"""
Trading workflow orchestration.

This module contains:
- State definitions (TradingState, Signal, etc.)
- LangGraph workflow (in graph.py - import directly to avoid circular imports)
- Cycle runner service (in runner.py - import directly to avoid circular imports)

Usage:
    from src.workflows.state import TradingState, Signal
    from src.workflows.runner import TradingCycleRunner
    from src.workflows.graph import trading_graph

    # Run via service
    runner = TradingCycleRunner(db_session)
    result = await runner.run_scheduled_cycle(["AAPL", "MSFT"])

    # Or run graph directly
    state = TradingState(symbols=["AAPL"])
    result = await trading_graph.ainvoke(state)

Note: graph.py and runner.py are NOT exported from __init__ to avoid
circular imports with the agents module.
"""

# Only export state types - these don't have circular import issues
from src.workflows.state import (
    Decision,
    ExecutionResult,
    FinalDecision,
    PortfolioSnapshot,
    RiskAssessment,
    Signal,
    SignalAction,
    TradingState,
    Validation,
)

__all__ = [
    # State types only - import graph and runner directly from their modules
    "Decision",
    "ExecutionResult",
    "FinalDecision",
    "PortfolioSnapshot",
    "RiskAssessment",
    "Signal",
    "SignalAction",
    "TradingState",
    "Validation",
]
