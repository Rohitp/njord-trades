"""
Trading agents module.

Each agent implements the BaseAgent protocol and handles one step of the
trading pipeline:

    DataAgent → RiskManager → Validator → MetaAgent

Usage:
    from src.agents import DataAgent, RiskManager, Validator, MetaAgent

    # Create agents
    data_agent = DataAgent()
    risk_manager = RiskManager()
    validator = Validator()
    meta_agent = MetaAgent()

    # Run pipeline
    state = TradingState(symbols=["AAPL", "MSFT"])
    state = await data_agent.run(state)
    state = await risk_manager.run(state)
    state = await validator.run(state)
    state = await meta_agent.run(state)

    # Check final decisions
    for decision in state.get_execute_decisions():
        print(f"Execute {decision.signal_id}: qty={decision.final_quantity}")
"""

from src.agents.base import BaseAgent
from src.agents.data_agent import DataAgent
from src.agents.meta_agent import MetaAgent
from src.agents.risk_manager import RiskManager
from src.agents.validator import Validator

__all__ = [
    "BaseAgent",
    "DataAgent",
    "RiskManager",
    "Validator",
    "MetaAgent",
]
