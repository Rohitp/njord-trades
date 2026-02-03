"""
LangGraph trading workflow.

Defines the multi-agent workflow as a directed graph:

    START → DataAgent → RiskManager → Validator → MetaAgent → END

Each node receives TradingState, calls the corresponding agent,
and returns the modified state to the next node.

Usage:
    from src.workflows.graph import trading_graph

    state = TradingState(symbols=["AAPL"])
    result = await trading_graph.ainvoke(state)
"""

from langgraph.graph import StateGraph, START, END

from src.agents import DataAgent, RiskManager, Validator, MetaAgent
from src.services.market_data import MarketDataService
from src.workflows.state import TradingState


# =============================================================================
# AGENT INSTANCES
# =============================================================================
# Created once at module load. Each agent maintains its own LLM client.
# The MarketDataService is shared across invocations.

_market_data_service = MarketDataService()
_data_agent = DataAgent(market_data_service=_market_data_service)
_risk_manager = RiskManager()
_validator = Validator()
_meta_agent = MetaAgent()


# =============================================================================
# NODE FUNCTIONS
# =============================================================================
# Each node function wraps an agent's run() method.
# LangGraph calls these with the current state and expects the updated state.


async def data_agent_node(state: TradingState) -> TradingState:
    """
    Node 1: Data Agent analyzes market data and generates signals.

    Input: TradingState with symbols to analyze
    Output: TradingState with signals[] populated
    """
    return await _data_agent.run(state)


async def risk_manager_node(state: TradingState) -> TradingState:
    """
    Node 2: Risk Manager evaluates signals against constraints.

    Input: TradingState with signals from Data Agent
    Output: TradingState with risk_assessments[] populated
    """
    return await _risk_manager.run(state)


async def validator_node(state: TradingState) -> TradingState:
    """
    Node 3: Validator checks for patterns and provides second opinion.

    Input: TradingState with signals and risk assessments
    Output: TradingState with validations[] populated
    """
    return await _validator.run(state)


async def meta_agent_node(state: TradingState) -> TradingState:
    """
    Node 4: Meta Agent synthesizes all perspectives and makes final decisions.

    Input: TradingState with signals, assessments, and validations
    Output: TradingState with final_decisions[] populated
    """
    return await _meta_agent.run(state)


# =============================================================================
# GRAPH DEFINITION
# =============================================================================


def build_trading_graph() -> StateGraph:
    """
    Build the trading workflow graph.

    Returns a compiled StateGraph that can be invoked with TradingState.

    The graph has a simple linear structure:
        START → data_agent → risk_manager → validator → meta_agent → END

    Each node is async and processes the state sequentially.
    """
    # Create a new StateGraph with TradingState as the state type
    # LangGraph will pass this state through each node
    graph = StateGraph(TradingState)

    # Add nodes - each node is an async function that takes and returns state
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("risk_manager", risk_manager_node)
    graph.add_node("validator", validator_node)
    graph.add_node("meta_agent", meta_agent_node)

    # Define the flow (linear for now, could add branching later)
    graph.add_edge(START, "data_agent")
    graph.add_edge("data_agent", "risk_manager")
    graph.add_edge("risk_manager", "validator")
    graph.add_edge("validator", "meta_agent")
    graph.add_edge("meta_agent", END)

    # Compile the graph into a runnable
    return graph.compile()


# Pre-built graph instance for direct use
trading_graph = build_trading_graph()
