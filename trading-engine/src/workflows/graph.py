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

import structlog
from langgraph.graph import StateGraph, START, END

from src.agents import DataAgent, RiskManager, Validator, MetaAgent
from src.services.market_data import MarketDataService
from src.utils.logging import get_logger
from src.utils.metrics import (
    record_agent_execution,
    agent_signals_generated,
    record_hard_constraint_violation,
)
from src.workflows.state import TradingState

log = get_logger(__name__)


def _bind_cycle_context(state: TradingState) -> None:
    """Bind cycle_id and trace_id to structlog context for all downstream logs."""
    structlog.contextvars.bind_contextvars(
        cycle_id=str(state.cycle_id),
        trace_id=state.trace_id,
    )


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
    _bind_cycle_context(state)
    log.debug("data_agent_node_start", symbols=state.symbols)

    async with record_agent_execution("DataAgent"):
        result = await _data_agent.run(state)

    # Record signal metrics
    for signal in result.signals:
        agent_signals_generated.labels(action=signal.action.value).inc()

    log.debug("data_agent_node_complete", signals_generated=len(result.signals))
    return result


async def risk_manager_node(state: TradingState) -> TradingState:
    """
    Node 2: Risk Manager evaluates signals against constraints.

    Input: TradingState with signals from Data Agent
    Output: TradingState with risk_assessments[] populated
    """
    _bind_cycle_context(state)
    log.debug("risk_manager_node_start", signals_count=len(state.signals))

    async with record_agent_execution("RiskManager"):
        result = await _risk_manager.run(state)

    # Record hard constraint violation metrics
    for ra in result.risk_assessments:
        if ra.hard_constraint_violated and ra.hard_constraint_reason:
            reason = ra.hard_constraint_reason.lower()
            if "cash" in reason:
                record_hard_constraint_violation("insufficient_cash")
            elif "position" in reason and "20%" in reason:
                record_hard_constraint_violation("max_position")
            elif "sector" in reason:
                record_hard_constraint_violation("max_sector")
            elif "max positions" in reason:
                record_hard_constraint_violation("max_positions")

    approved = len([ra for ra in result.risk_assessments if ra.approved])
    log.debug("risk_manager_node_complete", approved=approved, total=len(result.risk_assessments))
    return result


async def validator_node(state: TradingState) -> TradingState:
    """
    Node 3: Validator checks for patterns and provides second opinion.

    Input: TradingState with signals and risk assessments
    Output: TradingState with validations[] populated
    """
    _bind_cycle_context(state)
    log.debug("validator_node_start", signals_count=len(state.signals))

    async with record_agent_execution("Validator"):
        result = await _validator.run(state)

    approved = len([v for v in result.validations if v.approved])
    log.debug("validator_node_complete", approved=approved, total=len(result.validations))
    return result


async def meta_agent_node(state: TradingState) -> TradingState:
    """
    Node 4: Meta Agent synthesizes all perspectives and makes final decisions.

    Input: TradingState with signals, assessments, and validations
    Output: TradingState with final_decisions[] populated
    """
    _bind_cycle_context(state)
    log.debug("meta_agent_node_start", signals_count=len(state.signals))

    async with record_agent_execution("MetaAgent"):
        result = await _meta_agent.run(state)

    execute_count = len(result.get_execute_decisions())
    log.debug("meta_agent_node_complete", execute_decisions=execute_count, total=len(result.final_decisions))
    return result


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS
# =============================================================================
# These functions determine whether to continue to the next node or end early.


def should_continue_to_risk_manager(state: TradingState) -> str:
    """
    Check if we should proceed to risk manager.

    Only proceed if Data Agent generated signals.
    """
    if not state.signals:
        log.debug("skipping_risk_manager", reason="no_signals")
        return END
    return "risk_manager"


def should_continue_to_validator(state: TradingState) -> str:
    """
    Check if we should proceed to validator.

    Only proceed if Risk Manager approved any signals.
    """
    approved_count = len([ra for ra in state.risk_assessments if ra.approved])
    if approved_count == 0:
        log.debug("skipping_validator", reason="no_approved_signals")
        return END
    return "validator"


def should_continue_to_meta_agent(state: TradingState) -> str:
    """
    Check if we should proceed to meta agent.

    Only proceed if Validator approved any signals.
    """
    approved_count = len([v for v in state.validations if v.approved])
    if approved_count == 0:
        log.debug("skipping_meta_agent", reason="no_validated_signals")
        return END
    return "meta_agent"


# =============================================================================
# GRAPH DEFINITION
# =============================================================================


def build_trading_graph() -> StateGraph:
    """
    Build the trading workflow graph with conditional edges.

    Returns a compiled StateGraph that can be invoked with TradingState.

    The graph structure:
        START → data_agent → [conditional] → risk_manager → [conditional] → 
        validator → [conditional] → meta_agent → END

    Conditional edges skip nodes if there's no data to process.
    """
    # Create a new StateGraph with TradingState as the state type
    graph = StateGraph(TradingState)

    # Add nodes - each node is an async function that takes and returns state
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("risk_manager", risk_manager_node)
    graph.add_node("validator", validator_node)
    graph.add_node("meta_agent", meta_agent_node)

    # Define the flow with conditional edges
    graph.add_edge(START, "data_agent")
    
    # Conditional: Only go to risk_manager if signals were generated
    graph.add_conditional_edges(
        "data_agent",
        should_continue_to_risk_manager,
        {
            "risk_manager": "risk_manager",
            END: END,
        },
    )
    
    # Conditional: Only go to validator if signals were approved
    graph.add_conditional_edges(
        "risk_manager",
        should_continue_to_validator,
        {
            "validator": "validator",
            END: END,
        },
    )
    
    # Conditional: Only go to meta_agent if signals were validated
    graph.add_conditional_edges(
        "validator",
        should_continue_to_meta_agent,
        {
            "meta_agent": "meta_agent",
            END: END,
        },
    )
    
    graph.add_edge("meta_agent", END)

    # Compile the graph into a runnable
    return graph.compile()


# Pre-built graph instance for direct use
trading_graph = build_trading_graph()
