"""
Trading cycle runner service.

Orchestrates the execution of trading cycles by:
1. Loading portfolio state from database
2. Invoking the LangGraph workflow
3. Returning results

Usage:
    runner = TradingCycleRunner()

    # Scheduled cycle for watchlist
    result = await runner.run_scheduled_cycle(["AAPL", "MSFT", "GOOGL"])

    # Event-triggered cycle for single symbol
    result = await runner.run_event_cycle("TSLA")
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.models import Position, PortfolioState
from src.workflows.graph import trading_graph
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


class TradingCycleRunner:
    """
    Service for running trading cycles.

    Handles the setup and teardown around the LangGraph workflow,
    including loading portfolio state and preparing the initial TradingState.
    """

    def __init__(self, db_session: AsyncSession | None = None):
        """
        Initialize the runner.

        Args:
            db_session: Optional database session. If not provided,
                        uses default portfolio snapshot.
        """
        self.db_session = db_session

    async def run_scheduled_cycle(
        self,
        symbols: list[str] | None = None,
    ) -> TradingState:
        """
        Run a scheduled trading cycle for multiple symbols.

        This is the regular cycle that runs at configured times
        (e.g., 11:00 and 14:30 EST).

        Args:
            symbols: List of symbols to analyze. Defaults to watchlist from config.

        Returns:
            TradingState with all agent outputs populated
        """
        # Use watchlist from config if no symbols provided
        if symbols is None:
            symbols = settings.trading.watchlist_symbols

        # Load current portfolio state
        portfolio = await self._load_portfolio_snapshot()

        # Create initial state
        state = TradingState(
            cycle_type="scheduled",
            symbols=symbols,
            portfolio_snapshot=portfolio,
            started_at=datetime.now(),
        )

        # Run the workflow (returns dict due to LangGraph serialization)
        result_dict = await trading_graph.ainvoke(state)

        # Convert dict back to TradingState
        return self._dict_to_state(result_dict)

    async def run_event_cycle(
        self,
        trigger_symbol: str,
    ) -> TradingState:
        """
        Run an event-triggered trading cycle for a single symbol.

        Triggered when a significant price move (>5% in 15 min) is detected.

        Args:
            trigger_symbol: The symbol that triggered the event

        Returns:
            TradingState with all agent outputs populated
        """
        # Load current portfolio state
        portfolio = await self._load_portfolio_snapshot()

        # Create initial state
        state = TradingState(
            cycle_type="event",
            trigger_symbol=trigger_symbol,
            symbols=[trigger_symbol],  # Focus on the triggering symbol
            portfolio_snapshot=portfolio,
            started_at=datetime.now(),
        )

        # Run the workflow (returns dict due to LangGraph serialization)
        result_dict = await trading_graph.ainvoke(state)

        # Convert dict back to TradingState
        return self._dict_to_state(result_dict)

    def _dict_to_state(self, data: dict[str, Any]) -> TradingState:
        """
        Convert LangGraph output dict back to TradingState.

        LangGraph serializes dataclasses to dicts, so we need to
        reconstruct the TradingState and its nested objects.
        """
        # Reconstruct nested objects from dicts
        signals = [
            Signal(**s) if isinstance(s, dict) else s
            for s in data.get("signals", [])
        ]

        risk_assessments = [
            RiskAssessment(**ra) if isinstance(ra, dict) else ra
            for ra in data.get("risk_assessments", [])
        ]

        validations = [
            Validation(**v) if isinstance(v, dict) else v
            for v in data.get("validations", [])
        ]

        final_decisions = [
            FinalDecision(**fd) if isinstance(fd, dict) else fd
            for fd in data.get("final_decisions", [])
        ]

        execution_results = [
            ExecutionResult(**er) if isinstance(er, dict) else er
            for er in data.get("execution_results", [])
        ]

        portfolio = data.get("portfolio_snapshot", {})
        if isinstance(portfolio, dict):
            portfolio = PortfolioSnapshot(**portfolio)

        return TradingState(
            cycle_id=data.get("cycle_id"),
            cycle_type=data.get("cycle_type", "scheduled"),
            trigger_symbol=data.get("trigger_symbol"),
            started_at=data.get("started_at"),
            symbols=data.get("symbols", []),
            portfolio_snapshot=portfolio,
            signals=signals,
            risk_assessments=risk_assessments,
            validations=validations,
            final_decisions=final_decisions,
            execution_results=execution_results,
            errors=data.get("errors", []),
        )

    async def _load_portfolio_snapshot(self) -> PortfolioSnapshot:
        """
        Load current portfolio state from database.

        If no database session is available or portfolio doesn't exist,
        returns a default snapshot with initial capital.

        Returns:
            PortfolioSnapshot with current cash, positions, etc.
        """
        if self.db_session is None:
            # No database - return default with initial capital
            return PortfolioSnapshot(
                cash=settings.trading.initial_capital,
                total_value=settings.trading.initial_capital,
                deployed_capital=0.0,
                peak_value=settings.trading.initial_capital,
                positions={},
                sector_exposure={},
            )

        try:
            # Load portfolio state (single row table)
            result = await self.db_session.execute(
                select(PortfolioState).where(PortfolioState.id == 1)
            )
            portfolio_state = result.scalar_one_or_none()

            if portfolio_state is None:
                # No portfolio in DB yet - use initial capital
                return PortfolioSnapshot(
                    cash=settings.trading.initial_capital,
                    total_value=settings.trading.initial_capital,
                    deployed_capital=0.0,
                    peak_value=settings.trading.initial_capital,
                    positions={},
                    sector_exposure={},
                )

            # Load positions
            positions_result = await self.db_session.execute(
                select(Position).where(Position.portfolio_state_id == 1)
            )
            positions = positions_result.scalars().all()

            # Build positions dict and sector exposure
            positions_dict: dict[str, dict[str, Any]] = {}
            sector_exposure: dict[str, float] = {}

            for pos in positions:
                positions_dict[pos.symbol] = {
                    "quantity": pos.quantity,
                    "value": pos.current_value,
                    "sector": pos.sector,
                    "avg_cost": pos.average_cost,
                }

                # Accumulate sector exposure
                if pos.sector:
                    sector_exposure[pos.sector] = (
                        sector_exposure.get(pos.sector, 0.0) + pos.current_value
                    )

            return PortfolioSnapshot(
                cash=portfolio_state.cash,
                total_value=portfolio_state.total_value,
                deployed_capital=portfolio_state.deployed_capital,
                peak_value=portfolio_state.peak_value,
                positions=positions_dict,
                sector_exposure=sector_exposure,
            )

        except Exception:
            # On any DB error, return safe default
            return PortfolioSnapshot(
                cash=settings.trading.initial_capital,
                total_value=settings.trading.initial_capital,
                deployed_capital=0.0,
                peak_value=settings.trading.initial_capital,
                positions={},
                sector_exposure={},
            )
