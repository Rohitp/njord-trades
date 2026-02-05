# Trading Engine User Guide

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [How Stock Picking Works](#how-stock-picking-works)
4. [The Three Pickers](#the-three-pickers)
5. [Parallel Execution](#parallel-execution)
6. [Trading Cycle Workflow](#trading-cycle-workflow)
7. [Decision Making Process](#decision-making-process)
8. [Trade Execution](#trade-execution)
9. [Safety Features](#safety-features)
10. [Configuration](#configuration)

---

## System Overview

The Trading Engine is an autonomous, multi-agent system that discovers promising stocks, analyzes them, and executes trades based on a sophisticated decision-making pipeline. Think of it as having multiple specialized analysts working together, each with their own expertise, to make trading decisions.

### Key Concepts

- **Symbol Discovery**: The system continuously finds new stocks to consider, not just trading from a fixed list
- **Multi-Agent Pipeline**: Four specialized AI agents analyze each potential trade from different angles
- **Ensemble Voting**: Multiple "pickers" evaluate stocks independently, then their opinions are combined
- **Safety First**: Hard constraints and circuit breakers prevent catastrophic losses

---

## Core Components

### 1. Symbol Discovery System

**Purpose**: Automatically find promising stocks to trade, rather than relying on a static watchlist.

**How it works**:
- Runs weekly (Sunday evening) to prepare for the week ahead
- Three different "pickers" evaluate stocks using different strategies
- Results are combined into a ranked list
- Top symbols are added to the watchlist for trading

**Key Files**:
- `src/services/discovery/service.py` - Orchestrates the discovery process
- `src/services/discovery/pickers/` - Contains the three picker implementations
- `src/services/discovery/ensemble.py` - Combines picker results

### 2. Trading Agents

**Purpose**: Four specialized AI agents analyze potential trades from different perspectives.

**The Four Agents**:

1. **Data Agent**: Analyzes market data (price, volume, technical indicators) and generates trading signals
2. **Risk Manager**: Enforces capital constraints and position limits (hard rules that cannot be overridden)
3. **Validator**: Provides a "second opinion" by checking for patterns, repetition, and timing issues
4. **Meta-Agent**: Synthesizes all perspectives and makes the final decision

**Key Files**:
- `src/agents/` - Contains all agent implementations
- `src/workflows/graph.py` - Defines how agents are orchestrated

### 3. Execution System

**Purpose**: Executes approved trades and maintains portfolio state.

**How it works**:
- Receives final decisions from Meta-Agent
- Executes trades via broker API (Alpaca)
- Updates database in a single transaction (all-or-nothing)
- Logs everything to event log for audit trail

**Key Files**:
- `src/services/execution/service.py` - Trade execution orchestration
- `src/services/execution/alpaca_broker.py` - Alpaca API integration

### 4. Safety Systems

**Purpose**: Prevent catastrophic losses and ensure system stability.

**Components**:
- **Circuit Breaker**: Automatically halts trading if losses exceed thresholds
- **Hard Constraints**: Mathematical limits (cash, position size, sector concentration) that cannot be overridden
- **Event Logging**: Complete audit trail of every decision and action

**Key Files**:
- `src/services/circuit_breaker.py` - Circuit breaker logic
- `src/database/models.py` - Database schema with constraints

---

## How Stock Picking Works

The system uses a **three-stage discovery process** to find promising stocks:

### Stage 1: Parallel Picker Execution

Three independent "pickers" evaluate stocks simultaneously:

1. **MetricPicker**: Uses hard quantitative filters (volume, spread, market cap)
2. **FuzzyPicker**: Scores stocks using weighted factors (liquidity, volatility, momentum)
3. **LLMPicker**: Uses AI reasoning to identify opportunities based on context

Each picker runs **in parallel** (simultaneously) to maximize speed. They don't wait for each other.

### Stage 2: Ensemble Combination

The `EnsembleCombiner` merges results from all three pickers:

- **Deduplication**: If multiple pickers find the same stock, it's combined into one entry
- **Weighted Scoring**: Each picker has a weight (configurable). The final score is a weighted average
- **Ranking**: Stocks are sorted by composite score (highest first)

**Example**:
```
MetricPicker finds: AAPL (score: 1.0), MSFT (score: 1.0)
FuzzyPicker finds: AAPL (score: 0.85), GOOGL (score: 0.72)
LLMPicker finds: AAPL (score: 0.90), TSLA (score: 0.65)

EnsembleCombiner combines:
- AAPL: (1.0 * 0.3) + (0.85 * 0.3) + (0.90 * 0.4) = 0.915 (top ranked)
- MSFT: 1.0 * 0.3 = 0.30
- GOOGL: 0.72 * 0.3 = 0.216
- TSLA: 0.65 * 0.4 = 0.26
```

### Stage 3: Watchlist Update

Top-ranked symbols (configurable, default: top 20) are added to the watchlist. This watchlist is what the trading agents analyze during scheduled cycles.

---

## The Three Pickers

### 1. MetricPicker (Quantitative Filters)

**Strategy**: Pure pass/fail filters. No scoring, no AI - just hard rules.

**What it checks**:
- **Volume**: Minimum daily volume (default: 1M shares/day) - ensures liquidity
- **Spread**: Maximum bid-ask spread (default: 1%) - ensures low transaction costs
- **Market Cap**: Range check (default: $100M - $10T) - filters out penny stocks and mega-caps
- **Beta**: Volatility range (default: 0.5 - 2.0) - avoids extreme volatility

**Output**: Binary - either a stock passes all filters (score: 1.0) or it doesn't (excluded).

**When to use**: Fast, reliable baseline filter. Removes obviously unsuitable stocks.

**Code Location**: `src/services/discovery/pickers/metric.py`

### 2. FuzzyPicker (Weighted Multi-Factor Scoring)

**Strategy**: Scores stocks on multiple dimensions, then combines them with weights.

**What it evaluates**:
- **Liquidity Score** (30% weight): Higher volume relative to average = better
- **Volatility Score** (25% weight): Moderate volatility preferred (not too low, not too high)
- **Momentum Score** (35% weight): Positive momentum preferred (but not extreme/overbought)
- **Sector Balance** (10% weight): Penalizes over-concentration in one sector
- **Similarity Adjustment** (15% weight): Boosts scores for stocks similar to past winning trades

**Output**: Continuous scores (0.0 - 1.0). Only stocks above threshold (default: 0.3) are returned.

**Special Feature**: Uses vector embeddings to find stocks similar to past winning trades. If a stock is similar to a trade that made money, its score is boosted.

**When to use**: Nuanced ranking that considers multiple factors and learns from history.

**Code Location**: `src/services/discovery/pickers/fuzzy.py`

### 3. LLMPicker (AI-Powered Context-Aware Selection)

**Strategy**: Uses Large Language Models (LLMs) to reason about market conditions and portfolio context.

**What it considers**:
- **Portfolio Context**: Current positions, sector exposure, diversification needs
- **Market Conditions**: Volatility, trends, sector rotation (learned from similar historical conditions)
- **Trading Opportunities**: Momentum, value, contrarian plays
- **Candidate List**: Pre-filtered by MetricPicker to avoid token limits

**How it works**:
1. Pre-filters candidates using MetricPicker (top 30 symbols)
2. Queries vector database for similar historical market conditions
3. Builds a rich prompt with portfolio state, market context, and candidate list
4. LLM analyzes and returns ranked list with scores and reasoning

**Output**: JSON array of symbol recommendations with scores (0.0 - 1.0) and reasoning.

**Special Feature**: Learns from historical market conditions. If current conditions are similar to conditions where certain stocks performed well, it factors that into recommendations.

**When to use**: Sophisticated, context-aware selection that considers the "big picture."

**Code Location**: `src/services/discovery/pickers/llm.py`

---

## Parallel Execution

### Why Parallel?

Running pickers in parallel dramatically speeds up discovery. If each picker takes 10 seconds:
- **Sequential**: 10s + 10s + 10s = 30 seconds total
- **Parallel**: max(10s, 10s, 10s) = 10 seconds total

### How It Works

The `SymbolDiscoveryService.run_discovery_cycle()` method uses Python's `asyncio.gather()` to run all pickers simultaneously:

```python
# Run all pickers concurrently
tasks = [
    run_picker(picker_name, picker)
    for picker_name, picker in self.pickers.items()
]
results_list = await asyncio.gather(*tasks)
```

**Error Handling**: If one picker fails, others continue. The failed picker returns empty results, and the ensemble continues with the remaining pickers.

**Code Location**: `src/services/discovery/service.py` (lines 88-116)

### What Runs in Parallel?

1. **MetricPicker**: Fetches all stocks from Alpaca, filters them
2. **FuzzyPicker**: Fetches market data, calculates scores, queries vector database
3. **LLMPicker**: Pre-filters candidates, queries vector database, calls LLM API

All three run simultaneously, independent of each other.

---

## Trading Cycle Workflow

A trading cycle is the process of analyzing symbols and making trading decisions. It follows a strict pipeline:

### Step 1: Cycle Initiation

**Triggered by**:
- **Scheduled**: Runs at configured times (default: 11:00 AM and 2:30 PM EST, weekdays only)
- **Event-Driven**: Triggered when a stock moves ≥5% in 15 minutes (during market hours)

**What happens**:
- Circuit breaker is checked (trading halted if active)
- Portfolio state is loaded from database
- Initial `TradingState` object is created with symbols to analyze

**Code Location**: `src/workflows/runner.py`

### Step 2: Data Agent Analysis

**Purpose**: Generate trading signals from market data.

**What it does**:
- Fetches real-time market data (price, volume, technical indicators) for each symbol
- Analyzes data using LLM to identify:
  - Breakouts above moving averages
  - Volume confirmation (>2x average)
  - RSI momentum (ideal: 60-70 range)
  - Overbought conditions (RSI >85 = avoid)

**Output**: List of `Signal` objects (BUY/SELL/HOLD) with confidence scores (0.0 - 1.0).

**Runs**: In parallel for all symbols (each symbol analyzed independently).

**Code Location**: `src/agents/data_agent.py`, `src/workflows/graph.py` (data_agent_node)

### Step 3: Risk Manager Assessment

**Purpose**: Enforce hard constraints and assess risk.

**What it does**:
- **Hard Constraints** (programmatic, cannot be overridden):
  - Checks if sufficient cash is available
  - Verifies position limits (max 20% per position)
  - Verifies sector limits (max 30% per sector)
  - Verifies max positions limit (default: 10 concurrent)
- **Soft Adjustments** (can be tuned):
  - Volatility-based sizing (reduce size if volatility >2x average)
  - Extension-based sizing (reduce size if price >30% above 200-day MA)

**Output**: List of `RiskAssessment` objects (approved/rejected/adjusted) with risk scores.

**Runs**: In parallel for all signals (read-only assessment, no state changes).

**Code Location**: `src/agents/risk_manager.py`, `src/workflows/graph.py` (risk_manager_node)

### Step 4: Validator Review

**Purpose**: Pattern recognition and quality control.

**What it checks**:
- **Repetition**: "We've traded TSLA 5 times this week"
- **Sector Clustering**: "4th tech trade today"
- **Pattern Failures**: "Similar setups failed 3x recently" (uses vector embeddings)
- **Timing Issues**: "Too close to earnings announcement"

**Output**: List of `Validation` objects (approved/rejected) with concerns and suggestions.

**Runs**: **Sequentially** (one at a time) with portfolio-level lock to prevent race conditions.

**Code Location**: `src/agents/validator.py`, `src/workflows/graph.py` (validator_node)

### Step 5: Meta-Agent Decision

**Purpose**: Final synthesis and conflict resolution.

**What it does**:
- Reviews all agent outputs (signals, risk assessments, validations)
- Synthesizes conflicting perspectives
- Makes final decision: EXECUTE or DO_NOT_EXECUTE
- Cannot override hard constraints (mathematical facts)
- Can debate soft constraints (sector concentration, volatility concerns)

**Output**: List of `FinalDecision` objects with final quantities and reasoning.

**Runs**: Can process multiple in parallel (read-only synthesis).

**Code Location**: `src/agents/meta_agent.py`, `src/workflows/graph.py` (meta_agent_node)

### Step 6: Trade Execution

**Purpose**: Execute approved trades and update database.

**What it does**:
- For each EXECUTE decision:
  1. Execute trade via broker API (Alpaca)
  2. Update OLTP tables (trades, positions, portfolio_state) in a single transaction
  3. Append to event log (immutable audit trail)
  4. If any step fails, entire transaction rolls back

**Runs**: **Sequentially** (one trade at a time) to ensure database consistency.

**Code Location**: `src/services/execution/service.py`, `src/workflows/graph.py` (executor_node)

### Workflow Diagram

```
START
  ↓
[Data Agent] ──→ signals[] (parallel for all symbols)
  ↓
[Risk Manager] ──→ risk_assessments[] (parallel, read-only)
  ↓
[Conditional: Any approved?]
  NO → END
  YES ↓
[Validator] ──→ validations[] (sequential with lock)
  ↓
[Conditional: Any validated?]
  NO → END
  YES ↓
[Meta-Agent] ──→ final_decisions[] (parallel synthesis)
  ↓
[Conditional: Any EXECUTE?]
  NO → END
  YES ↓
[Executor] ──→ execution_results[] (sequential, DB transactions)
  ↓
END
```

**Code Location**: `src/workflows/graph.py` (build_trading_graph function)

---

## Decision Making Process

### How Agents Make Decisions

Each agent uses a combination of:
1. **Programmatic Rules**: Hard constraints, mathematical calculations
2. **LLM Reasoning**: Context-aware analysis using Large Language Models

### Example: Buying AAPL

**Scenario**: System has $10,000 cash, no positions. AAPL is trading at $150.

#### 1. Data Agent

**Input**: Market data for AAPL
- Price: $150
- Volume: 2.5M (2x average)
- RSI: 65 (healthy momentum)
- Price above SMA_200

**LLM Analysis**: "Strong momentum, volume confirmation, healthy RSI. BUY signal."

**Output**: `Signal(action=BUY, symbol="AAPL", confidence=0.85, quantity=10)`

#### 2. Risk Manager

**Hard Constraints Check**:
- Cash available: $10,000 ✓
- Trade value: $1,500 (10 shares × $150) = 15% of portfolio ✓ (under 20% limit)
- No existing positions ✓ (under 10 limit)
- No sector concentration yet ✓ (under 30% limit)

**Soft Assessment** (LLM):
- Volatility: Normal ✓
- Extension: Price is 5% above 200-day MA (acceptable) ✓

**Output**: `RiskAssessment(approved=True, adjusted_quantity=10, risk_score=0.2)`

#### 3. Validator

**Pattern Checks**:
- Similar trades: Queries vector database for similar setups
- Finds: 3 similar trades, 2 wins, 1 loss (67% win rate) ✓
- Repetition: Haven't traded AAPL this week ✓
- Timing: No earnings in next 7 days ✓

**Output**: `Validation(approved=True, concerns=[], reasoning="Similar setups have 67% win rate")`

#### 4. Meta-Agent

**Synthesis**:
- Data Agent: Strong signal (0.85 confidence)
- Risk Manager: Approved, low risk (0.2)
- Validator: Approved, historical precedent

**Decision**: EXECUTE

**Output**: `FinalDecision(decision=EXECUTE, final_quantity=10, confidence=0.82)`

#### 5. Executor

**Execution**:
1. Submit order to Alpaca: BUY 10 shares AAPL at market
2. Order fills at $150.05 (slight slippage)
3. Update database:
   - Insert trade record
   - Update position (AAPL: 10 shares, avg_cost: $150.05)
   - Update portfolio (cash: $8,499.50, total_value: $10,000)
   - Insert event log entry
4. Commit transaction

**Output**: `ExecutionResult(success=True, fill_price=150.05, slippage=0.05)`

---

## Trade Execution

### Dual-Write Pattern

Every trade is written to **two places** in a single database transaction:

1. **OLTP Tables** (fast queries):
   - `trades` table: Trade history
   - `positions` table: Current holdings
   - `portfolio_state` table: Current cash and total value

2. **Event Log** (complete audit trail):
   - `events` table: Immutable, append-only log
   - Stores full reasoning from all agents
   - Enables complete audit trail and debugging

**Why**: OLTP tables enable fast queries (e.g., "What's my current P&L?"), while the event log provides complete history and reasoning (e.g., "Why did we buy AAPL?").

### Transaction Guarantees

All database operations happen in a **single PostgreSQL transaction**:

```python
BEGIN TRANSACTION;
  -- 1. Execute trade via broker
  -- 2. Update OLTP tables
  -- 3. Append to event log
COMMIT;  -- All or nothing
```

If **any** step fails, the entire transaction rolls back:
- Broker order is cancelled (if possible)
- Database remains unchanged
- Event log remains unchanged
- No orphaned records

**Code Location**: `src/services/execution/service.py` (_persist_trade method)

---

## Safety Features

### 1. Circuit Breaker

**Purpose**: Automatically halt trading if losses exceed thresholds.

**Triggers**:
- **Drawdown**: Portfolio value drops 20% from peak → trading halted
- **Consecutive Losses**: 10 consecutive losing trades → trading halted
- **Sharpe Ratio**: Negative Sharpe for 30 days → trading halted (not yet implemented)

**Auto-Resume Conditions** (requires manual approval):
- Drawdown recovers to <15%
- Win streak of 3+ trades
- Sharpe ratio >0.3 for 7 days

**Code Location**: `src/services/circuit_breaker.py`

### 2. Hard Constraints

**Purpose**: Mathematical limits that **cannot be overridden** by any agent.

**Constraints**:
- **Insufficient Cash**: Cannot buy if cash < trade value
- **Position Limits**: Max 20% of portfolio per symbol
- **Sector Limits**: Max 30% of portfolio per sector
- **Max Positions**: Max 10 concurrent positions

**Enforcement**: Risk Manager checks these programmatically (not via LLM). If violated, trade is rejected or quantity is reduced.

**Code Location**: `src/agents/risk_manager.py`

### 3. Retry Logic

**Purpose**: Handle transient failures gracefully.

**LLM Calls**:
- 3 retry attempts with exponential backoff (2s, 4s, 8s)
- Retries on: API errors, timeouts, connection errors
- After 3 failures: Returns safe default (HOLD signal with 0 confidence)

**Broker API Calls**:
- Retries on network errors
- Handles partial fills gracefully

**Code Location**: `src/utils/retry.py`

### 4. Event Logging

**Purpose**: Complete audit trail of every decision.

**What's Logged**:
- Every signal generated (with reasoning)
- Every risk assessment (with constraints checked)
- Every validation (with concerns)
- Every final decision (with synthesis)
- Every trade execution (with fill details)

**Enables**:
- Debugging: "Why did we buy TSLA?"
- Analysis: "Which validator concerns predicted losses?"
- Compliance: Complete history for regulatory requirements

**Code Location**: `src/services/persistence.py`, `src/database/models.py` (Event model)

---

## Configuration

### Key Settings

**Discovery** (`DISCOVERY_*` environment variables):
- `enabled_pickers`: Which pickers to use (default: `["metric", "fuzzy", "llm"]`)
- `metric_weight`: Weight for MetricPicker (default: 0.3)
- `fuzzy_weight`: Weight for FuzzyPicker (default: 0.3)
- `llm_weight`: Weight for LLMPicker (default: 0.4)
- `max_watchlist_size`: Top N symbols to add to watchlist (default: 20)

**Trading** (`TRADING_*` environment variables):
- `watchlist_symbols`: Initial watchlist (default: `["SPY", "QQQ", "AAPL", ...]`)
- `max_position_pct`: Max 20% per position
- `max_sector_pct`: Max 30% per sector
- `max_positions`: Max 10 concurrent positions

**Scheduling** (`SCHEDULE_*` environment variables):
- `scan_times`: When to run scheduled cycles (default: `["11:00", "14:30"]`)
- `timezone`: Trading timezone (default: `"America/New_York"`)

**Event Monitor** (`EVENT_MONITOR_*` environment variables):
- `enabled`: Enable event-driven cycles (default: `True`)
- `price_move_threshold_pct`: Trigger threshold (default: 0.05 = 5%)
- `move_window_minutes`: Time window (default: 15 minutes)
- `cooldown_minutes`: Cooldown between scans (default: 15 minutes)

**Code Location**: `src/config.py`

---

## Summary

The Trading Engine is a sophisticated, multi-layered system that:

1. **Discovers** stocks using three parallel pickers (quantitative, fuzzy, AI)
2. **Analyzes** them through a four-agent pipeline (data, risk, validation, meta)
3. **Executes** trades safely with hard constraints and circuit breakers
4. **Learns** from history using vector embeddings and similarity search
5. **Logs** everything for debugging and analysis

The system is designed for **autonomous operation** with **safety first**, making it suitable for real-money trading with appropriate capital limits.

---

## Further Reading

- **Architecture Details**: See `agent-instructions.md` for technical requirements
- **Implementation Status**: See `IMPLEMENTATION_PLAN.md` for feature roadmap
- **Development Log**: See `devlog.md` for recent changes and decisions
- **API Documentation**: See `api.md` for REST API endpoints

