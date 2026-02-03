# Trading Engine Development Log

## Overview

Multi-agent LLM-powered trading system built with LangGraph. The system uses a pipeline of specialized AI agents to analyze market data, assess risk, validate patterns, and make trading decisions.

**Stack**: Python 3.12, FastAPI, LangGraph, LangChain, PostgreSQL, Prometheus

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HTTP API (FastAPI)                              │
│  /cycles/run  /health  /portfolio  /trades  /events  /metrics  /market/*   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Workflow (graph.py)                        │
│                                                                              │
│   START → DataAgent → RiskManager → Validator → MetaAgent → END             │
│              │              │            │            │                      │
│              ▼              ▼            ▼            ▼                      │
│          signals[]    risk_assess[]  validations[] decisions[]              │
│                                                                              │
│   Conditional edges skip downstream nodes if no data to process              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
            │   LLM API   │   │ Market Data │   │  PostgreSQL │
            │ (Anthropic/ │   │  (Alpaca/   │   │  (Events,   │
            │   OpenAI)   │   │  yfinance)  │   │  Trades,    │
            └─────────────┘   └─────────────┘   │  Portfolio) │
                                                └─────────────┘
```

---

## Agent Pipeline

| Agent | Role | Input | Output | Model |
|-------|------|-------|--------|-------|
| **DataAgent** | Analyze market data, generate signals | symbols[], market data | signals[] | claude-3-5-sonnet |
| **RiskManager** | Hard constraints (code) + soft assessment (LLM) | signals[], portfolio | risk_assessments[] | claude-3-5-sonnet |
| **Validator** | Pattern detection, second opinion | signals[], assessments | validations[] | claude-3-5-sonnet |
| **MetaAgent** | Synthesize perspectives, final decision | all above | final_decisions[] | claude-3-opus |

### Hard Constraints (enforced in code, not LLM)

- Max 20% portfolio in single position
- Max 30% portfolio in single sector
- Max 10 concurrent positions
- Insufficient cash check

### Conditional Flow

The graph uses conditional edges to skip downstream agents when there's no work:
- No signals generated → skip RiskManager, Validator, MetaAgent
- No signals approved by RiskManager → skip Validator, MetaAgent
- No signals validated → skip MetaAgent

---

## Directory Structure

```
src/
├── agents/                    # LLM-powered trading agents
│   ├── __init__.py           # Exports: DataAgent, RiskManager, Validator, MetaAgent
│   ├── base.py               # BaseAgent ABC with LLM client, retry logic
│   ├── data_agent.py         # Market analysis → signals
│   ├── risk_manager.py       # Hard constraints + LLM soft assessment
│   ├── validator.py          # Pattern detection, quality checks
│   └── meta_agent.py         # Final decision synthesis
│
├── api/                       # FastAPI HTTP layer
│   ├── main.py               # App factory, middleware, routers
│   ├── middleware.py         # Request logging, trace ID extraction
│   ├── exceptions.py         # Custom exception handlers
│   ├── schemas.py            # Pydantic request/response models
│   └── routers/              # Route handlers by domain
│       ├── cycles.py         # POST /cycles/run - trigger trading cycle
│       ├── health.py         # GET /health
│       ├── portfolio.py      # GET /api/portfolio, /api/positions
│       ├── trades.py         # GET/POST /api/trades
│       ├── events.py         # GET/POST /api/events (audit log)
│       ├── market_data.py    # GET /api/market/quote, /indicators
│       ├── capital.py        # GET/POST /api/capital
│       ├── strategies.py     # CRUD /api/strategies
│       └── metrics.py        # GET /metrics (Prometheus)
│
├── database/                  # SQLAlchemy models and connection
│   ├── connection.py         # AsyncSession factory, get_session dependency
│   └── models.py             # ORM models (see Database Schema below)
│
├── services/                  # Business logic services
│   ├── market_data/          # Market data with provider fallback
│   │   ├── provider.py       # Abstract MarketDataProvider protocol
│   │   ├── service.py        # MarketDataService with fallback logic
│   │   ├── alpaca_provider.py    # Alpaca API implementation
│   │   └── yfinance_provider.py  # yfinance fallback implementation
│   └── persistence.py        # CyclePersistenceService - saves cycle to events
│
├── utils/                     # Shared utilities
│   ├── logging.py            # structlog configuration
│   ├── metrics.py            # Prometheus metrics definitions
│   ├── retry.py              # Retry with exponential backoff
│   └── llm.py                # JSON parsing utilities for LLM responses
│
├── workflows/                 # LangGraph workflow definition
│   ├── state.py              # TradingState dataclass + Signal, RiskAssessment, etc.
│   ├── graph.py              # StateGraph definition, node functions
│   └── runner.py             # TradingCycleRunner service
│
└── config.py                  # Pydantic settings from environment
```

---

## Key Files

### `src/workflows/state.py`
Defines all data types that flow through the pipeline:
- `TradingState` - Main state object passed between nodes
- `Signal` - Output from DataAgent (symbol, action, confidence, quantity)
- `RiskAssessment` - Output from RiskManager (approved, adjusted_quantity, concerns)
- `Validation` - Output from Validator (approved, patterns detected)
- `FinalDecision` - Output from MetaAgent (EXECUTE/DO_NOT_EXECUTE)
- `PortfolioSnapshot` - Point-in-time portfolio state

### `src/workflows/graph.py`
LangGraph StateGraph wiring:
- Creates agent instances at module load (singleton pattern)
- Node functions wrap `agent.run(state)` with metrics recording
- Conditional edges check for empty data before proceeding
- Binds `cycle_id` + `trace_id` to structlog context for tracing

### `src/workflows/runner.py`
`TradingCycleRunner` service:
- `run_scheduled_cycle(symbols)` - Regular trading cycle for watchlist
- `run_event_cycle(symbol)` - Event-triggered cycle (>5% price move)
- Loads portfolio from database
- Checks circuit breaker before running
- Persists results to events table

### `src/agents/base.py`
`BaseAgent` abstract class:
- LLM client creation with provider inference from model name
- `_call_llm()` with retry logic for rate limits/server errors
- `_format_portfolio_context()` helper for prompts

### `src/agents/risk_manager.py`
Two-phase risk assessment:
1. **Hard constraints** (code): Position size, sector exposure, cash check
2. **Soft assessment** (LLM): Market conditions, correlation, timing

Hard constraint violations skip LLM call entirely.

### `src/utils/metrics.py`
Prometheus metrics:
- `trading_agent_latency_seconds` - Agent execution time histogram
- `trading_llm_call_duration_seconds` - LLM API call duration
- `trading_market_data_fallbacks_total` - Fallback provider usage
- `trading_hard_constraint_violations_total` - Constraint violations by type
- `trading_cycle_duration_seconds` - Full cycle timing

### `src/config.py`
Environment-based configuration using Pydantic Settings:
- `DatabaseSettings` - PostgreSQL connection
- `LLMSettings` - API keys, model names per agent, retry settings
- `TradingSettings` - Hard constraints, circuit breaker thresholds
- `SchedulingSettings` - Scan times, timezone
- `EventMonitorSettings` - Price move thresholds

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `system_state` | Single row - trading enabled, circuit breaker status |
| `portfolio_state` | Single row - cash, total value, peak value |
| `positions` | Current holdings (one row per symbol) |
| `trades` | Trade history with agent scores and outcome |
| `events` | Append-only audit log (JSONB data) |
| `capital_events` | Deposits, withdrawals, realized P&L |
| `strategies` | Strategy configuration and performance |

### Event Sourcing

All agent decisions are recorded in the `events` table:
- `SignalGenerated` - DataAgent output
- `RiskAssessed` - RiskManager output
- `ValidationComplete` - Validator output
- `MetaAgentDecision` - Final decision
- `TradeExecuted` - Execution result

Events use `aggregate_id` (cycle_id) for grouping related events.

---

## Tracing & Observability

### Current Implementation

1. **Trace ID Flow**
   ```
   HTTP Request (X-Request-ID header)
       → Middleware extracts/generates trace_id
       → Stored in request.state.trace_id
       → Bound to structlog context
       → Passed to TradingState.trace_id
       → Each graph node binds cycle_id + trace_id
       → All logs include both IDs
       → Response includes trace_id
   ```

2. **Structured Logging** (structlog)
   - JSON format in production
   - Console format in development
   - Context variables auto-included in all logs

3. **Prometheus Metrics**
   - `/metrics` endpoint for scraping
   - Histograms for latency (agents, LLM, market data)
   - Counters for signals, decisions, errors

### Correlation IDs

Every log line includes:
- `trace_id` - HTTP request correlation (UUID)
- `cycle_id` - Trading cycle identifier (UUID)

---

## Configuration

### Required Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=secret

# LLM (at least one required)
LLM_ANTHROPIC_API_KEY=sk-ant-...
LLM_OPENAI_API_KEY=sk-...

# Market Data (optional - falls back to yfinance)
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

### Model Configuration

```bash
LLM_DATA_AGENT_MODEL=claude-3-5-sonnet-20241022
LLM_RISK_AGENT_MODEL=claude-3-5-sonnet-20241022
LLM_VALIDATOR_MODEL=claude-3-5-sonnet-20241022
LLM_META_AGENT_MODEL=claude-3-opus-20240229
```

Provider is inferred from model name (`claude-*` → Anthropic, `gpt-*` → OpenAI).

---

## Future Extensions

### 1. Trade Execution (High Priority)

**Current**: Pipeline ends at MetaAgent decision
**Needed**: Execute trades via broker API

```
Files to create:
- src/services/execution/broker.py      # Abstract broker protocol
- src/services/execution/alpaca.py      # Alpaca implementation
- src/services/execution/paper.py       # Paper trading for testing
- src/agents/executor.py                # Execution agent (optional)

Changes:
- Add "executor" node to graph after meta_agent
- Or call broker directly from runner after getting EXECUTE decisions
- Update Position and PortfolioState after fills
- Record TradeExecuted events
```

### 2. OpenTelemetry Tracing

**Current**: Correlation IDs in logs
**Needed**: Full distributed tracing with spans

```
Dependencies to add:
- opentelemetry-api
- opentelemetry-sdk
- opentelemetry-instrumentation-fastapi
- opentelemetry-exporter-otlp (or jaeger/zipkin)

Changes:
- Wrap record_agent_execution() to create OTEL spans
- Add span attributes: agent name, signal count, decisions
- trace_id already flows end-to-end
```

### 3. Scheduled Execution

**Current**: Manual API trigger only
**Needed**: Run at configured times (11:00, 14:30 EST)

```
Options:
- APScheduler integration in FastAPI lifespan
- External cron calling /cycles/run
- Celery beat for distributed scheduling

Files to create:
- src/scheduler/jobs.py           # Job definitions
- src/scheduler/triggers.py       # Market hours logic
```

### 4. Event-Triggered Cycles

**Current**: Manual event cycle trigger
**Needed**: Auto-detect >5% price moves

```
Files to create:
- src/services/event_monitor.py   # Price monitoring service
- Background task polling quotes
- Trigger run_event_cycle on threshold breach
```

### 5. LLM-Powered Observability Querying

**Current**: Raw logs and Prometheus
**Needed**: Natural language queries over logs/metrics

```
Files to create:
- src/agents/observability_agent.py
- Tools: query_logs(), query_prometheus(), get_trace()
- RAG over events table for historical analysis
```

### 6. Vector Store for Trade Memory

**Current**: Events in PostgreSQL only
**Needed**: Semantic search over past trades

```
Files to create:
- src/services/vector_store/       # Chroma or Qdrant
- Embed trade context (reasoning, outcome)
- Validator queries similar past setups
- "Similar setup failed 3 times" detection
```

### 7. Circuit Breaker Auto-Resume

**Current**: Manual resume only
**Needed**: Auto-resume when conditions met

```
Logic to add in runner.py:
- If drawdown < 15% and was > 20%, resume
- If 3 consecutive wins after loss halt, resume
- If Sharpe > 0.3 for 7 days after Sharpe halt, resume
```

### 8. Multi-Strategy Support

**Current**: Single strategy
**Needed**: Run multiple strategies with separate allocations

```
Changes:
- Add strategy_id to TradingState
- Route signals to appropriate strategy
- Track performance per strategy
- Disable underperforming strategies independently
```

### 9. Backtesting Framework

**Current**: Live/paper only
**Needed**: Test strategies on historical data

```
Files to create:
- src/backtest/engine.py          # Replay historical data
- src/backtest/data_loader.py     # Load historical bars
- Mock market data provider returning historical data
- Performance metrics calculation
```

### 10. Alert System

**Current**: Logs only
**Needed**: Discord/email alerts for key events

```
Files to create:
- src/services/alerts/discord.py
- src/services/alerts/email.py (AWS SES)

Trigger alerts on:
- Circuit breaker activation
- Large position changes
- Daily P&L summary
- System errors
```

---

## Running the System

### Development

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Start server
uv run uvicorn src.api.main:app --reload

# Trigger a cycle
curl -X POST http://localhost:8000/cycles/run \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"]}'
```

### Database Setup

```bash
# Start PostgreSQL
docker run -d --name trading-db \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=trading \
  -p 5432:5432 \
  postgres:16

# Run migrations (when implemented)
uv run alembic upgrade head
```

---

## Test Coverage

```
tests/
├── test_agents/
│   └── test_risk_manager.py    # Hard constraints, LLM parsing, full flow
├── test_workflows/
│   ├── test_state.py           # State dataclass tests
│   └── test_graph.py           # Graph compilation, execution, runner
```

Current: **45 tests passing**

---

## Design Decisions

1. **Hard constraints in code, not LLM**: Position limits, sector limits, and cash checks are enforced in Python. LLM cannot override these.

2. **Provider inference from model name**: `claude-*` → Anthropic, `gpt-*` → OpenAI. No separate provider config needed.

3. **Conditional graph edges**: Skip agents when there's no data, reducing unnecessary LLM calls.

4. **Event sourcing**: All decisions recorded in events table for complete audit trail.

5. **Singleton agents**: Created once at module load, reused across requests.

6. **Fallback market data**: Alpaca primary, yfinance fallback. System works without Alpaca API keys.

7. **Trace ID from HTTP header**: Supports distributed tracing across services via `X-Request-ID`.

8. **Serialization in dataclasses**: Each state dataclass has a `from_dict()` classmethod for deserializing LangGraph's dict output. This keeps serialization logic co-located with schema definitions, making `_dict_to_state()` a one-liner and centralizing future schema changes.

---

*Last updated: 2026-02-03*
