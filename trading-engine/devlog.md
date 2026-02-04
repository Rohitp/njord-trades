# Trading Engine Development Log

## Overview

Multi-agent LLM-powered trading system built with LangGraph. The system uses a pipeline of specialized AI agents to analyze market data, assess risk, validate patterns, and make trading decisions.

**Stack**: Python 3.12, FastAPI, LangGraph, LangChain, PostgreSQL + pgvector, Prometheus

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HTTP API (FastAPI)                              │
│  /cycles/run  /health  /portfolio  /trades  /events  /discovery  /metrics  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           │                           ▼
┌───────────────────────┐             │             ┌───────────────────────┐
│   Symbol Discovery    │             │             │   LangGraph Workflow  │
│                       │             │             │                       │
│  Metric ─┐            │             │             │  DataAgent            │
│  Fuzzy  ─┼─► Ensemble │─────────────┼────────────►│      ▼                │
│  LLM   ──┘      │     │   symbols   │             │  RiskManager          │
│                 ▼     │             │             │      ▼                │
│           Watchlist   │             │             │  Validator ◄── pgvector
│                       │             │             │      ▼        (similar│
└───────────────────────┘             │             │  MetaAgent    setups) │
          │                           │             └───────────────────────┘
          │                           │                           │
          │         ┌─────────────────┴─────────────────┐         │
          │         ▼                 ▼                 ▼         │
          │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
          │  │   LLM API   │   │ Market Data │   │  PostgreSQL │  │
          │  │ (Anthropic/ │   │  (Alpaca/   │   │  + pgvector │◄─┘
          │  │  OpenAI/    │   │  yfinance)  │   │             │
          │  │  Gemini)    │   └─────────────┘   │  Embeddings │
          │  └─────────────┘                     │  Events     │
          │         ▲                            │  Trades     │
          └─────────┴────────────────────────────┴─────────────┘
                   LLM Picker uses market context + similar conditions
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

## Completed Features

### Trade Execution ✓

**Status**: IMPLEMENTED

Execution service submits orders to broker when `execute=true` is passed to `/cycles/run`.

```
Files created:
- src/services/execution/broker.py       # Abstract Broker protocol
- src/services/execution/alpaca_broker.py # Alpaca API implementation
- src/services/execution/paper_broker.py  # Local simulation for testing
- src/services/execution/service.py       # ExecutionService orchestrator
```

**API Usage**:
```bash
# Dry run (default) - no trades executed
curl -X POST http://localhost:8000/cycles/run \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"]}'

# Live execution - execute EXECUTE decisions via broker
curl -X POST http://localhost:8000/cycles/run \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "execute": true}'
```

**Broker Selection**:
- If `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are set → AlpacaBroker
- Otherwise → PaperBroker (local simulation)

---

### Circuit Breaker ✓

**Status**: IMPLEMENTED

Automatic trading halt when risk thresholds are breached.

```
Files created:
- src/services/circuit_breaker.py   # CircuitBreakerService
- src/api/routers/system.py         # System management endpoints
```

**Trigger Conditions**:
| Condition | Threshold | Config Key |
|-----------|-----------|------------|
| Drawdown from peak | > 20% | `TRADING_DRAWDOWN_HALT_PCT` |
| Consecutive losses | ≥ 10 | `TRADING_CONSECUTIVE_LOSS_HALT` |

**Auto-Resume Conditions**:
| Condition | Threshold | Config Key |
|-----------|-----------|------------|
| Drawdown recovery | < 15% | `TRADING_DRAWDOWN_RESUME_PCT` |
| Win streak after halt | 3 wins | `TRADING_WIN_STREAK_RESUME` |

**API Endpoints**:
```bash
GET  /api/system/circuit-breaker          # Check status
POST /api/system/circuit-breaker/reset    # Manual reset
POST /api/system/circuit-breaker/evaluate # Force evaluation
POST /api/system/trading/disable          # Emergency stop
POST /api/system/trading/enable           # Re-enable
```

**Integration**:
- Checked before each trading cycle in `TradingCycleRunner`
- Evaluated after each trade execution in `ExecutionService`
- Metrics: `circuit_breaker_triggers`, `current_drawdown`, `consecutive_losses`

---

### Scheduled Execution ✓

**Status**: IMPLEMENTED

APScheduler-based automatic execution of trading cycles at configured times.

```
Files created:
- src/scheduler/__init__.py       # Module exports
- src/scheduler/triggers.py       # Market hours logic
- src/scheduler/jobs.py           # Job definitions and registration
- src/api/routers/scheduler.py    # Scheduler API endpoints
```

**Configuration** (in .env):
```bash
SCHEDULE_SCAN_TIMES=["11:00", "14:30"]    # Times in EST
SCHEDULE_TIMEZONE=America/New_York        # Trading timezone
SCHEDULE_MARKET_HOURS_ONLY=true           # Only run during market hours
SCHEDULE_WEEKDAYS_ONLY=true               # Only run on weekdays
```

**API Endpoints**:
```bash
GET  /api/scheduler/status           # Scheduler status and jobs list
GET  /api/scheduler/jobs             # List all scheduled jobs
POST /api/scheduler/jobs/{id}/trigger  # Manually trigger a job
GET  /api/scheduler/market-status    # Market open status
```

**How it works**:
1. Scheduler starts automatically with FastAPI (except in test environment)
2. Jobs are registered for each time in `SCHEDULE_SCAN_TIMES`
3. Jobs check market conditions before running (`should_run_scheduled_job`)
4. Uses watchlist symbols from `TRADING_WATCHLIST_SYMBOLS`

---

## Implementation Progress

### Phase 1: pgvector Foundation (IN PROGRESS)

**Status**: IN PROGRESS

#### 1.1 Docker & Database Setup ✓
- [x] Updated `docker-compose.yml` to use `pgvector/pgvector:pg15` image
- [x] Added `pgvector>=0.3.0` to `pyproject.toml` dependencies
- [x] Created integration tests for pgvector extension (`tests/test_database/test_pgvector.py`)
- [x] Created Alembic migration to enable `vector` extension (`ce4acf99c9b7_enable_pgvector_extension.py`)

**Files Modified**:
- `docker-compose.yml` - Changed postgres image to `pgvector/pgvector:pg15`
- `pyproject.toml` - Added `pgvector>=0.3.0` dependency
- `tests/test_database/test_pgvector.py` - Created tests for extension verification

**Testing**:
```bash
# Run pgvector tests
uv run pytest tests/test_database/test_pgvector.py -v
```

#### 1.2 Embedding Models ✓
- [x] Added `TradeEmbedding` model to `src/database/models.py`
- [x] Added `MarketConditionEmbedding` model
- [x] Added `SymbolContextEmbedding` model
- [x] Created unit tests for embedding models (`tests/test_database/test_embedding_models.py`)
- [x] Created Alembic migration for embedding tables (`7da48d279115_add_embedding_tables.py`)

**Files Modified**:
- `src/database/models.py` - Added 3 embedding models with Vector(384) columns
- `alembic/versions/7da48d279115_add_embedding_tables.py` - Migration for embedding tables
- `tests/test_database/test_embedding_models.py` - Created unit tests

**Model Details**:
- `TradeEmbedding`: Stores embeddings for completed trades (384-dim, linked to trades.id)
- `MarketConditionEmbedding`: Stores embeddings for market conditions (384-dim, with condition_metadata JSONB)
- `SymbolContextEmbedding`: Stores embeddings for symbol context (384-dim, with context_type and source_url)

**Testing**:
```bash
# Run embedding model tests
uv run pytest tests/test_database/test_embedding_models.py -v
```


#### 1.3 Embedding Service ✓
- [x] Added `sentence-transformers>=2.2.0` and `torch>=2.0.0` to `pyproject.toml`
- [x] Added `EmbeddingSettings` to `src/config.py`
- [x] Created `src/services/embeddings/__init__.py`
- [x] Created `src/services/embeddings/providers/__init__.py`
- [x] Created `src/services/embeddings/providers/bge.py` (BGE-small-en provider)
- [x] Created `src/services/embeddings/service.py` (EmbeddingService)
- [x] Created unit tests (`tests/test_services/test_embeddings.py`)

**Files Created**:
- `src/services/embeddings/__init__.py` - Module exports
- `src/services/embeddings/providers/__init__.py` - Provider exports
- `src/services/embeddings/providers/bge.py` - BGE-small-en provider (384-dim, local, free)
- `src/services/embeddings/service.py` - EmbeddingService orchestrator
- `tests/test_services/test_embeddings.py` - Unit tests

**Files Modified**:
- `pyproject.toml` - Added sentence-transformers and torch to optional dependencies
- `src/config.py` - Added EmbeddingSettings

**Features**:
- BGE-small-en provider: 384-dimensional embeddings, runs locally, free
- Async embedding generation (non-blocking)
- Batch embedding support (more efficient)
- Normalized embeddings for cosine similarity search

**Testing**:
```bash
# Run embedding service tests (fast, mocked)
uv run pytest tests/test_services/test_embeddings.py -v -m "not slow"

# Run all tests including slow ones (downloads models)
uv run pytest tests/test_services/test_embeddings.py -v
```

**Recent Fixes**:
- Moved `sentence-transformers` and `torch` to optional dependencies (`[project.optional-dependencies.embedding]`)
- Marked slow tests with `@pytest.mark.slow` (downloads 100MB+ models)
- Added mocked tests for fast CI runs (no downloads, no internet required)
- Added ImportError handling in BGESmallProvider for missing dependencies

---

## Phase 2: Symbol Discovery Foundation (IN PROGRESS)

**Status**: IN PROGRESS

#### 2.1 Base Protocols ✓
- [x] Created `src/services/discovery/__init__.py`
- [x] Created `src/services/discovery/pickers/base.py` (SymbolPicker ABC, PickerResult dataclass)
- [x] Created `src/services/discovery/scoring.py` (shared scoring utilities)

**Files Created**:
- `src/services/discovery/__init__.py` - Module initialization
- `src/services/discovery/pickers/__init__.py` - Pickers module
- `src/services/discovery/pickers/base.py` - SymbolPicker protocol + PickerResult dataclass
- `src/services/discovery/scoring.py` - normalize_score, liquidity_score, volatility_score, momentum_score

#### 2.2 Database Models ✓
- [x] Added `DiscoveredSymbol` model to `src/database/models.py`
- [x] Added `Watchlist` model
- [x] Added `PickerSuggestion` model
- [ ] Create Alembic migration (use `make migrate-diff`)

**Files Modified**:
- `src/database/models.py` - Added 3 discovery models

**Model Details**:
- `DiscoveredSymbol`: Tracks symbols discovered by pickers (symbol, picker_name, score, reason, discovered_at)
- `Watchlist`: Active watchlist management (symbol, source, active, added_at/removed_at)
- `PickerSuggestion`: Picker suggestions with forward return tracking (forward_return_1d/5d/20d for performance analysis)

**Next Steps**:
1. Run `make migrate-diff` to auto-generate migration for discovery tables
2. Review and commit migration

#### 2.3 Configuration ✓
- [x] Added `DiscoverySettings` to `src/config.py`

**Files Modified**:
- `src/config.py` - Added DiscoverySettings with picker weights, enabled pickers, LLM config, schedule settings

**Configuration Options**:
- `metric_weight`, `fuzzy_weight`, `llm_weight`: Ensemble weights (default: 0.3, 0.4, 0.3)
- `enabled_pickers`: List of enabled pickers (default: ["metric", "fuzzy", "llm"])
- `llm_picker_model`: LLM model for LLMPicker (default: claude-3-5-sonnet-20241022)
- `interval_hours`: Discovery job interval (default: 4 hours)
- `max_watchlist_size`: Maximum symbols in watchlist (default: 20)

#### 2.4 Data Sources ✓
- [x] Created `src/services/discovery/sources/__init__.py`
- [x] Created `src/services/discovery/sources/alpaca.py` (Alpaca assets API)

**Files Created**:
- `src/services/discovery/sources/__init__.py` - Module exports
- `src/services/discovery/sources/alpaca.py` - AlpacaAssetSource for fetching tradable symbols

**Features**:
- `get_tradable_symbols()`: Fetch symbols filtered by asset class and status
- `get_stocks()`: Convenience method for active US equity stocks
- Async wrapper around Alpaca SDK (non-blocking)
- Retry logic with exponential backoff

---

## Phase 3: Pickers Implementation (IN PROGRESS)

**Status**: IN PROGRESS

#### 3.1 MetricPicker ✓
- [x] Created `src/services/discovery/pickers/metric.py`
- [x] Implemented volume filter (min volume threshold)
- [x] Implemented spread filter (max spread threshold)
- [x] Created unit tests (`tests/test_services/test_discovery_pickers.py`)

**Files Created**:
- `src/services/discovery/pickers/metric.py` - MetricPicker implementation
- `tests/test_services/test_discovery_pickers.py` - Unit tests

**Files Modified**:
- `src/services/discovery/pickers/__init__.py` - Export MetricPicker

**Features**:
- Volume filter: Minimum daily volume threshold (default: 1M shares)
- Spread filter: Maximum bid-ask spread percentage (default: 1%)
- Market cap and beta filters: Placeholder (requires Alpaca fundamentals API)
- Binary scoring: Pass (score=1.0) or fail (excluded)
- Uses AlpacaAssetSource to fetch tradable symbols
- Uses MarketDataService for quote data

**Testing**:
```bash
# Run picker tests
uv run pytest tests/test_services/test_discovery_pickers.py -v
```

**Recent Fixes**:
- Fixed ZeroDivisionError in spread calculation (guard for price == 0)
- Added debug logging for failed filters (volume/spread) to aid diagnostics
- Empty migration file deleted (6feb8af8ee03) - tables already in 397ca6a8afc7

#### 3.2 FuzzyPicker ✓
- [x] Created `src/services/discovery/pickers/fuzzy.py`
- [x] Implemented liquidity score (volume vs average volume)
- [x] Implemented volatility score (RSI-based, moderate preferred)
- [x] Implemented momentum score (price vs SMA, positive momentum preferred)
- [x] Implemented sector balance score (placeholder - requires sector data)
- [x] Weighted combination of scores (configurable weights)
- [x] Created unit tests

**Files Created**:
- `src/services/discovery/pickers/fuzzy.py` - FuzzyPicker implementation

**Files Modified**:
- `src/services/discovery/pickers/__init__.py` - Export FuzzyPicker
- `tests/test_services/test_discovery_pickers.py` - Added FuzzyPicker tests

**Features**:
- **Liquidity Score**: Higher volume relative to 20-day average = higher score
- **Volatility Score**: RSI 30-70 range preferred (moderate volatility)
- **Momentum Score**: Price above SMA preferred, but not extreme moves
- **Sector Balance**: Placeholder (requires sector data from Alpaca fundamentals API)
- **Weighted Composite**: Configurable weights (default: liquidity 30%, volatility 25%, momentum 35%, sector 10%)
- **Min Threshold**: Only returns symbols above minimum score (default: 0.3)
- **Sorted Results**: Returns ranked list (highest score first)

**Scoring Logic**:
- Uses `scoring.py` utilities: `liquidity_score()`, `volatility_score()`, `momentum_score()`
- Weights are automatically normalized to sum to 1.0
- Composite score = weighted sum of all factor scores
- Results filtered by `min_score_threshold` and sorted by score

**Testing**:
```bash
# Run picker tests (includes FuzzyPicker)
uv run pytest tests/test_services/test_discovery_pickers.py -v
```

---

## TODO: Symbol Discovery System

**Status**: PLANNED
**Spec**: See `agent-instructions.md` for full requirements

### Overview

Dynamic symbol discovery with three picker strategies and vector database for semantic memory.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Symbol Discovery Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────┐   ┌────────────┐   ┌────────────┐              │
│   │   Metric   │   │   Fuzzy    │   │    LLM     │              │
│   │   Picker   │   │   Picker   │   │   Picker   │              │
│   │            │   │            │   │            │              │
│   │ Pure quant │   │ Weighted   │   │ Raw Claude │              │
│   │ pass/fail  │   │ scoring    │   │ /Gemini    │              │
│   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│                 ┌─────────────────┐     ┌─────────────────┐     │
│                 │    Ensemble     │◄───►│   pgvector      │     │
│                 │    Combiner     │     │                 │     │
│                 └────────┬────────┘     │ - Trade memory  │     │
│                          │              │ - Similar setups│     │
│                          ▼              │ - Market context│     │
│                 ┌─────────────────┐     └─────────────────┘     │
│                 │ Active Watchlist │                             │
│                 └─────────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Three Picker Strategies

| Picker | Approach | Use Case |
|--------|----------|----------|
| **Metric** | Pure quantitative filters (volume > X, spread < Y) | Baseline, always-on filter |
| **Fuzzy** | Weighted multi-factor scoring (0-1 scales) | Nuanced ranking with trade-offs |
| **LLM** | Raw Claude/Gemini call with rich context | Market regime awareness, news integration |

### Vector Database (pgvector)

Using PostgreSQL's pgvector extension for embeddings:

| Table | Purpose |
|-------|---------|
| `trade_embeddings` | Embed trade context + outcome for similarity search |
| `market_condition_embeddings` | Historical market states for regime matching |
| `symbol_context_embeddings` | News, events, analyst notes per symbol |

**Use cases**:
- "Find trades with similar setups" → Validator warning
- "What happened last time VIX was this high?" → LLM Picker context
- "Similar news events for this symbol" → DataAgent enrichment

### Implementation Phases

#### Phase 1: Foundation
- [ ] pgvector extension setup (Alembic migration)
- [ ] `EmbeddingService` with OpenAI text-embedding-3-small
- [ ] Base `SymbolPicker` protocol and `PickerResult` dataclass
- [ ] Database models: `discovered_symbols`, `watchlist`, `picker_suggestions`
- [ ] Configuration: `DiscoverySettings`, `EmbeddingSettings`

#### Phase 2: Pickers
- [ ] `MetricPicker` - volume, spread, market cap, beta filters
- [ ] `FuzzyPicker` - liquidity, volatility, momentum, sector balance scores
- [ ] `LLMPicker` - prompt with portfolio/market context, parse JSON response
- [ ] `EnsembleCombiner` - weighted merge, deduplication, ranking

#### Phase 3: Vector Integration
- [ ] Embed trades on completion (`TradeExecuted` event handler)
- [ ] Embed market conditions daily (VIX, SPY trend, sector performance)
- [ ] `FuzzyPicker`: query similar trades to adjust scores
- [ ] `LLMPicker`: include similar conditions in prompt
- [ ] `Validator`: warn on similar failed setups

#### Phase 4: Analysis & Backtesting
- [ ] `picker_suggestions` table with forward return tracking
- [ ] Background job to calculate 1d/5d/20d forward returns
- [ ] Performance comparison API (`GET /api/discovery/performance`)
- [ ] Paper trade tracker for hypothetical trades
- [ ] A/B testing: run all pickers, compare outcomes

#### Phase 5: Production Integration
- [ ] Scheduled discovery job (every 4 hours during market)
- [ ] `WatchlistManager` feeds into scheduled trading cycles
- [ ] Alerts on picker divergence (metric says no, LLM says yes)
- [ ] Admin UI for manual include/exclude

### Files to Create

```
src/services/discovery/
├── __init__.py
├── service.py                 # SymbolDiscoveryService orchestrator
├── pickers/
│   ├── __init__.py
│   ├── base.py                # SymbolPicker protocol, PickerResult
│   ├── metric.py              # MetricPicker
│   ├── fuzzy.py               # FuzzyPicker
│   └── llm.py                 # LLMPicker
├── ensemble.py                # EnsembleCombiner
├── scoring.py                 # Shared scoring utilities
├── watchlist.py               # WatchlistManager
└── sources/
    ├── __init__.py
    ├── alpaca.py              # Alpaca assets API
    └── news.py                # News API (future)

src/services/embeddings/
├── __init__.py
├── service.py                 # EmbeddingService
└── providers/
    ├── __init__.py
    ├── openai.py              # OpenAI embeddings
    └── local.py               # Sentence transformers (future)

src/api/routers/
├── discovery.py               # Discovery API endpoints
└── embeddings.py              # Embedding admin/debug endpoints

tests/test_discovery/
├── test_metric_picker.py
├── test_fuzzy_picker.py
├── test_llm_picker.py
├── test_ensemble.py
└── test_embeddings.py
```

### Configuration

```bash
# Discovery settings
DISCOVERY_METRIC_WEIGHT=0.3
DISCOVERY_FUZZY_WEIGHT=0.4
DISCOVERY_LLM_WEIGHT=0.3
DISCOVERY_ENABLED_PICKERS=["metric", "fuzzy", "llm"]
DISCOVERY_LLM_PICKER_MODEL=claude-3-5-sonnet-20241022
DISCOVERY_INTERVAL_HOURS=4
DISCOVERY_MAX_WATCHLIST_SIZE=20

# Embedding settings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

### Success Metrics

| Metric | Target |
|--------|--------|
| Picker Win Rate | > 55% of traded suggestions |
| Suggestion → Trade Rate | > 30% |
| Forward Return (5d) | > 1% average |
| Ensemble vs Best Single | > 10% improvement |
| Vector Search p95 | < 100ms |

---

## Future Extensions

### 1. Event-Triggered Cycles

**Current**: Manual event cycle trigger
**Needed**: Auto-detect >5% price moves

```
Files to create:
- src/services/event_monitor.py   # Price monitoring service
- Background task polling quotes
- Trigger run_event_cycle on threshold breach
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

### 3. LLM-Powered Observability Querying

**Current**: Raw logs and Prometheus
**Needed**: Natural language queries over logs/metrics

```
Files to create:
- src/agents/observability_agent.py
- Tools: query_logs(), query_prometheus(), get_trace()
- RAG over events table for historical analysis
```

### 4. Circuit Breaker Auto-Resume

**Current**: Logic exists in `CircuitBreakerService.check_auto_resume()`, but not automatically triggered
**Needed**: Periodic or event-driven invocation of auto-resume check

```
Logic implemented in src/services/circuit_breaker.py:
- Drawdown recovery: < 15% when circuit breaker is active
- Win streak: 3 consecutive wins after halt

Remaining work:
- Call check_auto_resume() from scheduler (when implemented)
- Or call it at start of each trading cycle attempt
- Consider: evaluate before blocking in runner.py
```

### 5. Multi-Strategy Support

**Current**: Single strategy
**Needed**: Run multiple strategies with separate allocations

```
Changes:
- Add strategy_id to TradingState
- Route signals to appropriate strategy
- Track performance per strategy
- Disable underperforming strategies independently
```

### 6. Backtesting Framework

**Current**: Live/paper only
**Needed**: Test strategies on historical data

```
Files to create:
- src/backtest/engine.py          # Replay historical data
- src/backtest/data_loader.py     # Load historical bars
- Mock market data provider returning historical data
- Performance metrics calculation
```

### 7. Alert System

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
├── test_scheduler/
│   ├── test_triggers.py        # Market hours, time parsing
│   └── test_jobs.py            # Job registration, scheduler lifecycle
├── test_workflows/
│   ├── test_state.py           # State dataclass tests
│   └── test_graph.py           # Graph compilation, execution, runner
```

Current: **79 tests passing**

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

*Last updated: 2026-02-04*
