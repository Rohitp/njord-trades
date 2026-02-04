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

#### 3.3 LLMPicker ✓
- [x] Created `src/services/discovery/pickers/llm.py`
- [x] Built prompt with portfolio context
- [x] Built prompt with market conditions
- [x] Parse JSON response from LLM
- [x] Handle LLM errors gracefully
- [x] Created unit tests

**Files Created**:
- `src/services/discovery/pickers/llm.py` - LLMPicker implementation

**Files Modified**:
- `src/services/discovery/pickers/__init__.py` - Export LLMPicker
- `tests/test_services/test_discovery_pickers.py` - Added LLMPicker tests

**Features**:
- **LLM-Powered Selection**: Uses Claude/OpenAI to analyze market conditions and portfolio context
- **Portfolio Context**: Considers current positions, sector exposure, diversification needs
- **Market Conditions**: Incorporates volatility, trends, sector rotation signals
- **Candidate Filtering**: Only evaluates symbols from provided candidate list
- **Score Clamping**: Ensures scores are in [0, 1] range
- **Error Handling**: Returns empty list on LLM errors (graceful degradation)
- **Retry Logic**: Uses `retry_llm_call` with exponential backoff
- **Sorted Results**: Returns ranked list (highest score first)

**LLM Configuration**:
- Model: Configurable via `settings.discovery.llm_picker_model` (default: claude-3-5-sonnet-20241022)
- Temperature: 0.3 (slightly higher for creative selection)
- Max Candidates: 50 symbols (to avoid token limits)

**Prompt Structure**:
- System prompt: Defines role, constraints, output format
- User prompt: Includes portfolio positions, market conditions, candidate symbols
- Output: JSON array with symbol, score (0.0-1.0), and reason

**Testing**:
```bash
# Run picker tests (includes all three pickers)
uv run pytest tests/test_services/test_discovery_pickers.py -v
```

#### 3.4 EnsembleCombiner ✓
- [x] Created `src/services/discovery/ensemble.py`
- [x] Implemented weighted merge of picker results
- [x] Implemented deduplication logic
- [x] Implemented ranking algorithm
- [x] Created unit tests

**Files Created**:
- `src/services/discovery/ensemble.py` - EnsembleCombiner implementation
- `tests/test_services/test_discovery_ensemble.py` - Unit tests

**Files Modified**:
- `src/services/discovery/__init__.py` - Export EnsembleCombiner

**Features**:
- **Weighted Voting**: Combines scores from multiple pickers using configurable weights
- **Deduplication**: Merges same symbol from multiple pickers into single result
- **Weighted Average**: Composite score = sum(score * weight) / sum(weights)
- **Reason Combination**: Combines reasons from all pickers that found the symbol
- **Metadata Merging**: Preserves picker-specific metadata and tracks which pickers found each symbol
- **Ranking**: Returns results sorted by composite score (highest first)
- **Case Normalization**: Handles symbol case differences (AAPL vs aapl)

**Weight Configuration**:
- Default weights: metric 30%, fuzzy 40%, llm 30% (from `settings.discovery`)
- Weights are automatically normalized to sum to 1.0
- Configurable via constructor or config file

**Usage Example**:
```python
combiner = EnsembleCombiner()
results = combiner.combine(
    metric_results=metric_picker.pick(),
    fuzzy_results=fuzzy_picker.pick(),
    llm_results=llm_picker.pick(),
)
# Returns deduplicated, ranked list of symbols
```

**Testing**:
```bash
# Run ensemble tests
uv run pytest tests/test_services/test_discovery_ensemble.py -v

# Run all discovery tests
uv run pytest tests/test_services/test_discovery*.py -v
```

---

## Phase 4: Vector Integration ✓

**Status**: COMPLETE | All sub-phases (4.1-4.5) complete

#### 4.1 Trade Embeddings ✓
- [x] Created `src/services/embeddings/trade_embedding.py`
- [x] Format trade context text (symbol, action, reasoning, outcome)
- [x] Generate embedding on trade completion
- [x] Store in `TradeEmbedding` table
- [x] Integrated into ExecutionService (automatic on trade completion)
- [x] Created unit tests

**Files Created**:
- `src/services/embeddings/trade_embedding.py` - TradeEmbeddingService
- `tests/test_services/test_trade_embedding.py` - Unit tests

**Files Modified**:
- `src/services/execution/service.py` - Integrated trade embedding generation

**Features**:
- **Automatic Generation**: Embeddings created automatically when trades are executed
- **Context Formatting**: Includes symbol, action, reasoning, technical indicators, outcome, P&L
- **Similarity Search**: `find_similar_trades()` method for finding similar trade setups
- **Graceful Degradation**: Embedding failures don't block trade execution
- **Deduplication**: Skips embedding generation if one already exists for a trade

**Context Text Format**:
```
Symbol: AAPL | Action: BUY | Quantity: 5 | Price: $150.00 | Signal reasoning: RSI oversold | Signal confidence: 0.75 | Technical indicators: RSI: 28.0, SMA_20: $148.00, Volume ratio: 2.00x | Decision reasoning: Strong signal | Risk score: 0.50 | Outcome: WIN | P&L: $50.00 | P&L %: 6.67%
```

**Integration**:
- Called automatically in `ExecutionService._persist_trade()` after trade is created
- Part of same database transaction (rolls back if trade fails)
- Non-blocking: embedding failures logged but don't prevent trade completion

**Testing**:
```bash
# Run trade embedding tests
uv run pytest tests/test_services/test_trade_embedding.py -v
```

#### 4.2 Market Condition Embeddings ✓
- [x] Created `src/services/embeddings/market_condition.py`
- [x] Collect VIX, SPY trend, sector performance
- [x] Format market context text
- [x] Generate and store embeddings
- [x] Created unit tests

**Files Created**:
- `src/services/embeddings/market_condition.py` - MarketConditionService
- `tests/test_services/test_market_condition_embedding.py` - Unit tests

**Features**:
- **Market Data Collection**: Fetches VIX, SPY trend (vs SMA_200), and sector ETF performance
- **Context Formatting**: Formats market conditions into readable text for embedding
- **Similarity Search**: `find_similar_conditions()` method for finding similar market regimes
- **Deduplication**: Skips embedding generation if one exists within 1 hour (prevents duplicates)
- **Sector Tracking**: Tracks 11 major sector ETFs (XLK, XLF, XLE, XLV, etc.)

**Market Data Collected**:
- **VIX**: Volatility index (High >25, Moderate 15-25, Low <15)
- **SPY Trend**: Price vs SMA_200, trend direction (Bullish/Bearish), percentage difference
- **Sector Performance**: Current prices for 11 major sector ETFs

**Context Text Format**:
```
VIX: 22.50 (Moderate volatility) | SPY: $450.00, Bullish trend (+2.27% vs SMA_200) | Sectors: Technology: $180.00, Financials: $40.00, ...
```

**Usage**:
```python
service = MarketConditionService()
embedding = await service.embed_market_condition(timestamp=datetime.now(), session=session)
# Can be called manually or by scheduled job
```

**Testing**:
```bash
# Run market condition embedding tests
uv run pytest tests/test_services/test_market_condition_embedding.py -v
```

#### 4.3 FuzzyPicker Vector Integration ✓
- [x] Query similar trades from `TradeEmbedding`
- [x] Adjust scores based on similar trade outcomes
- [x] Integrate into `FuzzyPicker` scoring logic
- [x] Test similarity-based scoring

**Files Modified**:
- `src/services/discovery/pickers/fuzzy.py` - Added similarity-based score adjustment
- `tests/test_services/test_discovery_pickers.py` - Added similarity adjustment test

**Features**:
- **Similarity Search**: Queries `TradeEmbedding` for similar trade setups using vector similarity
- **Outcome-Based Adjustment**: Calculates win rate from similar trades and adjusts score:
  - High win rate (e.g., 80%) → positive adjustment (boosts score)
  - Low win rate (e.g., 20%) → negative adjustment (reduces score)
  - Neutral win rate (50%) → no adjustment
- **Weighted Integration**: Similarity adjustment is weighted (default: 15% of final score)
- **Context Matching**: Builds context text from current market data (symbol, price, RSI, SMAs, volume) to match trade embedding format
- **Graceful Degradation**: If similarity search fails or no similar trades found, continues with base score

**Adjustment Calculation**:
- Finds top 5 similar trades (cosine similarity > 0.7)
- Calculates win rate: `wins / (wins + losses)`
- Converts to adjustment: `(win_rate - 0.5) * 2.0` (range: -1.0 to +1.0)
- Final score: `base_score + (adjustment * similarity_weight)`

**Usage**:
```python
# With DB session for similarity search
picker = FuzzyPicker(
    similarity_weight=0.15,  # 15% weight for similarity adjustment
    db_session=session,
)
results = await picker.pick()
```

**Testing**:
```bash
# Run picker tests (includes similarity adjustment)
uv run pytest tests/test_services/test_discovery_pickers.py::TestFuzzyPicker -v
```

#### 4.4 LLMPicker Vector Integration ✓
- [x] Query similar market conditions from `MarketConditionEmbedding`
- [x] Include similar conditions in LLM prompt
- [x] Test context enrichment

**Files Modified**:
- `src/services/discovery/pickers/llm.py` - Added vector similarity integration
- `tests/test_services/test_discovery_pickers.py` - Added vector integration tests

**Features**:
- **Similarity Search**: Queries `MarketConditionEmbedding` for similar historical market conditions using vector similarity
- **Context Enrichment**: Includes top 3 similar historical conditions in the LLM prompt under "SIMILAR HISTORICAL MARKET CONDITIONS"
- **Market Context Building**: `_build_market_context_text()` formats current market conditions for similarity search
- **Graceful Degradation**: Continues working if similarity search fails or no `db_session` is provided (backward compatible)
- **Optional Integration**: Requires `db_session` parameter to enable vector similarity (optional for backward compatibility)

**How It Works**:
1. When `LLMPicker` is initialized with a `db_session`, it queries similar historical market conditions
2. The current market context (from `context["market_conditions"]`) is formatted into text
3. Vector similarity search finds the top 3 most similar historical market conditions
4. Similar conditions are included in the LLM prompt with:
   - Date of the condition
   - Context text (VIX, SPY trend, sector performance)
   - Relevant metadata (if available)
5. The LLM uses this historical context to inform recommendations based on what worked in similar market regimes

**Prompt Enhancement**:
The LLM prompt now includes a "SIMILAR HISTORICAL MARKET CONDITIONS" section that shows:
```
SIMILAR HISTORICAL MARKET CONDITIONS:
These are market conditions from the past that are similar to the current state.
Use these to inform your recommendations based on what worked well in similar regimes.

  1. Date: 2024-01-15
     Conditions: VIX: 18.5 (Moderate volatility) | SPY: $450.0, Bullish trend
     Details: vix: 18.5, spy_trend: bullish
```

**Usage**:
```python
# With vector similarity (recommended)
picker = LLMPicker(db_session=session)
results = await picker.pick(context={
    "market_conditions": {"volatility": "Moderate", "trend": "Bullish"}
})

# Without vector similarity (backward compatible)
picker = LLMPicker()  # No db_session
results = await picker.pick(context={...})
```

**Testing**:
```bash
# Run LLM picker vector integration tests
uv run pytest tests/test_services/test_discovery_pickers.py::TestLLMPicker::test_llm_picker_vector_integration_* -v
```

#### 4.5 Validator Vector Integration ✓
- [x] Query similar failed setups from `TradeEmbedding`
- [x] Add warning to Validator if similar setup failed
- [x] Test pattern detection

**Files Modified**:
- `src/agents/validator.py` - Added vector similarity integration
- `src/workflows/graph.py` - Added context variable for db_session passing
- `src/workflows/runner.py` - Set db_session in context before workflow execution
- `tests/test_agents/test_validator.py` - Created comprehensive test suite

**Features**:
- **Similarity Search**: Queries `TradeEmbedding` for similar trade setups using vector similarity
- **Failure Detection**: Filters to only count LOSS trades (not WIN trades)
- **Context Matching**: Builds signal context text matching trade embedding format for accurate similarity search
- **Prompt Warnings**: Includes similar failed setups in LLM prompt with clear warnings
- **Failure Count**: Sets `similar_setup_failures` count in Validation (used by LLM for decision)
- **Graceful Degradation**: Continues working if similarity search fails or no `db_session` is provided (backward compatible)
- **Context Variable**: Uses `contextvars` to pass `db_session` through LangGraph workflow

**How It Works**:
1. When Validator runs with a `db_session`, it queries similar failed trade setups for each signal
2. Builds signal context text matching the trade embedding format (symbol, action, technical indicators, risk score)
3. Vector similarity search finds top 5 similar trades (cosine similarity > 0.7)
4. Filters results to only LOSS trades (outcome == "LOSS")
5. Includes similar failures in the LLM prompt with warnings:
   ```
   ⚠️  SIMILAR FAILED SETUPS FOUND: 2
   These are past trades with similar setups that resulted in LOSS:
     1. Symbol: AAPL | Action: BUY | RSI: 65.0 | SMA_20: $150.0...
     2. Symbol: AAPL | Action: BUY | RSI: 68.0 | SMA_20: $152.0...
   WARNING: 2 similar setup(s) failed in the past.
   ```
6. LLM uses this information to set `similar_setup_failures` count
7. If `similar_setup_failures > 2`, the prompt instructs rejection

**Context Variable Pattern**:
- `TradingCycleRunner` sets `_db_session_context` before invoking workflow
- `validator_node()` retrieves `db_session` from context variable
- Allows passing database session through LangGraph without modifying state schema

**Usage**:
```python
# Validator automatically uses db_session from context when available
runner = TradingCycleRunner(db_session=session)
result = await runner.run_scheduled_cycle(["AAPL", "MSFT"])
# Validator will query similar failures automatically
```

**Testing**:
```bash
# Run validator tests (includes vector integration)
uv run pytest tests/test_agents/test_validator.py -v
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
**Needed**: Telegram/email alerts for key events

```
Files to create:
- src/services/alerts/telegram.py
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

Current: **165+ tests passing** (includes discovery, embeddings, vector integration)

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

## Gap Fixes: Production Readiness

**Status**: COMPLETE | All identified gaps resolved

### Gap 3: SymbolDiscoveryService Orchestration ✓

**Problem**: Discovery system had pickers but no orchestration layer to run them, combine results, and persist to database.

**Solution**: Created `SymbolDiscoveryService` that:
- Runs all enabled pickers in parallel
- Applies `EnsembleCombiner` to merge results
- Persists `DiscoveredSymbol` and `PickerSuggestion` records
- Updates `Watchlist` with top-ranked symbols
- Handles errors gracefully (continues if one picker fails)

**Files Created**:
- `src/services/discovery/service.py` - SymbolDiscoveryService implementation
- `tests/unit/services/test_discovery_service.py` - Unit tests

**Files Modified**:
- `src/services/discovery/__init__.py` - Export SymbolDiscoveryService

**Features**:
- **Parallel Execution**: All pickers run concurrently for performance
- **Error Resilience**: Individual picker failures don't halt the cycle
- **Watchlist Management**: Automatically adds top symbols, reactivates existing entries
- **Database Persistence**: Full audit trail via DiscoveredSymbol and PickerSuggestion tables
- **Context Support**: Accepts portfolio/market context for pickers

**Usage**:
```python
from src.services.discovery.service import SymbolDiscoveryService

service = SymbolDiscoveryService(db_session=session)
result = await service.run_discovery_cycle(
    context={"portfolio_positions": [...], "market_conditions": {...}},
    update_watchlist=True,
)
# Returns: discovered_symbols, picker_suggestions, ensemble_results, watchlist_updates
```

**Testing**:
```bash
uv run pytest tests/unit/services/test_discovery_service.py -v
```

---

### Gap 4: Fundamentals Provider ✓

**Problem**: Market cap, beta, and sector filters were placeholders because no fundamentals data source existed.

**Solution**: Created `FundamentalsProvider` abstraction with:
- `AlpacaFundamentalsProvider`: Placeholder for Alpaca fundamentals API (ready for integration)
- `CachedFundamentalsProvider`: In-memory cache with TTL, falls back to Alpaca
- Integrated into `MetricPicker` and `FuzzyPicker` for sector/beta/market cap filters

**Files Created**:
- `src/services/market_data/fundamentals.py` - FundamentalsProvider ABC and implementations
- `tests/unit/services/test_fundamentals.py` - Unit tests

**Files Modified**:
- `src/services/market_data/__init__.py` - Export fundamentals classes
- `src/services/discovery/pickers/metric.py` - Use fundamentals for market cap/beta filters
- `src/services/discovery/pickers/fuzzy.py` - Use fundamentals for sector balance scoring

**Features**:
- **Abstraction**: `FundamentalsProvider` ABC allows swapping implementations
- **Caching**: `CachedFundamentalsProvider` reduces API calls with TTL-based cache
- **Graceful Degradation**: Pickers work without fundamentals (filters just skip)
- **Sector Balance**: FuzzyPicker now penalizes over-concentration (>30% per sector)

**Fundamentals Data Structure**:
```python
@dataclass
class Fundamentals:
    symbol: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None  # USD
    beta: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None
```

**Note**: Alpaca fundamentals API integration is placeholder (returns None). Ready for implementation when API access is available.

**Testing**:
```bash
uv run pytest tests/unit/services/test_fundamentals.py -v
```

---

### Gap 5: Background Processing Strategy ✓

**Problem**: Embedding generation and discovery cycles would block API calls if run synchronously.

**Solution**: Created background processing jobs using APScheduler:
- **Trade Embeddings Job**: Runs hourly, processes up to 100 new trades
- **Market Condition Embeddings Job**: Runs daily at market close (4:00 PM ET)
- **Discovery Cycle Job**: Runs weekly on Sunday evening

**Files Created**:
- `src/scheduler/background_jobs.py` - Background job definitions

**Files Modified**:
- `src/scheduler/jobs.py` - Register background jobs alongside trading cycles

**Features**:
- **Non-Blocking**: All jobs run asynchronously, don't block API
- **Error Handling**: Individual failures logged, don't crash scheduler
- **Batch Processing**: Trade embeddings job processes up to 100 trades per run
- **Deduplication**: Jobs check for existing embeddings before creating new ones
- **Automatic Registration**: Background jobs registered when scheduler starts

**Job Schedule**:
| Job | Frequency | Time | Purpose |
|-----|-----------|------|---------|
| Trade Embeddings | Hourly | Top of hour (market hours) | Process completed trades |
| Market Condition | Daily | 4:00 PM ET (market close) | Capture end-of-day state |
| Discovery Cycle | Weekly | Sunday 8:00 PM ET | Update watchlist |

**Testing**:
Jobs are tested via integration tests with scheduler. Unit tests verify job logic:
```bash
# Test background job registration
uv run pytest tests/unit/scheduler/ -v
```

---

## Phase 5: Event Monitor (Price Move Detection) ✓

**Status**: COMPLETE | All sub-phases (5.1-5.4) complete

### Overview

Event-driven trading cycles triggered by significant price moves (>5% within 15 minutes). Runs as a background job in the main FastAPI app, polling watchlist symbols every 60 seconds during market hours.

### Implementation

**Files Created**:
- `src/services/event_monitor.py` - EventMonitor service and PriceHistory class
- `src/scheduler/event_monitor_job.py` - Background job function
- `tests/unit/services/test_event_monitor.py` - Unit tests for EventMonitor
- `tests/unit/scheduler/test_event_monitor_job.py` - Unit tests for background job

**Files Modified**:
- `src/scheduler/background_jobs.py` - Registered event monitor job
- `src/config.py` - Added `enabled` field to EventMonitorSettings

### Features

**Price Tracking**:
- In-memory price history (15-minute sliding window)
- Tracks price points with timestamps
- Automatically prunes old prices outside window

**Move Detection**:
- Calculates price change percentage over time window
- Configurable threshold (default: 5%)
- Configurable time window (default: 15 minutes)
- Handles insufficient history gracefully

**Cooldown Management**:
- Per-symbol cooldown (default: 15 minutes)
- Prevents duplicate triggers for same symbol
- Tracks last trigger time per symbol

**Background Job**:
- Runs every 60 seconds via APScheduler
- Checks market hours before running
- Respects `event_monitor.enabled` config
- Calls `TradingCycleRunner.run_event_cycle()` directly
- Handles errors gracefully (doesn't crash scheduler)

**Configuration**:
```python
EVENT_MONITOR_ENABLED=true
EVENT_MONITOR_PRICE_MOVE_THRESHOLD_PCT=0.05  # 5%
EVENT_MONITOR_MOVE_WINDOW_MINUTES=15
EVENT_MONITOR_COOLDOWN_MINUTES=15
EVENT_MONITOR_POLL_INTERVAL_SECONDS=60
EVENT_MONITOR_STOCKS_ONLY=true
```

### Usage

The event monitor runs automatically when the scheduler starts. No manual intervention needed.

**Example Flow**:
1. Job runs every 60 seconds
2. Checks watchlist symbols: ["AAPL", "MSFT", "GOOGL"]
3. Fetches current prices
4. Compares to prices from 15 minutes ago
5. If AAPL moved 6% → triggers `run_event_cycle("AAPL")`
6. Cooldown prevents re-triggering AAPL for 15 minutes

### Testing

```bash
# Run event monitor tests
uv run pytest tests/unit/services/test_event_monitor.py -v
uv run pytest tests/unit/scheduler/test_event_monitor_job.py -v
```

**Test Coverage**:
- Price history tracking and windowing
- Price change calculation
- Cooldown logic
- Move detection (above/below threshold)
- Options filtering
- Error handling
- Background job execution
- Circuit breaker handling

---

## Phase 13: S&P 500 & Sentiment Integration (PLANNED)

**Status**: PLANNED | Not yet started

### Overview

This phase adds:
1. **S&P 500 Symbol Access**: LLMPicker can evaluate all S&P 500 constituents (500 symbols vs current limited set)
2. **Sentiment Data Collection**: Fetch and store news/sentiment for symbols from multiple sources
3. **Picker Integration**: All pickers can use sentiment data in their decisions

### Architecture

**S&P 500 Source**:
- Static list (updated quarterly) or API-based (S&P Dow Jones Indices)
- Cached in database or config file
- Integrated into symbol discovery flow
- LLMPicker uses S&P 500 as candidate pool instead of all Alpaca symbols

**Sentiment Pipeline**:
```
News APIs → SentimentService → NewsArticle (DB) → SentimentSnapshot (DB) → Pickers
```

**Data Flow**:
1. Background jobs fetch news articles daily for watchlist symbols
2. SentimentService analyzes and scores articles (LLM or pre-trained model)
3. Aggregated sentiment stored in `SentimentSnapshot` (daily per symbol)
4. Pickers query sentiment when evaluating symbols
5. LLMPicker includes sentiment in prompt context
6. FuzzyPicker uses sentiment as scoring factor
7. MetricPicker can filter by sentiment threshold

### Implementation Notes

**S&P 500 List**:
- Option 1: Static CSV file (update quarterly manually)
- Option 2: Fetch from S&P Dow Jones Indices API (may require subscription)
- Option 3: Use free sources (Wikipedia, financial data sites)
- Cache in database table or config file for fast access

**Sentiment Scoring**:
- Option 1: LLM-based (Claude/GPT) - Higher quality, more expensive
- Option 2: Pre-trained model (VADER, FinBERT) - Faster, cheaper, good quality
- Option 3: Hybrid - Use model for bulk, LLM for important articles
- Store both raw score (-1 to +1) and label (positive/negative/neutral)

**News Sources**:
- NewsAPI (free tier: 100 requests/day, paid: more)
- Alpaca News API (if available with subscription)
- Alpha Vantage News & Sentiment (free tier available)
- Consider multiple sources for redundancy

**Caching & Deduplication**:
- Cache articles by URL to avoid duplicates
- Check `NewsArticle.url` before inserting
- Update existing articles if re-fetched (sentiment may change)

**Rate Limits**:
- Respect API rate limits (NewsAPI, Alpha Vantage)
- Implement exponential backoff
- Queue requests if needed

**Cost Considerations**:
- LLM-based sentiment: ~$0.01-0.05 per article (depending on model)
- Pre-trained model: Free (local inference)
- News API costs: Free tier or $449/month (NewsAPI Pro)
- Estimate: 500 symbols × 5 articles/day × $0.02 = $50/day with LLM (consider model approach)

### Database Schema

**NewsArticle**:
```python
- id: UUID
- symbol: String(10) - Indexed
- headline: Text
- content: Text (full article or summary)
- source: String(100) - "newsapi", "alpaca", "alpha_vantage"
- url: String(500) - Unique index for deduplication
- published_at: DateTime(timezone=True) - Indexed
- sentiment_score: Numeric(3, 2) - -1.0 to +1.0
- sentiment_label: String(20) - "positive", "negative", "neutral"
- created_at: DateTime(timezone=True)
```

**Note**: News article embeddings are stored in the existing `SymbolContextEmbedding` table:
- `context_type="news"` for news articles
- `context_text` = headline + content summary
- `source_url` = article URL
- `timestamp` = published_at
- `embedding` = Vector(384) - Generated from headline + content
- Enables similarity search: "Find similar news patterns for this symbol"

**SentimentSnapshot**:
```python
- id: UUID
- symbol: String(10) - Indexed
- date: Date - Indexed, unique with symbol
- avg_sentiment: Numeric(3, 2) - Average sentiment score
- article_count: Integer - Number of articles analyzed
- sources: JSONB - List of sources used
- created_at: DateTime(timezone=True)
```

### Dependencies

- Phase 3.3 (LLMPicker) - Must be complete ✓
- Phase 12.4 (Discovery Service) - Must be complete ✓
- External APIs: NewsAPI, Alpaca News, or Alpha Vantage
- Optional: LLM API for sentiment analysis (or use local model)

### Integration Points

**LLMPicker Changes**:
- Accept S&P 500 symbols as candidate pool (instead of all Alpaca symbols)
- Include sentiment section in prompt:
  ```
  RECENT SENTIMENT FOR {SYMBOL}:
  - Average Sentiment: 0.65 (Positive)
  - Articles Analyzed: 12
  - Key Headlines:
    * "Company reports strong earnings" (Positive)
    * "Analyst upgrades rating" (Positive)
  ```

**FuzzyPicker Changes**:
- Add `sentiment_score` to composite score calculation
- Weight: `sentiment_weight` (default: 0.1)
- Formula: `score += sentiment_score * sentiment_weight`

**MetricPicker Changes**:
- Optional filter: `min_sentiment_threshold` (default: None)
- Reject symbols below threshold if enabled

**SymbolDiscoveryService Changes**:
- Query `SentimentSnapshot` for all candidate symbols before running pickers
- Pass sentiment data in context: `context["sentiment"] = {symbol: snapshot, ...}`

### Next Steps

1. Research S&P 500 data sources (free vs paid)
2. Choose sentiment analysis approach (LLM vs model vs hybrid)
3. Design database schema for news/sentiment
4. Implement S&P 500 source (15.1)
5. Update LLMPicker for S&P 500 access (15.2)
6. Implement sentiment providers (15.4)
7. Implement sentiment service (15.5)
8. Add background jobs (15.6)
9. Integrate into pickers (15.7)
10. Add API endpoints (15.8)

---

---

## Phase 6: Circuit Breaker Auto-Resume ✓

**Status**: COMPLETE | Completed: 2026-02-04

### Overview

Implemented auto-resume condition checking for the circuit breaker. The system now checks if recovery conditions are met before each trading cycle, but requires manual approval via API to actually resume trading.

### Implementation Details

**6.1 Scheduler Integration**:
- Modified `CircuitBreakerService.check_auto_resume()` to check conditions without auto-resetting
- Added `check_auto_resume_conditions()` method that returns `(conditions_met, resume_reason)`
- Integrated auto-resume check into:
  - `src/scheduler/jobs.py` - Before scheduled trading cycles
  - `src/workflows/runner.py` - Before event cycles
  - `src/scheduler/event_monitor_job.py` - Before event monitor cycles
- Auto-resume check logs when conditions are met but does NOT auto-reset (requires manual approval)

**6.2 Manual Approval API**:
- Added `POST /api/system/circuit-breaker/resume` endpoint
- Endpoint checks auto-resume conditions before allowing reset
- Returns detailed response with `conditions_met`, `resume_reason`, and `success` status
- Rejects resume attempts when conditions are not met
- Uses auto-resume reason in reset message when conditions are met

**Auto-Resume Conditions**:
1. **Drawdown Recovery**: Drawdown recovered to < 15% (if halted due to >20% drawdown) ✓
2. **Win Streak**: 3 consecutive wins (if halted due to 10 consecutive losses) ✓
3. **Sharpe Ratio**: > 0.3 for 7 days (⚠️ **NOT YET IMPLEMENTED** - requires historical returns calculation)

**Note**: Only drawdown recovery and win streak conditions are currently implemented. Sharpe ratio condition is explicitly not implemented and will be added in a future phase when historical returns calculation is available.

### Files Modified

- `src/services/circuit_breaker.py`:
  - Added `check_auto_resume_conditions()` method
  - Modified `check_auto_resume()` to log conditions without auto-resetting
- `src/api/routers/system.py`:
  - Added `POST /api/system/circuit-breaker/resume` endpoint
  - Added `CircuitBreakerResumeResponse` schema
- `src/scheduler/jobs.py`:
  - Added auto-resume check before scheduled cycles
- `src/workflows/runner.py`:
  - Added auto-resume check before event cycles
- `src/scheduler/event_monitor_job.py`:
  - Added auto-resume check before event monitor cycles

### Tests

- `tests/unit/services/test_circuit_breaker.py`:
  - Test drawdown recovery condition
  - Test win streak condition
  - Test insufficient conditions
  - Test logging when conditions are met
- `tests/unit/api/test_system.py`:
  - Test resume endpoint with conditions not met
  - Test resume endpoint with conditions met
  - Test resume endpoint when circuit breaker not active

### Configuration

Uses existing config values:
- `settings.trading.drawdown_resume_pct` (default: 0.15)
- `settings.trading.win_streak_resume` (default: 3)

### Next Steps

- ⚠️ **TODO**: Implement Sharpe ratio auto-resume condition (requires historical returns calculation)
- ⚠️ **TODO**: Wire auto-resume condition logging into Phase 7 alert system so operators receive notifications when conditions are met (currently only logs, no alerts)
- Add monitoring/alerting when auto-resume conditions are met (Phase 7: Alert System)

---

*Last updated: 2026-02-04 (Phase 6: Circuit Breaker Auto-Resume complete)*
