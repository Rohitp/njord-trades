# Implementation Plan

## Overview

**Core Trading Agent** (Phases 1-7): The autonomous trading system with 4 agents, workflow, execution, circuit breakers, and alerts. **Functionally complete after Phase 7.**

**Enhancements & Analysis** (Phases 8, 12-13): Discovery analysis, production integration, S&P 500 & sentiment data. **Enhancements to core functionality.**

**Observability & Operations** (Phase 9): Grafana dashboards, Langfuse tracing, Ops Portal chat. **Monitoring and debugging tools.**

**Evaluation & Deployment** (Phases 10-11, 14-15): Testing framework, CI/CD, paper trading, production launch. **Infrastructure and validation.**

## Phase 1: pgvector Foundation ✓

**Status**: COMPLETE | See devlog.md "Phase 1: pgvector Foundation" section

### 1.1 Docker & Database Setup ✓
- [x] Update `docker-compose.yml` to use `pgvector/pgvector:pg15` image
- [x] Create Alembic migration to enable `vector` extension
- [x] Test extension is enabled: `SELECT * FROM pg_extension WHERE extname = 'vector';`

### 1.2 Embedding Models ✓
- [x] Add `TradeEmbedding` model to `src/database/models.py`
- [x] Add `MarketConditionEmbedding` model
- [x] Add `SymbolContextEmbedding` model
- [x] Create Alembic migration for embedding tables
- [x] Add GIN indexes for similarity search

### 1.3 Embedding Service ✓
- [x] Create `src/services/embeddings/__init__.py`
- [x] Create `src/services/embeddings/service.py` (EmbeddingService)
- [x] Create `src/services/embeddings/providers/__init__.py`
- [x] Create `src/services/embeddings/providers/bge.py` (BGE-small-en provider)
- [x] Add `EmbeddingSettings` to `src/config.py`
- [x] Add `sentence-transformers>=2.2.0` and `torch>=2.0.0` to optional dependencies
- [x] Test embedding generation and similarity search

---

## Phase 2: Symbol Discovery Foundation ✓

**Status**: COMPLETE | See devlog.md "Phase 2: Symbol Discovery Foundation" section

### 2.1 Base Protocols ✓
- [x] Create `src/services/discovery/__init__.py`
- [x] Create `src/services/discovery/pickers/base.py` (SymbolPicker protocol, PickerResult dataclass)
- [x] Create `src/services/discovery/scoring.py` (shared scoring utilities)

### 2.2 Database Models ✓
- [x] Add `DiscoveredSymbol` model to `src/database/models.py`
- [x] Add `Watchlist` model
- [x] Add `PickerSuggestion` model
- [x] Create Alembic migration for discovery tables

### 2.3 Configuration ✓
- [x] Add `DiscoverySettings` to `src/config.py`
- [x] Add discovery weights, picker config, interval settings

### 2.4 Data Sources ✓
- [x] Create `src/services/discovery/sources/__init__.py`
- [x] Create `src/services/discovery/sources/alpaca.py` (Alpaca assets API)

---

## Phase 3: Pickers Implementation (IN PROGRESS)

**Status**: IN PROGRESS | See devlog.md "Phase 3: Pickers Implementation" section

### 3.1 MetricPicker ✓
- [x] Create `src/services/discovery/pickers/metric.py`
- [x] Implement volume filter (min volume threshold)
- [x] Implement spread filter (max spread threshold)
- [x] Implement market cap filter (min/max range) - Uses FundamentalsProvider
- [x] Implement beta filter (volatility range) - Uses FundamentalsProvider
- [x] Test with real symbols

### 3.2 FuzzyPicker ✓
- [x] Create `src/services/discovery/pickers/fuzzy.py`
- [x] Implement liquidity score (0-1 scale)
- [x] Implement volatility score
- [x] Implement momentum score
- [x] Implement sector balance score - Uses FundamentalsProvider
- [x] Weighted combination of scores
- [x] Test scoring logic

### 3.3 LLMPicker ✓
- [x] Create `src/services/discovery/pickers/llm.py`
- [x] Build prompt with portfolio context
- [x] Build prompt with market conditions
- [x] Parse JSON response from LLM
- [x] Handle LLM errors gracefully
- [x] Test with various market conditions
- [ ] **Note**: S&P 500 symbol access will be added in Phase 13.1-13.2
- [ ] **Note**: Sentiment data integration will be added in Phase 13.7

### 3.4 EnsembleCombiner ✓
- [x] Create `src/services/discovery/ensemble.py`
- [x] Implement weighted merge of picker results
- [x] Implement deduplication logic
- [x] Implement ranking algorithm
- [x] Test ensemble output quality

---

## Phase 4: Vector Integration

### 4.1 Trade Embeddings ✓
- [x] Create event handler for `TradeExecuted` events (integrated into ExecutionService)
- [x] Format trade context text (symbol, action, reasoning, outcome)
- [x] Generate embedding on trade completion
- [x] Store in `TradeEmbedding` table
- [x] Test embedding generation

### 4.2 Market Condition Embeddings ✓
- [x] Create service to embed market conditions (MarketConditionService)
- [x] Collect VIX, SPY trend, sector performance
- [x] Format market context text
- [x] Generate and store embeddings
- [x] Test embedding service
- [x] Create scheduled job (background job runs daily at market close)

### 4.3 FuzzyPicker Vector Integration ✓
- [x] Query similar trades from `TradeEmbedding`
- [x] Adjust scores based on similar trade outcomes
- [x] Integrate into `FuzzyPicker` scoring logic
- [x] Test similarity-based scoring

### 4.4 LLMPicker Vector Integration ✓
- [x] Query similar market conditions
- [x] Include similar conditions in LLM prompt
- [x] Test context enrichment

### 4.5 Validator Vector Integration ✓
- [x] Query similar failed setups from `TradeEmbedding`
- [x] Add warning to Validator if similar setup failed
- [x] Test pattern detection

---

## Phase 5: Event Monitor (Price Move Detection) ✓

**Status**: COMPLETE | See devlog.md "Phase 5: Event Monitor" section

### 5.1 Price Monitoring Service ✓
- [x] Create `src/services/event_monitor.py`
- [x] Implement price tracking (store last price per symbol)
- [x] Track price history (15-minute window in memory)
- [x] Detect 5% price moves (current vs 15min ago)
- [x] Implement 15-minute cooldown between scans (per symbol)
- [x] Stocks only (no options)

### 5.2 Background Job ✓
- [x] Create `src/scheduler/event_monitor_job.py`
- [x] Implement `monitor_price_moves_job()` function
- [x] Poll watchlist symbols every 60 seconds during market hours
- [x] Call `TradingCycleRunner.run_event_cycle()` directly (not via HTTP)
- [x] Handle errors gracefully (don't crash scheduler)
- [x] Log all triggers and price moves

### 5.3 Scheduler Integration ✓
- [x] Register job in `src/scheduler/background_jobs.py`
- [x] Use `IntervalTrigger` (every 60 seconds, job checks market hours)
- [x] Add to `register_background_jobs()` function
- [x] Test job execution

### 5.4 Configuration ✓
- [x] Add `event_monitor.enabled` to `src/config.py` (default: True)
- [x] Add `event_monitor.poll_interval_seconds` (default: 60)
- [x] Add `event_monitor.price_move_threshold_pct` (default: 0.05)
- [x] Add `event_monitor.cooldown_minutes` (default: 15)

---

## Phase 6: Circuit Breaker Auto-Resume ✓

**Status**: COMPLETE | See devlog.md "Phase 6: Circuit Breaker Auto-Resume" section

### 6.1 Scheduler Integration ✓
- [x] Call `check_auto_resume()` before each trading cycle
- [x] Integrate into `src/scheduler/jobs.py`
- [x] Integrate into `src/workflows/runner.py` (event cycles)
- [x] Integrate into `src/scheduler/event_monitor_job.py`
- [x] Test auto-resume conditions:
  - Drawdown recovery < 15% ✓
  - Win streak of 3+ ✓
  - Sharpe ratio > 0.3 for 7 days ⚠️ **NOT YET IMPLEMENTED** (requires historical returns calculation)

### 6.2 Manual Approval ✓
- [x] Require manual approval via API before resuming
- [x] Add endpoint `POST /api/system/circuit-breaker/resume`
- [x] Check conditions before allowing reset
- [x] Test approval flow
- [x] Write unit tests for service and API endpoint

---

## Phase 7: Alert System ✓

**Status**: COMPLETE | See devlog.md "Phase 7: Alert System" section

### 7.1 Telegram Integration ✓
- [x] Create `src/services/alerts/__init__.py`
- [x] Create `src/services/alerts/telegram.py`
- [x] Implement Telegram bot client with retry logic
- [x] Create alert templates (HTML formatting, emojis)
- [x] Test message delivery via API endpoint
- [x] Add `POST /api/system/alerts/test` endpoint

### 7.2 Alert Service ✓
- [x] Create `src/services/alerts/service.py`
- [x] Implement alert routing (Telegram provider only)
- [x] Add alert triggers:
  - Circuit breaker activation ✓
  - Position changes (on trade execution, gated by severity rules) ✓
  - Auto-resume conditions met ✓
  - System errors (template ready)
  - Daily P&L summary (template ready, needs background job)
- [x] Integrate into circuit breaker service
- [x] Integrate into execution service (position changes)
- [x] Write unit tests for Telegram client and alert service
- [ ] Add daily P&L summary background job (can be done in Phase 8 or later)

**Note**: Telegram Bot Query Interface moved to Phase 9.1 (fits better with observability stack)

---

## Phase 8: Discovery Analysis

### 8.1 Forward Return Tracking ✓
- [x] Add forward return fields to `PickerSuggestion` model (already existed)
- [x] Create background job to calculate 1d/5d/20d returns
- [x] Update suggestions with actual returns
- [x] Test return calculation (unit tests complete)

### 8.2 Performance API ✓
- [x] Create `src/api/routers/discovery.py`
- [x] Implement `GET /api/discovery/performance` endpoint
- [x] Compare picker performance
- [x] Return win rates, average returns
- [x] Test API endpoint (unit tests complete)

### 8.3 Paper Trade Tracker ✓
- [x] Track hypothetical trades from suggestions
- [x] Compare to actual watchlist trades
- [x] Calculate A/B test metrics
- [x] Create API endpoint `GET /api/discovery/ab-test`
- [x] Write unit tests (service and API)
- [x] **Improvements**:
  - [x] Add `suggested_price` field to capture price at suggestion time (realistic paper trading)
  - [x] Update `SymbolDiscoveryService` to fetch and store prices when creating suggestions
  - [x] Update `paper_tracker` to use `suggested_price` instead of fixed $1k notional
  - [x] Show pending suggestions in performance API (don't hide pickers waiting for forward return calculation)
  - [x] Add pagination to `/api/discovery/performance` endpoint

---

## Phase 9: Observability & Operations Portal

**Status**: PLANNED | Grafana + Langfuse + Ops Portal architecture

### 9.1 Telegram Bot Query Interface (REQUIRED)

**Status**: PLANNED | Provides basic query capabilities via Telegram commands for quick checks

**Note**: This provides mobile-friendly query interface. Full observability stack (Grafana + Ops Portal) provides dashboards, tracing, and advanced chat interface.

#### 9.1.1 Bot Implementation
- [ ] Create `src/services/alerts/telegram_bot.py` (bot command handler)
- [ ] Implement webhook endpoint `POST /api/system/telegram/webhook` for Telegram updates
- [ ] Parse incoming Telegram messages and commands
- [ ] Add authentication (verify chat_id matches configured chat_id)
- [ ] Add rate limiting (prevent spam, e.g., max 10 commands per minute)
- [ ] Handle errors gracefully (send error messages to user)

#### 9.1.2 Command Handlers
- [ ] `/status` - System status:
  - Trading enabled/disabled
  - Circuit breaker status
  - Portfolio total value
  - Cash available
  - Number of positions
- [ ] `/portfolio` - Current holdings:
  - List all positions (symbol, quantity, value, P&L)
  - Cash balance
  - Sector allocation summary
  - Total portfolio value
- [ ] `/trades` - Recent trades:
  - Last N trades (default: 10, configurable)
  - Format: symbol, action, quantity, price, outcome, timestamp
  - Optional: `/trades 20` for last 20 trades
- [ ] `/metrics` - Performance metrics:
  - Sharpe ratio (30-day)
  - Win rate (30-day)
  - Current drawdown
  - Total P&L (today, week, month, all-time)
  - Alpha vs deposits
- [ ] `/logs` - Query recent logs:
  - Filter by level (ERROR, WARNING, INFO)
  - Filter by component (agent name, service name)
  - Time range (last hour, day, week)
  - Example: `/logs ERROR last_hour`
- [ ] `/query` - Natural language query:
  - Uses LLM to convert natural language to SQL/API calls
  - Examples:
    - "What trades did we make on AAPL this week?"
    - "Show me all losing trades in the last month"
    - "What's the current exposure to tech stocks?"
  - Integrates with existing services (portfolio, trades, performance)

#### 9.1.3 Service Integration
- [ ] Integrate with portfolio state queries (`PortfolioState`, `Position` models)
- [ ] Integrate with trade history queries (`Trade` model)
- [ ] Integrate with performance analytics (calculate Sharpe, win rate, drawdown)
- [ ] Integrate with log search (query structured logs from PostgreSQL or Loki)
- [ ] Integrate with Langfuse for `/query` LLM calls (trace natural language queries)

#### 9.1.4 Testing & Documentation
- [ ] Test all commands with real Telegram bot
- [ ] Test authentication (reject messages from unauthorized chat_id)
- [ ] Test rate limiting (verify spam prevention)
- [ ] Test error handling (invalid commands, service failures)
- [ ] Document bot commands in README
- [ ] Add command examples and usage guide

### 9.2 Grafana Integration (REQUIRED)

**Status**: REQUIRED | Handles dashboards, alerts, log panels

**Note**: Prometheus server is not currently running. The `/metrics` endpoint exists and exposes Prometheus-format metrics, but Prometheus server + Grafana setup will be done in this phase.

#### 9.2.1 Prometheus Setup
- [ ] Add Prometheus to `docker-compose.yml`
- [ ] Create `prometheus.yml` configuration file
- [ ] Configure Prometheus to scrape `/metrics` endpoint (scrape interval: 15s)
- [ ] Set up persistent storage for Prometheus data
- [ ] Configure retention policy (e.g., 30 days)
- [ ] Test Prometheus can scrape metrics endpoint
- [ ] Access Prometheus UI at `http://localhost:9045` (or configured port)

#### 9.2.2 Grafana Setup
- [ ] Add Grafana to `docker-compose.yml`
- [ ] Configure Grafana with persistent storage
- [ ] Set up Prometheus as data source in Grafana
- [ ] Configure PostgreSQL as data source (for custom queries)
- [ ] Set up authentication (API key or OAuth)
- [ ] Access Grafana UI at `http://localhost:3045` (or configured port)
- [ ] Create initial dashboard structure

#### 9.2.3 Grafana Dashboards
- [ ] **Portfolio Dashboard**:
  - [ ] Real-time portfolio value (time series panel)
  - [ ] Position breakdown (pie chart)
  - [ ] Sector allocation (bar chart)
  - [ ] Cash vs deployed capital (stat panel)
  - [ ] Query from PostgreSQL `portfolio_state` and `positions` tables
- [ ] **Performance Dashboard**:
  - [ ] P&L over time (time series from `trades` table)
  - [ ] Win rate by confidence bucket (bar chart)
  - [ ] Sharpe ratio trend (calculated metric)
  - [ ] Drawdown visualization (area chart)
  - [ ] Query from `trades` and `portfolio_state` tables
- [ ] **Trading Activity Dashboard**:
  - [ ] Recent trades timeline (table panel)
  - [ ] Trade outcomes distribution (pie chart)
  - [ ] Symbol frequency (bar chart)
  - [ ] Query from `trades` table
- [ ] **Discovery Dashboard**:
  - [ ] Picker performance comparison (bar chart)
  - [ ] Watchlist changes over time (time series)
  - [ ] Discovery cycle results (table)
  - [ ] Query from `discovered_symbols` and `picker_suggestions` tables
- [ ] **Circuit Breaker Dashboard**:
  - [ ] Circuit breaker status (stat panel)
  - [ ] Drawdown history (time series)
  - [ ] Consecutive losses tracking (bar chart)
  - [ ] Query from `system_state` and `trades` tables
- [ ] **System Metrics Dashboard**:
  - [ ] CPU, memory, request latency (from Prometheus)
  - [ ] Trading cycle duration (from Prometheus metrics)
  - [ ] Agent execution time (from Prometheus metrics)
  - [ ] Error rates (from Prometheus metrics)

#### 9.2.4 Grafana Alerts
- [ ] Configure alert rules for:
  - [ ] Circuit breaker activation
  - [ ] High drawdown (>15%)
  - [ ] System errors (error rate spike)
  - [ ] Trading cycle failures
- [ ] Set up notification channels (Telegram integration)
- [ ] Test alert delivery

#### 9.2.5 Grafana Log Panels
- [ ] Configure Loki as log aggregation (or use PostgreSQL log queries)
- [ ] Create log panels for:
  - [ ] Trading cycle logs
  - [ ] Agent execution logs
  - [ ] Error logs
  - [ ] System logs
- [ ] Set up log search and filtering
- [ ] Link logs to traces (via Langfuse)

### 9.3 Langfuse Integration (REQUIRED)

**Status**: REQUIRED | Provides LLM observability, tracing, and prompt visibility (open source, free)

#### 9.3.1 Langfuse Setup
- [ ] Add Langfuse dependencies to `pyproject.toml` (`langfuse>=2.0.0`)
- [ ] Configure Langfuse credentials in `.env`:
  - [ ] `LANGFUSE_PUBLIC_KEY` (for cloud) or self-hosted URL
  - [ ] `LANGFUSE_SECRET_KEY` (for cloud) or skip for self-hosted
  - [ ] `LANGFUSE_HOST` (default: `https://cloud.langfuse.com` or `http://localhost:3000` for self-hosted)
- [ ] Set up Langfuse project for trading system
- [ ] Configure tracing for all LLM calls:
  - [ ] Data Agent LLM calls
  - [ ] Risk Manager LLM calls (if any)
  - [ ] Validator LLM calls
  - [ ] Meta-Agent LLM calls
  - [ ] Ops Portal chat LLM calls
- [ ] Test tracing works

#### 9.3.2 Langfuse Trace Integration
- [ ] Instrument all agent LLM calls with Langfuse
- [ ] Add trace metadata:
  - [ ] Cycle ID
  - [ ] Symbol
  - [ ] Agent type
  - [ ] Signal ID
- [ ] Link traces to database events (via trace IDs)
- [ ] Configure trace sampling (100% for production)
- [ ] Test trace visibility in Langfuse UI

#### 9.3.3 Prompt Visibility
- [ ] All system prompts visible in Langfuse
- [ ] All user prompts visible in Langfuse
- [ ] Show intermediate reasoning steps
- [ ] Display token usage and latency per call
- [ ] Link traces to Grafana dashboards (via deep links)

### 9.4 Operations Portal (REQUIRED)

**Status**: REQUIRED | FastAPI/Next.js portal with chat interface and deep links

#### 9.4.1 Portal Setup
- [ ] Create `ops-portal/` directory
- [ ] Set up Next.js frontend (`ops-portal/frontend/`)
- [ ] Set up FastAPI backend (`ops-portal/backend/`)
- [ ] Add to `docker-compose.yml`
- [ ] Configure authentication (API key or OAuth)
- [ ] Set up API client to trading engine

#### 9.4.2 Chat Interface
- [ ] Create chat UI component (Next.js)
- [ ] Implement LLM-powered chat backend (FastAPI)
- [ ] Add tools for chat:
  - [ ] SQL Executor (natural language → SQL)
  - [ ] Portfolio queries
  - [ ] Trade history queries
  - [ ] Performance analytics
  - [ ] Control commands (pause/resume)
- [ ] Integrate Langfuse tracing for chat LLM calls
- [ ] Display chat history with trace links
- [ ] Test chat interface

#### 9.4.3 Deep Links to Grafana
- [ ] Generate deep links to Grafana panels:
  - [ ] Portfolio dashboard links
  - [ ] Performance dashboard links
  - [ ] Specific trade detail links
  - [ ] Log panel links
- [ ] Generate deep links to Langfuse:
  - [ ] Trace links for LLM calls
  - [ ] Prompt visibility links
- [ ] Embed links in chat responses
- [ ] Add "View in Grafana" buttons in portal
- [ ] Add "View Trace" buttons linking to Langfuse
- [ ] Test deep link navigation

#### 9.4.4 Portal Features
- [ ] Command line interface (chat input)
- [ ] Recent queries history
- [ ] Quick actions panel:
  - [ ] System status
  - [ ] Recent trades
  - [ ] Portfolio summary
  - [ ] Circuit breaker status
- [ ] Navigation to Grafana dashboards
- [ ] Navigation to Langfuse traces
- [ ] Responsive design (mobile-friendly)
- [ ] Test all portal features

---

## Phase 10: Evaluation Framework

**Note**: Langfuse integration is now part of Phase 9.3. This phase focuses on eval datasets and testing.

### 10.1 Langfuse Eval Integration
- [ ] Use Langfuse for eval tracking (configured in Phase 9.3)
- [ ] Set up eval projects in Langfuse
- [ ] Link evals to traces
- [ ] Test eval tracking

### 10.2 Eval Datasets
- [ ] Create `evals/datasets/` directory
- [ ] Create `evals/datasets/signal_quality.jsonl` (20-30 scenarios)
- [ ] Create `evals/datasets/risk_management.jsonl`
- [ ] Create `evals/datasets/validator_patterns.jsonl`
- [ ] Label: "good trade" vs "bad trade"

### 10.3 Eval Tests
- [ ] Create `evals/test_signal_quality.py`
- [ ] Create `evals/test_risk_management.py`
- [ ] Create `evals/test_validator_patterns.py`
- [ ] Create `evals/test_end_to_end.py` (6 months backtest)
- [ ] Run all evals, establish baseline

### 10.4 LLM Bakeoff
- [ ] Run every signal through Sonnet/Opus/Gemini in parallel
- [ ] Track win rate per provider
- [ ] Track Sharpe ratio per provider
- [ ] Track latency per provider
- [ ] Track cost per decision
- [ ] Human eval of 20 random decisions
- [ ] Choose best provider per agent

### 10.5 Nightly Evals
- [ ] Create cron job at 2am EST
- [ ] Run eval suite on latest code
- [ ] Compare to baseline
- [ ] Alert if regression >10%
- [ ] Block deploy if regression detected

---

## Phase 11: Deployment & CI/CD

### 11.1 AWS EC2 Setup
- [ ] Provision EC2 instance
- [ ] Configure security groups
- [ ] Set up SSH access
- [ ] Install Docker and Docker Compose
- [ ] Configure environment variables

### 11.2 Docker Compose Production
- [ ] Create `docker-compose.prod.yml`
- [ ] Configure production settings
- [ ] Set up volumes for data persistence
- [ ] Configure health checks
- [ ] Test local production build

### 11.3 GitHub Actions CI/CD
- [ ] Create `.github/workflows/test.yml` (run tests on PR)
- [ ] Create `.github/workflows/deploy.yml` (deploy on merge to main)
- [ ] Configure secrets (API keys, DB passwords)
- [ ] Test CI/CD pipeline

### 11.4 Monitoring Setup
- [ ] Configure Prometheus scraping
- [ ] Set up Grafana dashboards (optional)
- [ ] Configure log aggregation
- [ ] Set up database backups
- [ ] Test monitoring

---

## Phase 12: Discovery Production Integration

### 12.1 Scheduled Discovery Job ✓
- [x] Create scheduled job (weekly on Sunday evening) - Background job implemented
- [x] Integrate into scheduler - Registered in background_jobs.py
- [x] Run all enabled pickers - SymbolDiscoveryService orchestrates all pickers
- [x] Update watchlist with results - Automatic watchlist updates
- [ ] Test scheduled execution - Unit tests complete, integration tests pending

### 12.2 Watchlist Manager
- [ ] Create `src/services/discovery/watchlist.py`
- [ ] Feed discovered symbols into trading cycles
- [ ] Merge with manual watchlist
- [ ] Enforce max watchlist size
- [ ] Test watchlist updates

### 12.3 Discovery Alerts
- [ ] Alert on picker divergence (metric says no, LLM says yes)
- [ ] Alert on watchlist changes
- [ ] Test alert triggers

### 12.4 Discovery Service ✓
- [x] Create `src/services/discovery/service.py` - SymbolDiscoveryService implemented
- [x] Orchestrate all pickers - Runs all enabled pickers in parallel
- [x] Coordinate with WatchlistManager - Updates watchlist automatically
- [ ] Expose API endpoints - Pending (can be added to cycles router)
- [x] Test full discovery flow - Unit tests complete

### 12.5 API Endpoints
- [ ] Create `src/api/routers/discovery.py`
- [ ] `GET /api/discovery/status` - Discovery status
- [ ] `POST /api/discovery/run` - Manual trigger
- [ ] `GET /api/discovery/suggestions` - Current suggestions
- [ ] `POST /api/discovery/watchlist/add` - Manual add
- [ ] `POST /api/discovery/watchlist/remove` - Manual remove
- [ ] Test all endpoints

---

## Phase 13: S&P 500 & Sentiment Integration

**Status**: PLANNED | Not yet started

### 13.1 S&P 500 Symbol Source
- [ ] Create `src/services/discovery/sources/sp500.py` (SP500SymbolSource)
- [ ] Fetch S&P 500 constituents (static list or API)
- [ ] Cache S&P 500 list (update quarterly)
- [ ] Integrate into AlpacaAssetSource or create separate source
- [ ] Add config option: `discovery.use_sp500_symbols` (default: True)
- [ ] Test symbol fetching

### 13.2 LLMPicker S&P 500 Access
- [ ] Modify LLMPicker to accept S&P 500 symbols as candidate pool
- [ ] Update `_build_user_prompt()` to indicate S&P 500 scope
- [ ] Add config option: `discovery.llm_picker_use_sp500` (default: True)
- [ ] Ensure pre-filtering works with S&P 500 symbols
- [ ] Test LLMPicker with S&P 500 symbols

### 13.3 Sentiment Data Models
- [ ] Add `NewsArticle` model to `src/database/models.py`
  - Fields: id, symbol, headline, content, source, url, published_at, sentiment_score, sentiment_label, created_at
  - **Note**: Use existing `SymbolContextEmbedding` model for vector embeddings of news articles
- [ ] Add `SentimentSnapshot` model (aggregated sentiment per symbol per day)
  - Fields: id, symbol, date, avg_sentiment, article_count, sources, created_at
- [ ] Create Alembic migration for sentiment tables
- [ ] Add indexes on symbol, date, published_at, url (for deduplication)
- [ ] **Use pgvector**: Store news article embeddings in `SymbolContextEmbedding` table
  - `context_type="news"` for news articles
  - `context_text` = headline + content summary
  - `source_url` = article URL
  - `timestamp` = published_at

### 13.4 Sentiment Data Providers
- [ ] Create `src/services/sentiment/__init__.py`
- [ ] Create `src/services/sentiment/providers/__init__.py`
- [ ] Create `src/services/sentiment/providers/base.py` (SentimentProvider ABC)
- [ ] Create `src/services/sentiment/providers/newsapi.py` (NewsAPI integration)
- [ ] Create `src/services/sentiment/providers/alpaca_news.py` (Alpaca News API)
- [ ] Create `src/services/sentiment/providers/alpha_vantage.py` (Alpha Vantage News)
- [ ] Add retry logic and error handling
- [ ] Add rate limiting
- [ ] Test each provider

### 13.5 Sentiment Analysis Service
- [ ] Create `src/services/sentiment/service.py` (SentimentService)
- [ ] Implement sentiment scoring (using LLM or pre-trained model)
- [ ] Generate embeddings for news articles using `EmbeddingService`
- [ ] Store embeddings in `SymbolContextEmbedding` table (context_type="news")
- [ ] Aggregate sentiment by symbol and date
- [ ] Store in `NewsArticle` and `SentimentSnapshot` tables
- [ ] Add caching layer (avoid re-fetching same articles by URL)
- [ ] Add deduplication (check URL before inserting)
- [ ] Implement similarity search: `find_similar_news(symbol, query_text, limit)`
- [ ] Test sentiment extraction, embedding generation, and storage

### 13.6 Sentiment Background Jobs
- [ ] Create `src/scheduler/sentiment_jobs.py`
- [ ] Daily job: Fetch news for watchlist symbols (runs at market open)
- [ ] Hourly job: Update sentiment snapshots (runs during market hours)
- [ ] Register jobs in `src/scheduler/jobs.py`
- [ ] Add config: `sentiment.fetch_interval_hours`, `sentiment.snapshot_interval_hours`
- [ ] Test scheduled sentiment collection

### 13.7 Picker Sentiment Integration
- [ ] Update `LLMPicker._build_user_prompt()` to include sentiment data
  - Add "RECENT SENTIMENT" section with avg sentiment, article count, key headlines
  - **Use vector similarity**: Query `SymbolContextEmbedding` for similar news patterns
  - Include similar historical news events that led to positive/negative outcomes
- [ ] Update `FuzzyPicker` to use sentiment in scoring (optional weight)
  - Add `sentiment_weight` parameter (default: 0.1)
  - Positive sentiment boosts score, negative reduces
  - **Optional**: Use vector similarity to find similar news patterns
- [ ] Update `MetricPicker` to filter by sentiment (optional)
  - Add `min_sentiment_threshold` parameter (default: None, disabled)
- [ ] Add sentiment context to `SymbolDiscoveryService.run_discovery_cycle()`
  - Query `SentimentSnapshot` for candidate symbols
  - Query `SymbolContextEmbedding` for similar news patterns (vector search)
  - Pass sentiment data in context to pickers
- [ ] Test pickers with sentiment data and vector similarity

### 13.8 Sentiment API Endpoints
- [ ] Create `src/api/routers/sentiment.py`
- [ ] `GET /api/sentiment/{symbol}` - Get sentiment for symbol (latest snapshot)
- [ ] `GET /api/sentiment/{symbol}/articles` - Get news articles for symbol
- [ ] `GET /api/sentiment/snapshot` - Get daily sentiment snapshot for all symbols
- [ ] `GET /api/sentiment/{symbol}/history` - Get sentiment history (last N days)
- [ ] Test all endpoints

---

## Phase 14: Paper Trading

### 14.1 Pre-Flight Checks
- [ ] Verify end-to-end cycle works
- [ ] Test database writes (trades, positions, events)
- [ ] Verify paper broker works
- [ ] Test scheduler (manual trigger first)
- [ ] Check all API endpoints
- [ ] Verify logging and metrics

### 14.2 Environment Setup
- [ ] Configure `.env` file
- [ ] Set LLM API keys
- [ ] Configure database connection
- [ ] Set watchlist symbols
- [ ] Test all connections

### 14.3 Deployment
- [ ] Deploy to AWS (or run locally)
- [ ] Start all services
- [ ] Verify scheduler starts
- [ ] Monitor initial cycles
- [ ] Fix any issues

### 14.4 Monitoring (2 weeks)
- [ ] Daily log reviews
- [ ] Check portfolio state daily
- [ ] Monitor for errors/crashes
- [ ] Track performance metrics
- [ ] Iterate on prompts based on results
- [ ] Run nightly evals

### 14.5 Success Criteria
- [ ] No system crashes
- [ ] Sharpe ratio > 0
- [ ] Win rate > 50%
- [ ] No losses > 20%
- [ ] All circuit breakers tested
- [ ] Telegram bot query interface functional
- [ ] Grafana dashboards functional with Langfuse traceability
- [ ] Evals passing

---

## Phase 15: Production Launch

### 15.1 Final Checks
- [ ] All safety checks pass
- [ ] Circuit breakers tested
- [ ] DB transactions rollback on error
- [ ] Hard constraints cannot be overridden
- [ ] All LLM calls have retry logic
- [ ] End-to-end pipeline executes
- [ ] Telegram bot query interface functional
- [ ] Grafana dashboards functional with Langfuse traceability
- [ ] Evals run and pass
- [ ] Paper trading completed (2 weeks)
- [ ] Ready to risk £500

### 15.2 Production Switch
- [ ] Switch Alpaca to live mode
- [ ] Deploy with £500 initial capital
- [ ] Monitor obsessively (first week)
- [ ] Daily P&L reviews
- [ ] Circuit breaker monitoring
- [ ] Alert system verification

---

## Dependencies

- Phase 1 (pgvector) → Phase 4 (Vector Integration)
- Phase 1 (pgvector) → Phase 9 (Ops Portal RAG)
- Phase 2 (Discovery Foundation) → Phase 3 (Pickers)
- Phase 3 (Pickers) → Phase 4 (Vector Integration)
- Phase 3 (Pickers) → Phase 8 (Discovery Analysis)
- Phase 4 (Vector Integration) → Phase 9 (Ops Portal RAG)
- Phase 5 (Event Monitor) → Can run independently
- Phase 6 (Auto-Resume) → Can run independently
- Phase 7 (Alerts) → Phase 9 (Telegram Bot Queries) → Phase 11 (Deployment)
- Phase 8 (Discovery Analysis) → Phase 12 (Discovery Production)
- Phase 9.1 (Telegram Bot Queries) → Phase 9.4 (Ops Portal) - Both provide query interfaces
- Phase 9 (Grafana) → Phase 11 (Deployment) - Prometheus required
- Phase 9 (Langfuse) → Phase 9 (Ops Portal) - Tracing required
- Phase 9 (Ops Portal) → Phase 1 (pgvector) + Phase 4 (Vector Integration) + Phase 9 (Langfuse)
- Phase 10 (Evals) → Phase 9 (Langfuse) - Uses Langfuse for eval tracking
- Phase 10 (Evals) → Can run in parallel
- Phase 11 (Deployment) → Phase 14 (Paper Trading)
- Phase 12 (Discovery Production) → Phase 13 (S&P 500 & Sentiment)
- Phase 13 (S&P 500 & Sentiment) → Phase 14 (Paper Trading)
- Phase 14 (Paper Trading) → Phase 15 (Production)

