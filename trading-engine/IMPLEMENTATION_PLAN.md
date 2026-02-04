# Implementation Plan

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

## Phase 6: Circuit Breaker Auto-Resume

### 6.1 Scheduler Integration
- [ ] Call `check_auto_resume()` before each trading cycle
- [ ] Integrate into `src/scheduler/jobs.py`
- [ ] Test auto-resume conditions:
  - Drawdown recovery < 15%
  - Win streak of 3+
  - Sharpe ratio > 0.3 for 7 days

### 6.2 Manual Approval
- [ ] Require manual approval via API before resuming
- [ ] Add endpoint `POST /api/system/circuit-breaker/resume`
- [ ] Test approval flow

---

## Phase 7: Alert System

### 7.1 Telegram Integration
- [ ] Create `src/services/alerts/__init__.py`
- [ ] Create `src/services/alerts/telegram.py`
- [ ] Implement Telegram bot client
- [ ] Create alert templates
- [ ] Test message delivery

### 7.2 Email Integration
- [ ] Create `src/services/alerts/email.py`
- [ ] Integrate AWS SES
- [ ] Create email templates
- [ ] Test email delivery

### 7.3 Alert Service
- [ ] Create `src/services/alerts/service.py`
- [ ] Implement alert routing (Telegram vs Email)
- [ ] Add alert triggers:
  - Circuit breaker activation
  - Large position changes
  - Daily P&L summary
  - System errors
- [ ] Test all alert types

---

## Phase 8: Discovery Analysis

### 8.1 Forward Return Tracking
- [ ] Add forward return fields to `PickerSuggestion` model
- [ ] Create background job to calculate 1d/5d/20d returns
- [ ] Update suggestions with actual returns
- [ ] Test return calculation

### 8.2 Performance API
- [ ] Create `src/api/routers/discovery.py`
- [ ] Implement `GET /api/discovery/performance` endpoint
- [ ] Compare picker performance
- [ ] Return win rates, average returns
- [ ] Test API endpoint

### 8.3 Paper Trade Tracker
- [ ] Track hypothetical trades from suggestions
- [ ] Compare to actual watchlist trades
- [ ] Calculate A/B test metrics

---

## Phase 9: Chat UI (Chainlit)

### 9.1 Setup
- [ ] Create `chat-ui/` directory
- [ ] Create `chat-ui/Dockerfile`
- [ ] Create `chat-ui/pyproject.toml`
- [ ] Create `chat-ui/src/app.py` (Chainlit app)
- [ ] Add to `docker-compose.yml`

### 9.2 Tool 1: SQL Executor
- [ ] Create `chat-ui/src/tools/sql_executor.py`
- [ ] Convert natural language → SQL
- [ ] Execute on PostgreSQL
- [ ] Return formatted results
- [ ] Test with various queries

### 9.3 Tool 2: Event Log RAG
- [ ] Create `chat-ui/src/tools/event_rag.py`
- [ ] Vector search over event log
- [ ] Use pgvector for similarity search
- [ ] Return relevant events with reasoning
- [ ] Test RAG queries

### 9.4 Tool 3: Portfolio State
- [ ] Create `chat-ui/src/tools/portfolio.py`
- [ ] Get current holdings, cash, exposure
- [ ] Real-time queries
- [ ] Test portfolio queries

### 9.5 Tool 4: Performance Analytics
- [ ] Create `chat-ui/src/tools/analytics.py`
- [ ] Calculate Sharpe ratio
- [ ] Calculate win rate
- [ ] Calculate drawdown
- [ ] Calculate alpha vs deposits
- [ ] Time-based: today, week, month, all-time
- [ ] Test analytics queries

### 9.6 Tool 5: Control Commands
- [ ] Create `chat-ui/src/tools/control.py`
- [ ] Implement pause/resume trading
- [ ] Implement run_cycle trigger
- [ ] Implement trigger_eval
- [ ] Require confirmation for destructive actions
- [ ] Test control commands

### 9.7 Tool 6: Pattern Analysis
- [ ] Create `chat-ui/src/tools/patterns.py`
- [ ] Which validator concerns predicted losses?
- [ ] Win rate by confidence bucket
- [ ] Optimal holding periods
- [ ] Test pattern queries

### 9.8 Agent Integration
- [ ] Create `chat-ui/src/agent.py`
- [ ] Integrate all 6 tools
- [ ] Configure LLM agent
- [ ] Message history persistence
- [ ] Simple auth (API key)
- [ ] Test full agent flow

---

## Phase 10: Evaluation Framework

### 10.1 LangSmith Integration
- [ ] Add LangSmith dependencies to `pyproject.toml`
- [ ] Configure LangSmith API key
- [ ] Set up tracing for all LLM calls
- [ ] Test tracing works

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
- [ ] Chat UI functional
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
- [ ] Chat UI functional
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
- Phase 1 (pgvector) → Phase 9 (Chat UI RAG)
- Phase 2 (Discovery Foundation) → Phase 3 (Pickers)
- Phase 3 (Pickers) → Phase 4 (Vector Integration)
- Phase 3 (Pickers) → Phase 8 (Discovery Analysis)
- Phase 4 (Vector Integration) → Phase 9 (Chat UI RAG)
- Phase 5 (Event Monitor) → Can run independently
- Phase 6 (Auto-Resume) → Can run independently
- Phase 7 (Alerts) → Phase 11 (Deployment)
- Phase 8 (Discovery Analysis) → Phase 12 (Discovery Production)
- Phase 9 (Chat UI) → Phase 1 (pgvector) + Phase 4 (Vector Integration)
- Phase 10 (Evals) → Can run in parallel
- Phase 11 (Deployment) → Phase 14 (Paper Trading)
- Phase 12 (Discovery Production) → Phase 13 (S&P 500 & Sentiment)
- Phase 13 (S&P 500 & Sentiment) → Phase 14 (Paper Trading)
- Phase 14 (Paper Trading) → Phase 15 (Production)

