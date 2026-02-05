# Local Deployment Guide

## Prerequisites

1. **PostgreSQL with pgvector** (via Docker/Podman)
2. **Python 3.11+** with `uv` package manager
3. **`.env` file** with required configuration

## Quick Start

### 1. Start Database

```bash
# Using podman-compose (or docker-compose)
cd trading-engine
podman compose up -d postgres

# Wait for DB to be ready (check logs)
podman compose logs postgres
```

### 2. Run Migrations

```bash
cd trading-engine
make migrate-exec
```

### 3. Create `.env` File

Create `trading-engine/.env` with at minimum:

```bash
# Database (REQUIRED)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=your_password_here

# LLM (REQUIRED for agents to work)
# At least one of these:
LLM_ANTHROPIC_API_KEY=sk-ant-...
# OR
LLM_OPENAI_API_KEY=sk-...

# Alpaca (OPTIONAL - falls back to yfinance if not set)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
# Defaults to paper trading: ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 4. Install Dependencies

```bash
cd trading-engine
uv sync
# If using embeddings (BGE-small is default, local, free):
uv sync --extra embedding
```

### 5. Start the Application

```bash
cd trading-engine
make dev
# Or manually:
# uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

---

## Required vs Optional Configuration

### ✅ REQUIRED

**Database** (must have PostgreSQL running):
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=your_password
```

**LLM API Keys** (at least one - agents won't work without this):
```bash
# Primary provider (OpenAI - DEFAULT)
LLM_OPENAI_API_KEY=sk-...

# Fallback provider (Anthropic)
LLM_ANTHROPIC_API_KEY=sk-ant-...

# Optional providers for experimentation:
LLM_GOOGLE_API_KEY=...  # For Gemini
LLM_DEEPSEEK_API_KEY=...  # For DeepSeek
```

**Provider Configuration**:
- **Default provider**: OpenAI (can be changed via `LLM_DEFAULT_PROVIDER`)
- **Fallback provider**: Anthropic (used if primary fails)
- **Per-agent/provider selection**: Configure in `.env`:
  ```bash
  LLM_DATA_AGENT_PROVIDER=openai  # or "anthropic", "google", "deepseek", "auto"
  LLM_RISK_AGENT_PROVIDER=openai
  LLM_VALIDATOR_PROVIDER=openai
  LLM_META_AGENT_PROVIDER=openai
  LLM_LLM_PICKER_PROVIDER=openai  # For LLMPicker in discovery
  ```

**Default Models** (OpenAI):
- Data/Risk agents: `gpt-4o-mini` (fast, cost-effective)
- Validator/Meta-Agent: `gpt-4o` (higher quality)
- LLM Picker: `gpt-4o-mini`

**Note**: The system automatically falls back to Anthropic if OpenAI fails. You can experiment with different providers per component to compare performance.

### ⚠️ OPTIONAL (but recommended)

**Alpaca API** (for real market data and paper trading):
```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

**Without Alpaca**:
- Market data falls back to `yfinance` (free, but rate-limited)
- Trade execution uses `PaperBroker` (local simulation, no real orders)

**Telegram Alerts** (for notifications):
```bash
ALERT_TELEGRAM_BOT_TOKEN=your_bot_token
ALERT_TELEGRAM_CHAT_ID=your_chat_id
```

**Langfuse** (OPTIONAL - for LLM tracing/debugging):
```bash
# For self-hosted (default - runs in Docker):
LANGFUSE_HOST=http://localhost:3010
LANGFUSE_PROJECT=trading-system
# public_key and secret_key not needed for self-hosted

# For cloud version:
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROJECT=trading-system
```

**Note**: Langfuse is **open source and free**. 
- **Self-hosted** (default): Runs in Docker at `http://localhost:3010`, no API keys needed
- **Cloud version**: Requires API keys from https://cloud.langfuse.com
- **Completely optional** - the system works fine without it

**Grafana** (OPTIONAL - for dashboards/metrics):
```bash
GRAFANA_ADMIN_USER=admin          # Default: admin
GRAFANA_ADMIN_PASSWORD=admin     # Default: admin (CHANGE IN PRODUCTION!)
```

**Note**: Grafana runs in Docker and is accessible at `http://localhost:3045`. Default credentials are `admin/admin` - **change these in production!**

**Prometheus** (no env vars needed):
- Runs automatically in Docker
- Accessible at `http://localhost:9045`
- Scrapes metrics from trading engine at `http://localhost:8000/metrics`

**Langfuse** (self-hosted - no env vars needed by default):
- Runs automatically in Docker
- Accessible at `http://localhost:3010`
- Uses the same PostgreSQL database as the trading engine
- Optional env vars for security:
  ```bash
  LANGFUSE_NEXTAUTH_SECRET=your-secret-key  # Default: your-secret-key-change-in-production
  LANGFUSE_SALT=your-salt                   # Default: your-salt-change-in-production
  ```

### 🔧 OPTIONAL (not yet implemented)

**Langfuse** (for LLM tracing - Phase 9.3, not yet built):
```bash
LANGFUSE_PUBLIC_KEY=your_public_key  # For cloud version
LANGFUSE_SECRET_KEY=your_secret_key  # For cloud version
LANGFUSE_HOST=https://cloud.langfuse.com  # or http://localhost:3000 for self-hosted
LANGFUSE_PROJECT=trading-system
```

**Note**: Langfuse is an open source LLM observability platform. You can use the free cloud version or self-host it. It's completely optional and not yet integrated into the codebase. You can skip this entirely.

### 🔧 OPTIONAL (advanced)

**Vector DB** (defaults to Chroma local):
```bash
VECTOR_DB_PROVIDER=chroma  # or qdrant
VECTOR_DB_CHROMA_PATH=./chroma_db
```

**Embeddings** (defaults to BGE-small, local, free):
```bash
EMBEDDING_PROVIDER=bge-small  # local, no API key needed
# OR
EMBEDDING_PROVIDER=openai
EMBEDDING_OPENAI_API_KEY=sk-...  # if using OpenAI embeddings
```

**Trading Settings** (defaults are fine for testing):
```bash
TRADING_INITIAL_CAPITAL=500.0
TRADING_MAX_POSITION_PCT=0.20
TRADING_MAX_SECTOR_PCT=0.30
TRADING_WATCHLIST_SYMBOLS=SPY,QQQ,AAPL,MSFT,GOOGL,TSLA,NVDA,AMZN
```

---

## Testing the Deployment

### 1. Health Check

```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

### 2. Check Database Connection

```bash
curl http://localhost:8000/api/portfolio
# Should return portfolio state (may be empty initially)
```

### 3. Run a Trading Cycle (Dry Run)

```bash
curl -X POST http://localhost:8000/api/cycles/run \
  -H "Content-Type: application/json" \
  -d '{
    "type": "scheduled",
    "symbols": ["AAPL"],
    "execute": false
  }'
```

**Expected behavior**:
- If LLM keys are set: Agents will generate signals, risk assessments, validations, and decisions
- If LLM keys are missing: You'll see errors in logs about missing API keys

### 4. Check Logs

```bash
# Application logs (if LOG_FORMAT=console)
# Or check logs/trading.log (if LOG_FORMAT=json)
tail -f logs/trading.log
```

---

## What Works Without API Keys?

### ✅ Works Without Any API Keys:
- Database operations
- API endpoints (health, portfolio, trades, etc.)
- Event logging
- Circuit breaker logic
- Background jobs (scheduler, embeddings, discovery)
- Vector embeddings (BGE-small is local/free)

### ❌ Requires LLM API Keys:
- **Trading agents** (DataAgent, RiskManager, Validator, MetaAgent)
- **LLMPicker** (symbol discovery)
- **Trading cycles** (`/api/cycles/run`)

### ⚠️ Works But Limited Without Alpaca:
- **Market data**: Falls back to yfinance (slower, rate-limited)
- **Trade execution**: Uses PaperBroker (local simulation, no real orders)

---

## Troubleshooting

### "No module named 'pgvector'"
```bash
# Make sure you've installed dependencies
cd trading-engine
uv sync
```

### "Connection refused" (database)
```bash
# Check if PostgreSQL is running
podman compose ps postgres

# Check logs
podman compose logs postgres

# Verify DB credentials in .env match docker-compose.yml
```

### "LLM API key not found"
- Check `.env` file exists in `trading-engine/` directory
- Verify `LLM_ANTHROPIC_API_KEY` or `LLM_OPENAI_API_KEY` is set
- Restart the application after adding keys

### "Alpaca API error"
- If you don't have Alpaca keys, the system will fall back to yfinance
- This is fine for testing, but you'll see warnings in logs

### "Embedding model download failed"
- BGE-small downloads automatically on first use (~130MB)
- Make sure you have internet connection
- Or set `EMBEDDING_PROVIDER=openai` and provide `LLM_OPENAI_API_KEY`

---

## Next Steps

Once deployed and tested:

1. **Run a full trading cycle** (with LLM keys)
2. **Check the event log**: `GET /api/events`
3. **Monitor portfolio**: `GET /api/portfolio`
4. **Review agent decisions**: Check logs for reasoning
5. **Test circuit breakers**: Trigger manually via API
6. **Set up Telegram alerts**: For notifications
7. **Configure Langfuse** (optional): For LLM tracing/debugging (open source, free)

---

## Production Checklist

Before going live with real money:

- [ ] All tests pass: `make test`
- [ ] Database migrations applied: `make migrate-exec`
- [ ] LLM API keys configured and tested
- [ ] Alpaca API keys configured (paper trading first!)
- [ ] Circuit breakers tested
- [ ] Telegram alerts working
- [ ] Logging configured and monitored
- [ ] Initial capital set correctly
- [ ] Watchlist symbols configured
- [ ] Scheduler jobs registered (check logs on startup)

