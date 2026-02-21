# Njord Trading Engine

Autonomous multi-agent trading platform that discovers symbols, runs a LangGraph-based decision workflow, executes paper/live trades via Alpaca, and logs everything for auditability. This repo includes the FastAPI backend, scheduler jobs, discovery services, observability stack, and operator tooling.

## Features

- **Discovery pipeline**: Metric/Fuzzy/LLM pickers run in parallel, ensemble their scores, and keep the watchlist fresh.
- **Four-agent trading workflow**: Data → Risk → Validator → Meta agents coordinate through LangGraph with strict risk controls and circuit breakers.
- **Execution + persistence**: Dual-writes trades/positions/events in a single transaction and supports paper or live Alpaca brokers.
- **Observability**: Prometheus metrics, Grafana dashboards, Langfuse LLM tracing, structured logs, and a Telegram bot for mobile status checks.

## Repository Layout

```
README.md                 ← project overview (this file)
docker-compose.yml        ← Postgres, Prometheus, Grafana, (optional) Langfuse
trading-engine/
  src/                    ← FastAPI app, agents, services, workflows
  tests/                  ← pytest suite (unit/integration)
  grafana/                ← datasource + dashboard provisioning
  prometheus.yml          ← scrape config for trading-engine /metrics endpoint
  Makefile                ← dev shortcuts (migrations, tests, lint, etc.)
```

## Prerequisites

- Podman (or Docker) for Postgres/Prometheus/Grafana/Langfuse
- Python 3.11+ with [uv](https://github.com/astral-sh/uv)
- Alpaca paper credentials, Telegram bot token/chat ID, LLM API keys (see `.env.example`)

## Quick Start

### 1. Clone & install deps

```bash
git clone <repo-url>
cd njord-commerce/trading-engine
uv sync --extra dev
```

### 2. Configure environment

Copy `.env.example` to `.env` and set:

```
DB_HOST=localhost
DB_PORT=5556
DB_USER=postgres
DB_PASSWORD=postgres
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALERT_TELEGRAM_BOT_TOKEN=...
ALERT_TELEGRAM_CHAT_ID=...
LLM_OPENAI_API_KEY=...
LLM_ANTHROPIC_API_KEY=...
LANGFUSE_PUBLIC_KEY=... (optional)
LANGFUSE_SECRET_KEY=...
```

### 3. Start infrastructure

From repo root:

```bash
podman compose up -d postgres prometheus grafana
```

This exposes Postgres on `localhost:5556`, Prometheus on `http://localhost:9045`, Grafana on `http://localhost:3045` (admin/admin).

### 4. Apply migrations

```bash
cd trading-engine
make migrate-exec
```

### 5. Run the API

```bash
cd trading-engine
make dev
```

FastAPI listens on `http://localhost:8000` (see `/docs` for Swagger UI). `/metrics` serves Prometheus metrics.

## Seeding Discovery & Running Cycles

1. **Populate watchlist automatically**
   ```bash
   curl -X POST http://localhost:8000/api/discovery/run \
     -H "Content-Type: application/json" \
     -d '{"update_watchlist":true}'
   ```
   This runs Metric/Fuzzy/LLM pickers, ensembles results, and updates the watchlist.

2. **Trigger a manual trading cycle**
   ```bash
   curl -X POST http://localhost:8000/api/cycles/run \
     -H "Content-Type: application/json" \
     -d '{"symbols":["AAPL","MSFT"],"cycle_type":"manual"}'
   ```
   Each cycle flows through the four agents, executes via the paper broker, and persists trades/events.

3. **Check results**
   - `GET /api/trades?limit=10` – recent paper trades
   - `GET /api/portfolio/positions` – live positions snapshot
   - `GET /api/events?limit=20` – agent reasoning/audit log

## Observability Toolkit

- **Prometheus** scrapes `http://localhost:8000/metrics` (configured via `prometheus.yml`).
- **Grafana** auto-loads Prometheus + Postgres datasources and a starter Portfolio dashboard (`grafana/dashboards/portfolio.json`). Visit `http://localhost:3045` to customize.
- **Langfuse** (optional) traces every LLM call when `LANGFUSE_*` env vars are set. Access the UI via your Langfuse deployment to inspect prompts/results.
- **Telegram bot**: expose `/api/system/telegram/webhook` (ngrok, etc.) and DM `/status`, `/portfolio`, `/metrics`, `/logs`, `/trades`, `/query` for real-time insights.

## Background Jobs

Run scheduled jobs (discovery, embeddings, event monitor, daily alerts) in a separate terminal:

```bash
cd trading-engine
uv run python -m src.scheduler.background_jobs
```

This registers cron-like APScheduler jobs based on `SCHEDULE_*` env settings and background services.

## Testing & Linting

```bash
cd trading-engine
make test        # full pytest suite
make test-fast   # skip slow integration tests
make lint        # ruff lint + format check
```

All Telegram/Langfuse calls are mocked in tests; no external traffic occurs.

## Troubleshooting

- **Collation warning** when running `psql`: safe to ignore locally; re-create DB if you need strict locale matching.
- **Circuit breaker halts trading**: check `system_state` table or `/api/system/status`; use `POST /api/system/circuit-breaker/resume` only after auto-resume conditions are met.
- **LLM errors**: `_call_llm` uses retries + Langfuse logging. Inspect logs (trace_id) and Langfuse traces for root causes.
- **Telegram webhook**: ensure `ALERT_TELEGRAM_CHAT_ID` matches your chat; unauthorized IDs are ignored.

## Further Reading

- `ENGINE_MANUAL.md` – operator-friendly walkthrough of the entire system
- `IMPLEMENTATION_PLAN.md` – phased roadmap & status
- `devlog.md` – detailed change log
- `api.md` – REST endpoint reference

Happy trading! 🎯
