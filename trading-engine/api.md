# Trading Engine API Reference

Base URL: `http://localhost:8000`

## Overview

The Trading Engine API provides endpoints for:
- Running multi-agent trading cycles
- Managing system state and circuit breaker
- Viewing portfolio, positions, and trades
- Accessing market data
- Monitoring via Prometheus metrics

## Authentication

Currently no authentication is required. In production, add API key or OAuth2.

---

## Endpoints

### Health

#### `GET /health`

Health check endpoint.

**Response**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

---

### System

#### `GET /api/system/circuit-breaker`

Get circuit breaker status.

**Response**
```json
{
  "active": false,
  "reason": null,
  "trading_enabled": true,
  "current_drawdown_pct": 5.2
}
```

#### `POST /api/system/circuit-breaker/reset`

Manually reset the circuit breaker.

**Request**
```json
{
  "reason": "Market conditions improved"
}
```

**Response**
```json
{
  "success": true,
  "message": "Circuit breaker has been reset. Trading is now enabled."
}
```

#### `POST /api/system/circuit-breaker/evaluate`

Force circuit breaker evaluation. Returns updated status.

**Response**
```json
{
  "active": true,
  "reason": "Drawdown 22.5% exceeds 20% threshold",
  "trading_enabled": true,
  "current_drawdown_pct": 22.5
}
```

#### `POST /api/system/trading/disable`

Emergency kill switch - disable all trading.

**Response**
```json
{
  "success": true,
  "message": "Trading has been disabled."
}
```

#### `POST /api/system/trading/enable`

Re-enable trading.

**Response**
```json
{
  "success": true,
  "message": "Trading has been enabled. Note: Circuit breaker is still active."
}
```

---

### Trading Cycles

#### `POST /cycles/run`

Trigger a trading cycle. Runs the full agent pipeline.

**Headers**
```
X-Request-ID: <uuid>  # Optional - for distributed tracing
```

**Request**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "cycle_type": "scheduled",
  "trigger_symbol": null,
  "execute": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `symbols` | `string[]` | watchlist | Symbols to analyze |
| `cycle_type` | `string` | `"scheduled"` | `"scheduled"` or `"event"` |
| `trigger_symbol` | `string` | `null` | For event cycles, the triggering symbol |
| `execute` | `boolean` | `false` | If true, execute EXECUTE decisions via broker |

**Response**
```json
{
  "cycle_id": "550e8400-e29b-41d4-a716-446655440000",
  "trace_id": "abc-123-def-456",
  "cycle_type": "scheduled",
  "started_at": "2024-01-15T11:00:00Z",
  "completed_at": "2024-01-15T11:00:45Z",
  "duration_seconds": 45.2,
  "symbols": ["AAPL", "MSFT", "GOOGL"],

  "signals": [
    {
      "id": "signal-uuid",
      "symbol": "AAPL",
      "action": "BUY",
      "confidence": 0.85,
      "proposed_quantity": 10,
      "reasoning": "Strong momentum with RSI breakout...",
      "price": 185.50,
      "rsi_14": 65.2
    }
  ],

  "risk_assessments": [
    {
      "signal_id": "signal-uuid",
      "approved": true,
      "adjusted_quantity": 8,
      "risk_score": 0.35,
      "hard_constraint_violated": false,
      "hard_constraint_reason": null,
      "concerns": ["Slightly elevated sector exposure"],
      "reasoning": "Position size within limits..."
    }
  ],

  "validations": [
    {
      "signal_id": "signal-uuid",
      "approved": true,
      "concerns": [],
      "suggestions": ["Consider scaling in"],
      "repetition_detected": false,
      "sector_clustering_detected": false,
      "similar_setup_failures": 0,
      "reasoning": "No concerning patterns detected..."
    }
  ],

  "final_decisions": [
    {
      "signal_id": "signal-uuid",
      "decision": "EXECUTE",
      "final_quantity": 8,
      "confidence": 0.82,
      "reasoning": "All agents approve with minor adjustments..."
    }
  ],

  "execution_results": [
    {
      "signal_id": "signal-uuid",
      "success": true,
      "trade_id": "trade-uuid",
      "symbol": "AAPL",
      "action": "BUY",
      "quantity": 8,
      "requested_price": 185.50,
      "fill_price": 185.55,
      "slippage": 0.00027,
      "broker_order_id": "alpaca-order-123",
      "error": null
    }
  ],

  "total_signals": 3,
  "approved_by_risk": 2,
  "approved_by_validator": 2,
  "execute_decisions": 1,
  "executed_trades": 1,
  "errors": []
}
```

**Error Responses**

| Status | Reason |
|--------|--------|
| 400 | Circuit breaker active or validation error |
| 500 | Cycle execution failed |

---

### Portfolio

#### `GET /api/portfolio`

Get current portfolio state.

**Response**
```json
{
  "cash": 450.00,
  "total_value": 1250.00,
  "deployed_capital": 800.00,
  "peak_value": 1300.00,
  "drawdown_pct": 3.85
}
```

#### `GET /api/positions`

Get all current positions.

**Response**
```json
[
  {
    "symbol": "AAPL",
    "quantity": 5,
    "avg_cost": 180.00,
    "current_price": 185.50,
    "current_value": 927.50,
    "unrealized_pnl": 27.50,
    "unrealized_pnl_pct": 3.06,
    "sector": "Technology"
  }
]
```

#### `GET /api/positions/{symbol}`

Get position for a specific symbol.

**Response**
```json
{
  "symbol": "AAPL",
  "quantity": 5,
  "avg_cost": 180.00,
  "current_price": 185.50,
  "current_value": 927.50,
  "unrealized_pnl": 27.50,
  "unrealized_pnl_pct": 3.06,
  "sector": "Technology"
}
```

---

### Trades

#### `GET /api/trades`

Get trade history.

**Query Parameters**
| Param | Type | Description |
|-------|------|-------------|
| `symbol` | string | Filter by symbol |
| `limit` | int | Max results (default 100) |
| `offset` | int | Pagination offset |

**Response**
```json
[
  {
    "id": "trade-uuid",
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 8,
    "price": 185.55,
    "total_value": 1484.40,
    "status": "FILLED",
    "signal_confidence": 0.85,
    "risk_score": 0.35,
    "outcome": "WIN",
    "pnl": 45.00,
    "pnl_pct": 3.03,
    "created_at": "2024-01-15T11:00:45Z"
  }
]
```

#### `GET /api/trades/{trade_id}`

Get a specific trade by ID.

---

### Events

#### `GET /api/events`

Get event log (audit trail).

**Query Parameters**
| Param | Type | Description |
|-------|------|-------------|
| `event_type` | string | Filter by type |
| `aggregate_id` | string | Filter by cycle_id |
| `limit` | int | Max results |

**Response**
```json
[
  {
    "id": "event-uuid",
    "timestamp": "2024-01-15T11:00:30Z",
    "event_type": "SignalGenerated",
    "aggregate_id": "cycle-uuid",
    "data": {
      "symbol": "AAPL",
      "action": "BUY",
      "confidence": 0.85
    }
  }
]
```

#### `POST /api/events`

Create a manual event (for external integrations).

**Request**
```json
{
  "event_type": "ManualNote",
  "aggregate_id": "cycle-uuid",
  "data": {"note": "Market volatility observed"}
}
```

---

### Market Data

#### `GET /api/market/quote/{symbol}`

Get current quote for a symbol.

**Response**
```json
{
  "symbol": "AAPL",
  "price": 185.50,
  "bid": 185.48,
  "ask": 185.52,
  "volume": 45000000,
  "timestamp": "2024-01-15T11:00:00Z"
}
```

#### `GET /api/market/indicators/{symbol}`

Get technical indicators for a symbol.

**Response**
```json
{
  "symbol": "AAPL",
  "price": 185.50,
  "sma_20": 182.30,
  "sma_50": 178.50,
  "sma_200": 165.20,
  "rsi_14": 65.2,
  "volume_ratio": 1.2,
  "timestamp": "2024-01-15T11:00:00Z"
}
```

---

### Capital

#### `GET /api/capital`

Get capital events (deposits, withdrawals, realized P&L).

**Response**
```json
[
  {
    "id": "event-uuid",
    "event_type": "DEPOSIT",
    "amount": 500.00,
    "balance_after": 500.00,
    "description": "Initial deposit",
    "trade_id": null,
    "created_at": "2024-01-01T00:00:00Z"
  },
  {
    "id": "event-uuid",
    "event_type": "REALIZED_PROFIT",
    "amount": 45.00,
    "balance_after": 545.00,
    "description": "Closed 5 shares of AAPL",
    "trade_id": "trade-uuid",
    "created_at": "2024-01-15T14:30:00Z"
  }
]
```

#### `POST /api/capital`

Record a capital event (deposit/withdrawal).

**Request**
```json
{
  "event_type": "DEPOSIT",
  "amount": 100.00,
  "description": "Additional funding"
}
```

#### `GET /api/capital/summary`

Get capital summary.

**Response**
```json
{
  "total_deposits": 500.00,
  "total_withdrawals": 0.00,
  "total_realized_pnl": 45.00,
  "total_fees": 2.50,
  "net_capital_flow": 500.00
}
```

---

### Strategies

#### `GET /api/strategies`

List all strategies.

**Response**
```json
[
  {
    "id": "strategy-uuid",
    "name": "momentum",
    "status": "ACTIVE",
    "allocation": 0.6,
    "capital": 300.00,
    "sharpe_30d": 1.2,
    "win_rate_30d": 0.65,
    "total_trades": 25,
    "total_pnl": 150.00
  }
]
```

#### `POST /api/strategies`

Create a new strategy.

**Request**
```json
{
  "name": "mean_reversion",
  "allocation": 0.4,
  "description": "Buy oversold, sell overbought"
}
```

#### `GET /api/strategies/{strategy_id}`

Get a specific strategy.

#### `PATCH /api/strategies/{strategy_id}`

Update a strategy.

**Request**
```json
{
  "status": "PAUSED",
  "allocation": 0.3
}
```

#### `DELETE /api/strategies/{strategy_id}`

Delete a strategy.

---

### Scheduler

#### `GET /api/scheduler/status`

Get scheduler status and all scheduled jobs.

**Response**
```json
{
  "running": true,
  "job_count": 2,
  "jobs": [
    {
      "id": "trading_cycle_1100",
      "name": "Trading cycle at 11:00 ET",
      "next_run": "2024-01-16T11:00:00-05:00",
      "trigger": "cron[hour='11', minute='0', day_of_week='mon-fri']"
    },
    {
      "id": "trading_cycle_1430",
      "name": "Trading cycle at 14:30 ET",
      "next_run": "2024-01-15T14:30:00-05:00",
      "trigger": "cron[hour='14', minute='30', day_of_week='mon-fri']"
    }
  ],
  "market_open": true,
  "trading_day": true,
  "next_market_open": "2024-01-16T09:30:00-05:00"
}
```

#### `GET /api/scheduler/jobs`

List all scheduled jobs.

**Response**
```json
[
  {
    "id": "trading_cycle_1100",
    "name": "Trading cycle at 11:00 ET",
    "next_run": "2024-01-16T11:00:00-05:00",
    "trigger": "cron[hour='11', minute='0', day_of_week='mon-fri']"
  }
]
```

#### `POST /api/scheduler/jobs/{job_id}/trigger`

Manually trigger a scheduled job to run immediately.

**Response**
```json
{
  "success": true,
  "message": "Job 'trading_cycle_1100' has been triggered. It will run momentarily."
}
```

#### `GET /api/scheduler/market-status`

Get current market status.

**Response**
```json
{
  "market_open": true,
  "trading_day": true,
  "next_market_open": "2024-01-16T09:30:00-05:00"
}
```

---

### Metrics

#### `GET /metrics`

Prometheus metrics endpoint.

**Response** (text/plain)
```
# HELP trading_agent_latency_seconds Time spent in each agent
# TYPE trading_agent_latency_seconds histogram
trading_agent_latency_seconds_bucket{agent="DataAgent",le="0.1"} 0
trading_agent_latency_seconds_bucket{agent="DataAgent",le="0.5"} 5
...

# HELP trading_circuit_breaker_triggers_total Circuit breaker activations
# TYPE trading_circuit_breaker_triggers_total counter
trading_circuit_breaker_triggers_total{reason="drawdown"} 1

# HELP trading_current_drawdown_percent Current portfolio drawdown
# TYPE trading_current_drawdown_percent gauge
trading_current_drawdown_percent 5.2
```

---

## Request Flow

### Trading Cycle Flow

```
Client                           API                          System
  │                               │                              │
  │  POST /cycles/run             │                              │
  │  {symbols, execute: true}     │                              │
  │──────────────────────────────>│                              │
  │                               │                              │
  │                               │  Check circuit breaker       │
  │                               │─────────────────────────────>│
  │                               │<─────────────────────────────│
  │                               │                              │
  │                               │  Run LangGraph workflow      │
  │                               │─────────────────────────────>│
  │                               │                              │
  │                               │  DataAgent                   │
  │                               │    → Fetch market data       │
  │                               │    → Generate signals        │
  │                               │                              │
  │                               │  RiskManager                 │
  │                               │    → Check hard constraints  │
  │                               │    → LLM soft assessment     │
  │                               │                              │
  │                               │  Validator                   │
  │                               │    → Pattern detection       │
  │                               │    → Second opinion          │
  │                               │                              │
  │                               │  MetaAgent                   │
  │                               │    → Synthesize decisions    │
  │                               │    → EXECUTE/DO_NOT_EXECUTE  │
  │                               │<─────────────────────────────│
  │                               │                              │
  │                               │  Execute trades (if execute) │
  │                               │─────────────────────────────>│
  │                               │    → Submit to broker        │
  │                               │    → Update positions        │
  │                               │    → Check circuit breaker   │
  │                               │<─────────────────────────────│
  │                               │                              │
  │  Response with results        │                              │
  │<──────────────────────────────│                              │
  │                               │                              │
```

### Circuit Breaker Flow

```
Trade Executed
      │
      ▼
┌─────────────────────────────────┐
│  CircuitBreakerService.evaluate │
└─────────────────────────────────┘
      │
      ├──► Check drawdown
      │    (peak_value - total_value) / peak_value > 20%?
      │
      ├──► Check consecutive losses
      │    Last N trades all losses where N >= 10?
      │
      └──► [Future] Check Sharpe ratio
           Negative for 30 days?
      │
      ▼
┌─────────────────────────────────┐
│  If threshold breached:         │
│  SystemState.circuit_breaker    │
│    = True                       │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Next cycle attempt:            │
│  Runner checks → ValueError     │
│  "Trading halted by circuit     │
│   breaker"                      │
└─────────────────────────────────┘
```

---

## Error Handling

All errors return JSON with this structure:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (validation error, circuit breaker active) |
| 404 | Resource not found |
| 500 | Internal server error |

---

## Tracing

Pass `X-Request-ID` header for distributed tracing:

```bash
curl -X POST http://localhost:8000/cycles/run \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-123" \
  -d '{"symbols": ["AAPL"]}'
```

The `trace_id` will be:
- Included in all log entries
- Returned in the response
- Echoed in `X-Request-ID` response header

---

## Examples

### Run a Dry-Run Cycle

```bash
curl -X POST http://localhost:8000/cycles/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "execute": false
  }'
```

### Run a Live Cycle (Execute Trades)

```bash
curl -X POST http://localhost:8000/cycles/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "execute": true
  }'
```

### Check and Reset Circuit Breaker

```bash
# Check status
curl http://localhost:8000/api/system/circuit-breaker

# Reset if active
curl -X POST http://localhost:8000/api/system/circuit-breaker/reset \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual review complete"}'
```

### Emergency Stop

```bash
# Disable all trading immediately
curl -X POST http://localhost:8000/api/system/trading/disable

# Re-enable later
curl -X POST http://localhost:8000/api/system/trading/enable
```
