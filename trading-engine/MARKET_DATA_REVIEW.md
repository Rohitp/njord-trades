# Market Data Integration Code Review

## ✅ Strengths

1. **Clean Architecture**: Good separation with provider abstraction
2. **Fallback Logic**: Automatic fallback from Alpaca to yfinance
3. **Async/Await**: Proper async implementation
4. **Logging**: Structured logging throughout
5. **Type Hints**: Good type annotations

## 🔴 Critical Issues

### 1. **Missing Error Handling in AlpacaProvider**

**Issue**: `get_quote()` and `get_bars()` can raise `KeyError` if symbol not in response.

**Location**: `alpaca_provider.py:45, 88`

```python
# Current (unsafe):
quote_data = response[symbol]  # KeyError if symbol missing

# Should be:
if symbol not in response:
    raise ValueError(f"Symbol {symbol} not found in Alpaca response")
quote_data = response[symbol]
```

### 2. **Division by Zero Risk**

**Issue**: `get_quote()` calculates price as `(bid + ask) / 2` without checking for None.

**Location**: `alpaca_provider.py:49`

```python
# Current (unsafe):
price=(quote_data.bid_price + quote_data.ask_price) / 2

# Should check:
if quote_data.bid_price is None or quote_data.ask_price is None:
    price = quote_data.bid_price or quote_data.ask_price
    if price is None:
        raise ValueError(f"No price data for {symbol}")
else:
    price = (quote_data.bid_price + quote_data.ask_price) / 2
```

### 3. **Incorrect RSI Calculation**

**Issue**: RSI uses simple average instead of Wilder's smoothing method (industry standard).

**Location**: `alpaca_provider.py:135-154`, `yfinance_provider.py:113-132`

**Current**: Simple average of gains/losses
**Should be**: Wilder's exponential smoothing

```python
# Current (incorrect):
avg_gain = sum(gains) / period
avg_loss = sum(losses) / period

# Should be (Wilder's method):
# First period: simple average
# Subsequent: avg = (prev_avg * (period - 1) + current) / period
```

### 4. **Missing Retry Logic**

**Issue**: Per requirements, all external API calls should retry 3x with exponential backoff.

**Location**: Both providers, all API calls

**Required**: Add retry decorator/wrapper with:
- 3 attempts max
- Exponential backoff (2s, 4s, 8s)
- Retry on: APIError, Timeout, ConnectionError

### 5. **YFinance Error Handling**

**Issue**: `get_quote()` doesn't handle `last_price` being None.

**Location**: `yfinance_provider.py:30-31`

```python
# Current (unsafe):
price=info.last_price  # Could be None

# Should be:
if info.last_price is None:
    raise ValueError(f"No price data available for {symbol}")
```

### 6. **Empty DataFrame Handling**

**Issue**: `get_bars()` doesn't handle empty DataFrame from yfinance.

**Location**: `yfinance_provider.py:61`

```python
# Should add:
if df.empty:
    raise ValueError(f"No historical data for {symbol}")
```

## ⚠️ Medium Priority Issues

### 7. **Type Safety in Service**

**Issue**: `_with_fallback()` operation parameter lacks type hints.

**Location**: `service.py:88`

```python
# Current:
async def _with_fallback(self, operation, operation_name: str):

# Should be:
from collections.abc import Callable, Awaitable
async def _with_fallback(
    self, 
    operation: Callable[[MarketDataProvider], Awaitable],
    operation_name: str
):
```

### 8. **Performance: DataFrame Iteration**

**Issue**: `iterrows()` is slow for large DataFrames.

**Location**: `yfinance_provider.py:64`

```python
# Current (slow):
for idx, row in df.iterrows():

# Better:
for idx in df.index:
    row = df.loc[idx]
    # or use vectorized operations
```

### 9. **Missing Timestamp in API Response**

**Issue**: `QuoteResponse` doesn't include timestamp.

**Location**: `api/routers/market_data.py:13-19`

Should add `timestamp: datetime | None` to `QuoteResponse`.

### 10. **No Symbol Validation**

**Issue**: No validation that symbol format is valid before API calls.

**Location**: All providers

Should validate symbol format (uppercase, alphanumeric, length, etc.).

### 11. **Singleton Pattern for Testing**

**Issue**: Global `market_data_service` singleton makes testing harder.

**Location**: `service.py:117`

Consider using dependency injection in FastAPI instead of global singleton.

## 💡 Suggestions

### 12. **Add Caching**

Consider caching expensive operations (technical indicators) for a short period (e.g., 5 minutes).

### 13. **Rate Limiting**

Add rate limiting considerations for API calls to avoid hitting provider limits.

### 14. **Batch Operations**

`get_indicators_batch()` processes sequentially. Could parallelize with `asyncio.gather()`.

### 15. **Error Types**

Create custom exceptions instead of generic `ValueError`:
- `SymbolNotFoundError`
- `NoDataAvailableError`
- `ProviderError`

## 📋 Priority Fix Order

1. **Critical**: Fix RSI calculation (affects trading decisions)
2. **Critical**: Add error handling for KeyError/None values
3. **Critical**: Add retry logic (per requirements)
4. **Medium**: Fix type hints
5. **Medium**: Add symbol validation
6. **Low**: Performance optimizations
7. **Low**: Add caching

