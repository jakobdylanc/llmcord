## Context

The `get_market_prices` tool in `bot/llm/tools/yahoo_finance.py` retrieves stock data from Yahoo Finance via the yfinance library. Currently, it has several reliability issues:

1. **Saturday morning failures**: Task runs at 06:30 UTC Monday-Friday, but when yfinance cache is stale, it returns empty data
2. **NaN handling**: When yfinance returns NaN values (common for certain Taiwan ETFs), the tool either crashes or produces broken output
3. **No retry logic**: Single fetch attempt with no fallback for transient failures
4. **Debugging difficulty**: Insufficient logging to diagnose issues

## Goals / Non-Goals

**Goals:**
- Handle NaN/None values gracefully in data extraction
- Add retry logic for transient empty data failures
- Ensure at least 2 trading days of data for reliable % change calculation
- Improve observability with better logging

**Non-Goals:**
- Change the data source (still use yfinance)
- Add new stock tickers or markets
- Change the persona or output format (that's separate)
- Add caching layer (out of scope for this fix)

## Decisions

### 1. NaN Handling Approach
**Decision**: Check for NaN values before using them, with clear error messaging

```python
# Before (line 53-68):
close = last["Close"]
prev_close = hist.iloc[-2]["Close"]
pct = (change / prev_close) * 100

# After:
if pd.isna(close):
    lines.append(f"{symbol}: close price unavailable [{trade_date}]")
    continue
```

**Rationale**: Prevents crashes and provides clear feedback to LLM

### 2. Retry Logic
**Decision**: Add retry with exponential backoff (max 3 attempts, 1s delay)

```python
def fetch_with_retry(symbol, days, max_attempts=3):
    for attempt in range(max_attempts):
        hist = yf.Ticker(symbol).history(period=f"{days}d")
        if not hist.empty:
            return hist
        if attempt < max_attempts - 1:
            time.sleep(1 * (2 ** attempt))  # 1s, 2s
    return hist  # Return even if empty for error handling
```

**Rationale**: Handles transient Saturday morning cache issues without adding complex infrastructure

### 3. Days Parameter
**Decision**: Increase default from 7 to 10 days for more reliability

**Rationale**: 10 calendar days ensures at least 2 trading days even with holidays or market closures

### 4. Use Explicit Date Range (Alternative Considered)
**Decision**: Keep using `period` parameter but with more days

**Alternatives considered**:
- Use explicit `start`/`end` dates: More precise but requires date calculation
- Keep with `period=10d`: Simpler and sufficient for the use case

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| yfinance still returns empty | Medium | Retry helps, but may still fail on major holidays |
| Slower execution | Low | 2 extra seconds max with retry |
| NaN handling changes output format | Low | LLM will see "[unavailable]" instead of "nan" |

## Additional Design Decisions (Phases 5-8)

### Phase 5: Schema & Skill Documentation Fixes

During testing, discovered the LLM was passing `days=2` instead of the default 10.

**Decision**: Fix schema defaults and add minimum constraint

```python
YAHOO_FINANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_market_prices",
        "description": "Get current stock prices...",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {...},
                "days": {
                    "type": "integer",
                    "description": "Number of days of historical data to fetch... Default is 10.",
                    "default": 10,
                    "minimum": 5
                }
            }
        }
    }
}
```

**Rationale**: Prevents LLM from passing low values that cause insufficient data

### Phase 6: yfinance Missing Historical Data Fix

**Problem**: yfinance historical data was missing trading days (e.g., March 17 for Taiwan ETFs), causing wrong % calculations when using `hist.iloc[-2]`.

**Decision**: Use info endpoint for reliable previous close

```python
def _get_previous_close(self, symbol):
    """Get previous close from info endpoint (more reliable)."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return info.get("previousClose") or info.get("regularMarketPreviousClose")
```

**Rationale**: The info endpoint always has the correct previous close, independent of historical data gaps

### Phase 7: Intraday Price During Market Hours

**Problem**: During Taiwan market hours (9:00-13:30), daily close data is stale (from previous day).

**Decision**: Fetch intraday 15-minute interval data during market hours

```python
def _is_taiwan_market_hours():
    """Check if Taiwan market is currently open (9:00-13:30 TW)."""
    tw_now = datetime.now(timezone(timedelta(hours=8)))
    hour = tw_now.hour
    minute = tw_now.minute
    market_time = hour * 60 + minute
    return 540 <= market_time <= 810  # 9:00 = 540 mins, 13:30 = 810 mins

def _get_current_price(self, symbol):
    """Get current price from intraday 15-min interval data."""
    ticker = yf.Ticker(symbol)
    intraday = ticker.history(interval="15m", period="1d")
    if not intraday.empty:
        return intraday.iloc[-1]["Close"]
    return None
```

**Rationale**: Fresh prices during market hours provide real value for scheduled tasks

### Phase 8: LLM Modifying Percentages

**Problem**: LLM was modifying percentages after receiving correct data (e.g., adding decimals).

**Decision**: Add explicit instructions to persona and skill documentation

```markdown
<!-- In stock_market_analyst.md -->
CRITICAL: Always use the EXACT percentage values returned by the get_market_prices tool.
Do NOT add decimals, modify, or recalculate percentages. The tool already provides
the correct calculations with proper decimal precision.
```

**Rationale**: Direct instruction prevents LLM from second-guessing correct tool output

## Migration Plan

1. Update `yahoo_finance.py` with new error handling
2. Deploy to production (no breaking changes to API)
3. Monitor logs for any remaining failures
4. If needed, adjust retry count or delay

## Open Questions

- Should we add a formatter for the tool output to handle edge cases at display level?
- Should we document known limitations for specific leverage ETFs?