## Why

The `get_market_prices` tool frequently fails to retrieve correct stock price data:
1. **Saturday mornings**: Task runs at 06:30 UTC, but yfinance cache can be stale, returning empty data
2. **NaN values**: When yfinance returns NaN for certain Taiwan ETFs (0050.TW, 006208.TW, etc.), tool crashes or shows "---"
3. **Missing trading days**: yfinance sometimes misses recent trading days (e.g., March 17), causing wrong % calculations
4. **Stale prices**: During market hours, daily close data doesn't reflect current prices
5. **LLM modification**: The LLM was modifying percentages after receiving correct data

## What Changes

### Phase 1: Data Extraction Robustness
- Add NaN/None handling in data extraction to prevent crashes
- Add clear error messages: "[unavailable]" for missing close, "[no prev day]" for missing %

### Phase 2: Retry Logic
- Add retry with exponential backoff (max 3 attempts: 1s, 2s delay)
- Only retry on empty data, not invalid tickers

### Phase 3: Data Reliability
- Increase default `days` from 5 to 10 for at least 2 trading days
- Add response shape logging for debugging

### Phase 4: Schema Fixes
- Fix schema default from 5 to 10
- Add `minimum: 5` constraint to prevent low values like 2

### Phase 5: LLM Behavior Fixes
- Add persona instruction: "use EXACT values, don't modify"
- Add skill doc instruction about using exact tool output

### Phase 6: Missing Historical Data (Discovered during implementation)
- Add `_get_previous_close()` using info endpoint for reliable previous close
- Fix: yfinance missing March 17 data for ETFs caused wrong % calculations

### Phase 7: Intraday Prices (Discovered during implementation)
- Add `_is_taiwan_market_hours()` check (9:00-13:30 TW)
- Add `_get_current_price()` for 15-minute interval intraday data
- Use intraday prices during market hours, daily close otherwise
- Add `*` marker to indicate intraday price

## Capabilities

### New Capabilities
- Intraday price fetching during Taiwan market hours for fresh data

### Modified Capabilities
- `tool-system`: Improved error handling, reliability, and data freshness

## Impact

- **Code**: `bot/llm/tools/yahoo_finance.py` - main implementation
- **Tests**: `bot/test/test_yahoo_finance.py` - 14 unit tests
- **Config**: `bot/config/tasks/stock-market-checker.yaml` - schedule (MON-FRI 06:30 UTC)
- **Documentation**: 
  - `bot/llm/tools/skills/yahoo_finance.md` - usage instructions
  - `bot/config/personas/stock_market_analyst.md` - LLM instructions