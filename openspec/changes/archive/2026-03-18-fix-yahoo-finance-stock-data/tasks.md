## 1. Data Extraction Robustness

- [x] 1.1 Add NaN check for close price before using it
- [x] 1.2 Add NaN check for prev_close before calculating % change
- [x] 1.3 Add check for zero prev_close to prevent division by zero
- [x] 1.4 Add clear messaging when data is unavailable
- [x] 1.5 **TEST**: Create `bot/test/test_yahoo_finance.py` with mock setup
- [x] 1.6 **TEST**: Test NaN close price → outputs "[unavailable]" message
- [x] 1.7 **TEST**: Test NaN prev_close → shows close without % change
- [x] 1.8 **TEST**: Test zero prev_close → prevents division by zero error
- [x] 1.9 **TEST**: Test empty yfinance response → returns clear error message
- [x] 1.10 **TEST**: Test multiple tickers (mixed success/failure)

## 2. Retry Logic

- [x] 2.1 Add retry function with exponential backoff (max 3 attempts)
- [x] 2.2 Add delay between retries (1s, 2s)
- [x] 2.3 Only retry on empty data, not on invalid tickers
- [x] 2.4 **TEST**: Test empty data triggers retry (verify 3 attempts made)
- [x] 2.5 **TEST**: Test exponential backoff delays (~1s first, ~2s second)
- [x] 2.6 **TEST**: Test invalid ticker does NOT retry (immediate failure)
- [x] 2.7 **TEST**: Test valid ticker succeeds after retry

## 3. Data Reliability

- [x] 3.1 Increase default days from 7 to 10 for more reliability
- [x] 3.2 Add logging of yfinance response shape for debugging
- [x] 3.3 **TEST**: Test default days parameter is 10
- [x] 3.4 **TEST**: Test logging shows response shape (rows, columns)
- [x] 3.5 **TEST**: Test with known problematic tickers (0050.TW, 006208.TW)

## 4. Edge Cases & Integration

- [x] 4.1 Test empty ticker list → returns "no tickers provided" error
- [x] 4.2 Test yfinance not installed → returns installation error message
- [x] 4.3 Test single day data → shows "[no prev day]" correctly
- [x] 4.4 Test single ticker vs multiple tickers all succeed
- [x] 4.5 **TEST**: Run full pytest suite for yahoo_finance

## 5. Schema & Skill Documentation Fixes

- [x] 5.1 Fix schema description: change "Default 5" to "Default 10" in YAHOO_FINANCE_SCHEMA
- [x] 5.2 Add `minimum: 5` constraint to days parameter in schema to prevent low values like 2
- [x] 5.3 Update docstring comment to match (currently says "default 5")
- [x] 5.4 **DOCS**: Update `bot/llm/tools/skills/yahoo_finance.md` with explicit instruction to use days=10
- [x] 5.5 **TEST**: Verify schema shows correct default and constraints ✅

## 6. LLM Percentage Calculation Fix

- [x] 6.1 Add instruction to persona: use EXACT values from tool, don't modify
- [x] 6.2 Add instruction to skill doc: use exact values from tool output
- [x] 6.3 **TEST**: Run task again and verify percentages match tool output exactly ✅

## 7. yfinance Missing Data Fix

- [x] 7.1 Add `_get_previous_close()` function using info endpoint
- [x] 7.2 Use info previous close instead of hist.iloc[-2]
- [x] 7.3 Add fallback to hist when info returns None
- [x] 7.4 **TEST**: All tests pass (14/14)

## 8. Intraday Price During Market Hours

- [x] 8.1 Add `_is_taiwan_market_hours()` check (9:00-13:30 TW)
- [x] 8.2 Add `_get_current_price()` for intraday data
- [x] 8.3 Use intraday price during market hours, daily close otherwise
- [x] 8.4 Add `*` marker for intraday prices
- [x] 8.5 **TEST**: All tests pass (14/14)