---
name: get_market_prices
description: Fetch accurate closing prices and daily % change from Yahoo Finance. Use this whenever the user asks for market data, stock prices, or index levels.
metadata: {"clawhub":{"emoji":"📈","requires":{},"tools":["get_market_prices"]}}
---

# get_market_prices

Fetches real closing prices and % change directly from Yahoo Finance.
Use this as the **primary tool** for any market data question — it returns exact numbers,
unlike web_search which only returns page snippets.

## When to use

- User asks for index levels, closing prices, or daily % change
- You need specific numbers to fill in a market report
- web_search returned no numeric data in snippets

## Alias Handling Rule (IMPORTANT)

If the user provides a Taiwan stock **alias** (e.g. `2330`, `0050`, `00878`):

- Convert the alias to the full Yahoo Finance ticker
- Append `.TW`
- Then call `get_market_prices` using the full ticker

### Example conversion

| User Input | Converted Ticker |
|------------|------------------|
| `2330`     | `2330.TW`        |
| `0050`     | `0050.TW`        |
| `00631L`   | `00631L.TW`      |
| `0056`     | `0056.TW`        |
| `00878`    | `00878.TW`       |

Never call the tool with alias values.
Always call using the full ticker format recognized by Yahoo Finance.

## Tool signature

```
get_market_prices(tickers: str, days: int = 10) -> str
```

> **IMPORTANT**: Always use `days=10` (or higher) when calling this tool. Using values lower than 5 may result in insufficient trading days for calculating % change, especially near weekends. The default is 10 to ensure reliable data.

## Output Format (IMPORTANT)

The tool returns data in this exact format:
```
0050.TW: 77.90  +2.30 (+3.04%)  [2026-03-18]
```

**DO NOT modify the numbers** - use the exact values from the tool output:
- **Close price**: Use exactly as shown (e.g., `77.90`)
- **Change amount**: Use exactly as shown (e.g., `+2.30`)
- **Percentage**: Use exactly as shown (e.g., `+3.04%`)

When displaying in your response, include both the absolute change and percentage from the tool output.

## Common tickers

### Index
| Ticker  | Name             | Market |
| ------- | ---------------- | ------ |
| `^TWII` | 台灣加權指數           | TWSE   |
| `^GSPC` | S&P 500          | US     |
| `^IXIC` | Nasdaq Composite | US     |
| `^SOX`  | 費城半導體指數 (SOX)    | US     |
| `^DJI`  | 道瓊工業指數           | US     |
| `^VIX`  | CBOE 波動率指數 (VIX) | US     |

### Taiwan Stocks
| Ticker      | Alias    | Name     | Market |
| ----------- | -------- | -------- | ------ |
| `2330.TW`   | `2330`   | 台積電      | TWSE   |
| `0050.TW`   | `0050`   | 元大台灣50   | TWSE   |
| `00631L.TW` | `00631L` | 元大台灣50正2 | TWSE   |
| `0056.TW`   | `0056`   | 元大高股息    | TWSE   |
| `00878.TW`  | `00878`  | 國泰永續高股息  | TWSE   |

### US Stocks
| Ticker | Company               | Market |
| ------ | --------------------- | ------ |
| `AAPL` | Apple Inc.            | NASDAQ |
| `NVDA` | NVIDIA Corporation    | NASDAQ |
| `MSFT` | Microsoft Corporation | NASDAQ |
| `META` | Meta Platforms, Inc.  | NASDAQ |
| `QQQ`  | Invesco QQQ Trust     | NASDAQ |

### ADR
| Ticker | Company                                              | Market |
| ------ | ---------------------------------------------------- | ------ |
| `TSM`  | Taiwan Semiconductor Manufacturing Company (台積電 ADR) | NYSE   |

### Crypto
| Ticker    | Asset    | Market |
| --------- | -------- | ------ |
| `BTC-USD` | Bitcoin  | Crypto |
| `ETH-USD` | Ethereum | Crypto |


## Usage examples

Fetch all major indices at once:
```
get_market_prices(tickers="^TWII,^GSPC,^IXIC,^SOX")
```

Fetch with extra days to cover weekends:
```
get_market_prices(tickers="^TWII,^GSPC", days=7)
```

## Output format

```
^TWII: 21543.20  +123.45 (+0.58%)  [2026-02-28]
^GSPC: 5832.10   -45.20  (-0.77%)  [2026-02-28]
^IXIC: 18421.50  -210.30 (-1.13%)  [2026-02-28]
^SOX:  4821.30   -98.40  (-2.00%)  [2026-02-28]
```

## Notes

- Always call with all needed tickers in **one call** to minimise round trips
- Use `days=7` near weekends to ensure at least 2 trading days are available for % change
- Prices are from Yahoo Finance's delayed feed (typically 15–20 min delay for live, exact for historical)
- For Taiwan stocks: append `.TW` (e.g. `2330.TW` for 台積電)