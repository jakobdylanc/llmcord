# Available Skills

## Web Search
**Name:** `web_search`
**Description:** Search the web for current information (Brave API).
**Parameters:**
- `query` (string): The search query

## Visualization Core
**Name:** `visuals_core`
**Description:** Generate data visualizations.
**Parameters:**
- `viz_type` (string): Type of visualization (e.g., 'bar', 'line', 'pie')
- `data` (string): JSON string containing data points
- `title` (string, optional): Title for the visualization

## Yahoo Finance (Market Prices)
**Name:** `get_market_prices`
**Description:** Fetch closing prices and daily % change for stock indices or tickers from Yahoo Finance.
**Parameters:**
- `tickers` (string): Comma-separated Yahoo Finance ticker symbols (e.g. ^TWII, ^GSPC, 0050.TW)
- `days` (integer, optional): Calendar days of history to fetch (default 5)