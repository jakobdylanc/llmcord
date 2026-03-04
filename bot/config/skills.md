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

## Google Tools (Gmail + Calendar)
**Name:** `google_tools`
**Description:** Access Gmail emails and Google Calendar events (read-only). Requires OAuth credentials.
**Parameters:**
- `action` (string): Action to perform - `get_emails`, `get_email_content`, or `get_events`
- `count` (integer, optional): Number of items to fetch (default 10, max 50)
- `label_id` (string, optional): Gmail label - `INBOX`, `UNREAD`, `STARRED`, `SENT`, `IMPORTANT` (default INBOX)
- `query` (string, optional): Gmail search query (e.g., `in:inbox -category:promotions -in:spam`). Use instead of label_id for advanced filtering
- `message_id` (string, optional): Gmail message ID (required for get_email_content)
- `calendar_id` (string, optional): Calendar ID (default primary)