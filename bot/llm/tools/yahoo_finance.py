"""
bot/llm/tools/yahoo_finance.py

Fetch closing prices and basic info from Yahoo Finance via the yfinance library.
Works with any LLM provider (Gemini, OpenAI, OpenRouter, Ollama, etc.) — 
data is fetched bot-side, no web_fetch or browser needed.

Entry point: get_market_prices(tickers, days) -> str
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logging.warning("yfinance not installed — run: pip install yfinance")


# ── Callable ──────────────────────────────────────────────────────────────────

def get_market_prices(tickers: str, days: int = 5) -> str:
    """
    Fetch recent closing prices for one or more tickers from Yahoo Finance.

    Args:
        tickers: Comma-separated Yahoo Finance ticker symbols.
                 Examples: "^TWII,^GSPC,^IXIC,^SOX" or "0050.TW,TSMC,AAPL"
        days:    Number of calendar days of history to fetch (default 5).
                 Use 5–10 to ensure you get at least 2 trading days across weekends.

    Returns:
        Formatted string with date, close price, and % change for each ticker.
    """
    if not _YF_AVAILABLE:
        return "Error: yfinance is not installed. Run: pip install yfinance"

    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return "Error: no tickers provided."

    lines: list[str] = []

    for symbol in ticker_list:
        try:
            hist = yf.Ticker(symbol).history(period=f"{days}d")
            if hist.empty:
                lines.append(f"{symbol}: no data returned (check ticker symbol)")
                continue

            last = hist.iloc[-1]
            close = last["Close"]
            trade_date = hist.index[-1].date()

            if len(hist) >= 2:
                prev_close = hist.iloc[-2]["Close"]
                change = close - prev_close
                pct = (change / prev_close) * 100
                lines.append(
                    f"{symbol}: {close:.2f}  {change:+.2f} ({pct:+.2f}%)  [{trade_date}]"
                )
            else:
                lines.append(f"{symbol}: {close:.2f}  [no prev day]  [{trade_date}]")

        except Exception as e:
            logging.error("yahoo_finance: error fetching '%s': %s", symbol, e)
            lines.append(f"{symbol}: error — {e}")

    return "\n".join(lines)


# ── Schema ────────────────────────────────────────────────────────────────────

YAHOO_FINANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_market_prices",
        "description": (
            "Fetch accurate closing prices and daily % change for stock indices or tickers "
            "from Yahoo Finance. Use this for any market data question — it returns real numbers. "
            "Common tickers: ^TWII (台灣加權指數), ^GSPC (S&P 500), ^IXIC (Nasdaq), "
            "^SOX (費半/SOX), 0050.TW (元大台灣50), TSMC (台積電 ADR), "
            "^DJI (道瓊), ^VIX (恐慌指數)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "string",
                    "description": (
                        "Comma-separated Yahoo Finance ticker symbols. "
                        "Example: \"^TWII,^GSPC,^IXIC,^SOX\""
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "Calendar days of history to fetch. Default 5. "
                        "Use 7 if fetching near a weekend to ensure 2 trading days are available."
                    ),
                },
            },
            "required": ["tickers"],
        },
    },
}