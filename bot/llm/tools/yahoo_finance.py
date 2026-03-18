"""
Fetch closing prices and basic info from Yahoo Finance via the yfinance library.
Works with any LLM provider (Gemini, OpenAI, OpenRouter, Ollama, etc.) — 
data is fetched bot-side, no browser needed.

Entry point: get_market_prices(tickers, days) -> str
"""

from __future__ import annotations

import json
import logging
import time as time_module
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logging.warning("yfinance not installed — run: pip install yfinance")


# ── Market Hours Helper ─────────────────────────────────────────────────────

def _is_taiwan_market_hours() -> bool:
    """Check if Taiwan market is currently open (9:00-13:30 TW)."""
    now = datetime.now()
    tw_time = now.time()
    # Taiwan market hours: 9:00 AM - 1:30 PM
    return time(9, 0) <= tw_time <= time(13, 30)


def _get_current_price(symbol: str) -> float | None:
    """
    Get intraday price during market hours, or None if market closed.
    Uses 15-minute interval data to get latest price during trading hours.
    """
    if not _is_taiwan_market_hours():
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        # Get 15-min interval data for last 2 days
        hist = ticker.history(period='2d', interval='15m')
        if hist.empty:
            return None
        # Get the most recent price
        current = hist.iloc[-1]["Close"]
        if pd.notna(current) and current > 0:
            logging.debug("yahoo_finance: %s current price (intraday) = %s", symbol, current)
            return float(current)
    except Exception as e:
        logging.warning("yahoo_finance: failed to get intraday price for %s: %s", symbol, e)
    return None


# ── Previous Close Helper ───────────────────────────────────────────────────

def _get_previous_close(symbol: str) -> float | None:
    """
    Fetch previous close from yfinance info endpoint.
    This is more reliable than hist.iloc[-2] when recent data is missing.
    
    Returns:
        Previous close price, or None if unavailable
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
        # Validate: must be a number > 0
        if prev_close is not None and not isinstance(prev_close, bool):
            try:
                prev_close_float = float(prev_close)
                if prev_close_float > 0:
                    logging.debug("yahoo_finance: %s previousClose from info = %s", symbol, prev_close_float)
                    return prev_close_float
            except (TypeError, ValueError):
                pass
    except Exception as e:
        logging.warning("yahoo_finance: failed to get previousClose for %s: %s", symbol, e)
    return None


# ── Retry Helper ──────────────────────────────────────────────────────────────

def _fetch_with_retry(symbol: str, days: int, max_attempts: int = 3) -> Any:
    """
    Fetch stock data with retry logic for transient failures.
    
    Args:
        symbol: Yahoo Finance ticker symbol
        days: Number of calendar days to fetch
        max_attempts: Maximum retry attempts (default 3)
    
    Returns:
        DataFrame with historical data (may be empty)
    """
    for attempt in range(max_attempts):
        try:
            hist = yf.Ticker(symbol).history(period=f"{days}d")
            
            # Log response shape for debugging (Phase 3)
            logging.debug(
                "yahoo_finance: fetched %s period=%dd -> shape=%s",
                symbol, days, hist.shape if hasattr(hist, 'shape') else 'N/A'
            )
            
            if not hist.empty:
                return hist
            
            # Empty data - retry with backoff
            if attempt < max_attempts - 1:
                delay = 1 * (2 ** attempt)  # 1s, 2s exponential backoff
                logging.debug(
                    "yahoo_finance: empty data for %s, retrying in %ds (attempt %d/%d)",
                    symbol, delay, attempt + 1, max_attempts
                )
                time_module.sleep(delay)
                
        except Exception as e:
            logging.warning("yahoo_finance: error fetching '%s' (attempt %d): %s", 
                          symbol, attempt + 1, e)
            if attempt < max_attempts - 1:
                time_module.sleep(1 * (2 ** attempt))
    
    # Return empty DataFrame after all retries failed
    return hist if 'hist' in locals() else pd.DataFrame()


# ── Callable ──────────────────────────────────────────────────────────────────

def get_market_prices(tickers: str, days: int = 10) -> str:
    """
    Fetch recent closing prices for one or more tickers from Yahoo Finance.

    Args:
        tickers: Comma-separated Yahoo Finance ticker symbols.
                 Examples: "^TWII,^GSPC,^IXIC,^SOX" or "0050.TW,TSMC,AAPL"
        days:    Number of calendar days of history to fetch (default 10).
                 Use 10 to ensure you get at least 2 trading days across weekends.

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
        hist = _fetch_with_retry(symbol, days)
        
        # 1.1 Check for empty data
        if hist.empty:
            lines.append(f"{symbol}: no data returned (check ticker symbol)")
            continue

        # Try to get current price during market hours (more fresh)
        current_price = _get_current_price(symbol)
        
        last = hist.iloc[-1]
        
        # Use current price if available (during market hours), otherwise use last close
        if current_price is not None:
            close = current_price
            trade_date = date.today()
        else:
            # 1.1 Add NaN check for close price before using it
            close = last["Close"]
            if pd.isna(close):
                trade_date = hist.index[-1].date()
                lines.append(f"{symbol}: close price unavailable [{trade_date}]")
                continue
            trade_date = hist.index[-1].date()
        
        # 1.2 Add NaN check for prev_close before calculating % change
        # 1.3 Add check for zero prev_close to prevent division by zero
        # Use info endpoint for previous close (more reliable when hist data has gaps)
        prev_close = _get_previous_close(symbol)
        
        if prev_close is None and len(hist) >= 2:
            # Fallback to hist if info endpoint fails
            prev_close = hist.iloc[-2]["Close"]
        
        if prev_close is None or prev_close == 0:
            lines.append(f"{symbol}: {close:.2f}  [no prev day]  [{trade_date}]")
        else:
            change = close - prev_close
            pct = (change / prev_close) * 100
            # Add marker if using intraday price during market hours
            marker = " *" if current_price is not None else ""
            lines.append(
                f"{symbol}: {close:.2f}  {change:+.2f} ({pct:+.2f}%)  [{trade_date}]{marker}"
            )

    return "\n".join(lines)


# ── Schema ────────────────────────────────────────────────────────────────────

YAHOO_FINANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_market_prices",
        "description": (
            "Fetch accurate closing prices and daily % change for stock indices or tickers "
            "from Yahoo Finance. Use this for any market data question — it returns real numbers. "
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
                        "Calendar days of history to fetch. Default 10. "
                        "Use 10 or more to ensure at least 2 trading days are available (especially near weekends)."
                    ),
                    "minimum": 5,
                },
            },
            "required": ["tickers"],
        },
    },
}


# ── Dynamic tool registration ────────────────────────────────────────────────

from bot.llm.tools._types import ToolEntry

TOOL_NAME = "get_market_prices"
TOOL_ENTRY = ToolEntry(
    schema=YAHOO_FINANCE_SCHEMA,
    fn=get_market_prices,
)