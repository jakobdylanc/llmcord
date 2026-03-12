"""
Web search via Brave API.

Use web_search when:
- The user asks for up-to-date, recent, or real-time information.
- The answer may change over time (e.g., news, prices, releases, events).
- You need external information that is not guaranteed to be in your knowledge base.

When using web_search:
- Extract relevant facts directly from the search results.
- Do NOT summarize what the website does.
- Do NOT describe the website itself.
- Only include information directly related to the user's question.

Always prioritize factual extraction over explanation of the source.
"""

from __future__ import annotations

import os
import time
import threading as _threading

import requests
from dotenv import load_dotenv

load_dotenv()


# ── Brave API ─────────────────────────────────────────────────────────────────

_BRAVE_LAST_CALL: float = 0.0
_BRAVE_MIN_INTERVAL: float = 1.2  # free tier: 1 req/s
_BRAVE_LOCK = _threading.Lock()  # serialise concurrent calls


def brave_web_search(query: str) -> dict:
    """Call Brave Search API and return the raw JSON response."""
    global _BRAVE_LAST_CALL
    with _BRAVE_LOCK:  # one request at a time — serialises rate limiting
        elapsed = time.monotonic() - _BRAVE_LAST_CALL
        if elapsed < _BRAVE_MIN_INTERVAL:
            time.sleep(_BRAVE_MIN_INTERVAL - elapsed)
        api_key = os.getenv("BRAVE_API_KEY") or os.getenv("brave_api_key")
        if not api_key:
            raise ValueError("Brave API key missing — set BRAVE_API_KEY in .env")
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 10},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        _BRAVE_LAST_CALL = time.monotonic()
        return response.json()


# ── Formatters ────────────────────────────────────────────────────────────────

def format_brave_results(result: dict, user_search: str) -> str:
    """Format Brave API response into clean text for the model."""
    output: list[str] = [f'Search results for "{user_search}":']
    results = result.get("web", {}).get("results", [])
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        desc = r.get("description", "")
        output.append(title or url)
        output.append(f"   URL: {url}")
        output.append(f"   Content: {desc}")
        output.append("")
    return "\n".join(output).rstrip()


# ── Formatter wrapper for dynamic registration ────────────────────────────────

def _fmt_brave_search(result: dict, args: dict) -> str:
    """Wrapper for dynamic tool registration."""
    return format_brave_results(result, user_search=args.get("query", ""))


# ── Schemas ───────────────────────────────────────────────────────────────────

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}


# ── Dynamic tool registration ────────────────────────────────────────────────

from bot.llm.tools._types import ToolEntry

TOOL_NAME = "web_search"
TOOL_ENTRY = ToolEntry(
    schema=WEB_SEARCH_SCHEMA,
    fn=brave_web_search,
    formatter=_fmt_brave_search,
)
