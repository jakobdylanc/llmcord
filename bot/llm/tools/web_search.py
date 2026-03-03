"""
You have access to two web tools: web_search and web_fetch.

Use web_search when:
- The user asks for up-to-date, recent, or real-time information.
- The answer may change over time (e.g., news, prices, releases, events).
- You need external information that is not guaranteed to be in your knowledge base.

When using web_search:
- Extract relevant facts directly from the search results.
- Do NOT summarize what the website does.
- Do NOT describe the website itself.
- Only include information directly related to the user's question.

Use web_fetch when:
- The user provides a specific URL.
- You need to retrieve detailed content from that page.

When using web_fetch:
- Only present the information requested by the user.
- Ignore irrelevant sections of the page.
- Do NOT describe the website itself.

Always prioritize factual extraction over explanation of the source.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Union

import time
import requests
from dotenv import load_dotenv
from ollama import WebFetchResponse, WebSearchResponse

load_dotenv()


# ── Brave API ─────────────────────────────────────────────────────────────────

import threading as _threading

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


def format_web_search_results(
    results: Union[WebSearchResponse, WebFetchResponse, dict],
    user_search: str,
) -> str:
    """Format Ollama-native or Brave results into clean text for the model."""
    # Brave returns a plain dict
    if isinstance(results, dict):
        return format_brave_results(results, user_search)

    output: list[str] = []

    if isinstance(results, WebSearchResponse):
        output.append(f'Search results for "{user_search}":')
        for result in results.results:
            output.append(f"{result.title}" if result.title else f"{result.content}")
            output.append(f"   URL: {result.url}")
            output.append(f"   Content: {result.content}")
            output.append("")
        return "\n".join(output).rstrip()

    if isinstance(results, WebFetchResponse):
        output.append(f'Fetch results for "{user_search}":')
        output.extend([
            f"Title: {results.title}",
            f"URL: {user_search}" if user_search else "",
            f"Content: {results.content}",
        ])
        if results.links:
            output.append(f"Links: {', '.join(results.links)}")
        output.append("")
        return "\n".join(output).rstrip()

    return ""


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

WEB_FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch the contents of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"],
        },
    },
}