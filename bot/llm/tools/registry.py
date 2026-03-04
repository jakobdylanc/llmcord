"""
bot/llm/tools/registry.py

Single source of truth for ALL bot tools.
Each ToolEntry bundles: the callable, the OpenAI-format schema, and an optional formatter.

Adding a new tool only requires:
  1. Create bot/llm/tools/my_tool.py  (fn + SCHEMA)
  2. Import and add a ToolEntry below in _ENTRIES
  3. Add the tool name to config.yaml under the model's `tools:` list

See SKILLS.md at the repo root for full documentation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, List

from .web_search import (
    WEB_SEARCH_SCHEMA,
    brave_web_search,
    format_brave_results,
)
from .visuals_core import VISUALS_CORE_SCHEMA, generate_visualization
from .yahoo_finance import YAHOO_FINANCE_SCHEMA, get_market_prices


# ── ToolEntry ─────────────────────────────────────────────────────────────────

@dataclass
class ToolEntry:
    schema: dict                       # OpenAI-format schema sent to the model
    fn: Callable | None = None         # called locally when model invokes this tool
    formatter: Callable | None = None  # optional: formatter(result, args) -> str


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_brave_search(result: Any, args: dict) -> str:
    return format_brave_results(result, user_search=args.get("query", ""))


# ── visuals_core wrapper ───────────────────────────────────────────────────────
# Unpacks the JSON `data` string arg before calling generate_visualization,
# because models pass nested dicts as JSON strings.

def _visuals_core(viz_type: str, data: str, title: str = "") -> str:
    try:
        kwargs = json.loads(data) if isinstance(data, str) else data
    except Exception:
        kwargs = {}
    return generate_visualization(viz_type=viz_type, title=title, **kwargs)


# ── Static entries (schema + callable) ────────────────────────────────────────

_ENTRIES: dict[str, ToolEntry] = {
    "web_search": ToolEntry(
        schema=WEB_SEARCH_SCHEMA,
        fn=lambda query: brave_web_search(query),
        formatter=_fmt_brave_search,
    ),
    "visuals_core": ToolEntry(
        schema=VISUALS_CORE_SCHEMA,
        fn=_visuals_core,
    ),
    "get_market_prices": ToolEntry(
        schema=YAHOO_FINANCE_SCHEMA,
        fn=get_market_prices,
    ),
}


# ── Registry builders ─────────────────────────────────────────────────────────

def get_tools() -> dict[str, ToolEntry]:
    """
    Return the tool registry (alias for build_tool_registry).
    Used by skill_command in llmcord.py.
    """
    return build_tool_registry()


def build_tool_registry() -> dict[str, ToolEntry]:
    """
    Return a fully-wired tool registry.
    web_search uses Brave API (requires BRAVE_API_KEY in .env). Works with all providers.
    """
    logging.info("ToolRegistry: web_search → Brave API (works with all providers)")
    return dict(_ENTRIES)


def build_brave_registry() -> dict[str, ToolEntry]:
    """
    Convenience: build a tool registry (Brave for web_search).
    Used by OpenAI/OpenRouter and Ollama providers.
    """
    return build_tool_registry()


# ── OpenAI / OpenRouter schema helper ────────────────────────────────────────

def get_openai_tools(tool_names: List[str] | None) -> List[dict[str, Any]]:
    """
    Return OpenAI-format schema dicts for the given tool names.
    Used for chat.completions.create(tools=...) with OpenAI/OpenRouter.
    """
    if not tool_names:
        return []
    return [_ENTRIES[n].schema for n in tool_names if n in _ENTRIES]


# ── Result formatter ──────────────────────────────────────────────────────────

def format_tool_result(entry: ToolEntry, result: Any, args: dict) -> str:
    """Format a tool result via the entry's formatter, or fall back to str()."""
    if entry.formatter:
        try:
            return entry.formatter(result, args)
        except Exception as e:
            logging.warning("Tool formatter failed: %s", e)
    return str(result)


# ── Tool executor (shared by Ollama + OpenAI paths) ───────────────────────────

def execute_tool_call(
    name: str,
    args: dict,
    registry: dict[str, ToolEntry],
    max_chars: int = 8000,
) -> str:
    """
    Execute a tool call by name and return the formatted result string.
    Used by both OllamaService and the OpenAI tool call loop in llmcord.py.
    """
    entry = registry.get(name)
    if not entry or not entry.fn:
        logging.warning("execute_tool_call: unknown or unavailable tool '%s'", name)
        return f"Tool '{name}' is not available."
    logging.info("execute_tool_call: '%s' args=%s", name, args)
    try:
        result = entry.fn(**args)
    except Exception as e:
        logging.error("execute_tool_call: tool '%s' failed: %s", name, e)
        return f"Tool error: {e}"
    return format_tool_result(entry, result, args)[:max_chars]
