"""
bot/llm/tools/registry.py

Dynamic tool registry with auto-discovery from bot/llm/tools/ directory.

Adding a new tool only requires:
  1. Create bot/llm/tools/my_tool.py with TOOL_NAME and TOOL_ENTRY
  2. Add the tool name to config.yaml under the model's `tools:` list

No manual registration needed - tools are auto-discovered!
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
from typing import Any, Callable, List

from ._types import ToolEntry
from .web_search import (
    WEB_SEARCH_SCHEMA,
    brave_web_search,
    format_brave_results,
)
from .visuals_core import VISUALS_CORE_SCHEMA, generate_visualization
from .yahoo_finance import YAHOO_FINANCE_SCHEMA, get_market_prices
from .google_tools import GOOGLE_TOOLS_SCHEMA, google_tools_wrapper


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


# ── Static entries (legacy format - for backward compatibility) ───────────────

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
    "google_tools": ToolEntry(
        schema=GOOGLE_TOOLS_SCHEMA,
        fn=google_tools_wrapper,
    ),
}


# ── Dynamic tool discovery ────────────────────────────────────────────────────

_TOOLS_DIR = os.path.dirname(__file__)


def _discover_tools() -> dict[str, ToolEntry]:
    """
    Auto-discover tools from Python files in the tools directory.
    Each tool file should define TOOL_NAME and TOOL_ENTRY.
    
    Error handling: malformed modules are skipped with warnings.
    """
    discovered: dict[str, ToolEntry] = {}
    
    for filename in os.listdir(_TOOLS_DIR):
        if not filename.endswith('.py') or filename.startswith('_'):
            continue
        
        module_name = filename[:-3]  # remove .py
        if module_name in ('registry',):  # skip self
            continue
        
        try:
            # Dynamic import
            module = importlib.import_module(f'.{module_name}', package='bot.llm.tools')
            
            # Look for TOOL_NAME and TOOL_ENTRY
            tool_name = getattr(module, 'TOOL_NAME', None)
            tool_entry = getattr(module, 'TOOL_ENTRY', None)
            
            if tool_name and tool_entry:
                if not isinstance(tool_entry, ToolEntry):
                    logging.warning(
                        f"Tool '{module_name}': TOOL_ENTRY is not a ToolEntry instance, skipping"
                    )
                    continue
                discovered[tool_name] = tool_entry
                logging.info(f"Discovered tool: {tool_name} from {filename}")
            else:
                logging.debug(
                    f"Tool '{module_name}': missing TOOL_NAME or TOOL_ENTRY, skipping"
                )
                
        except Exception as e:
            logging.warning(f"Failed to load tool module '{module_name}': {e}")
    
    return discovered


# Cache for discovered tools
_discovered_tools: dict[str, ToolEntry] | None = None


def _get_discovered_tools() -> dict[str, ToolEntry]:
    """Get cached discovered tools, or discover them on first call."""
    global _discovered_tools
    if _discovered_tools is None:
        _discovered_tools = _discover_tools()
        # Log on first discovery only (not on every build_tool_registry call)
        tool_names = list(_discovered_tools.keys())
        logging.info(f"ToolRegistry: {len(_discovered_tools)} tools available: {tool_names}")
    return _discovered_tools


def reload_tools() -> None:
    """Force reload of tool registry (useful for testing)."""
    global _discovered_tools
    _discovered_tools = None
    _get_discovered_tools()
    logging.info("Tool registry reloaded")


# ── Registry builders ─────────────────────────────────────────────────────────

def get_tools() -> dict[str, ToolEntry]:
    """
    Return the tool registry (alias for build_tool_registry).
    Used by skill_command in llmcord.py.
    """
    return build_tool_registry()


def build_tool_registry() -> dict[str, ToolEntry]:
    """
    Return a fully-wired tool registry with auto-discovered tools.
    Combines legacy static entries with dynamically discovered tools.
    Tools from files take precedence over legacy entries.
    Keywords are merged from skill.md if available.
    """
    # Start with discovered tools (they take precedence)
    registry = dict(_get_discovered_tools())
    
    # Add legacy entries for any tools not already discovered
    for name, entry in _ENTRIES.items():
        if name not in registry:
            registry[name] = entry
    
    return registry


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
    registry = build_tool_registry()
    return [registry[n].schema for n in tool_names if n in registry]


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
