"""
bot/llm/ollama_service.py

Ollama LLM runner with tool-calling support.
All tool definitions live in bot/llm/tools/registry.py — nothing is hardcoded here.

Skill docs (bot/llm/tools/skills/*.md) are automatically loaded and injected into
the system prompt when their corresponding tools are enabled, so the model always
knows when and how to call each tool correctly.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ollama import Client

from .tools.registry import build_tool_registry, execute_tool_call

_SKILLS_DIR = Path(__file__).parent / "tools" / "skills"


def _load_skill_doc(tool_name: str) -> str:
    """Load skill doc for a tool, stripping YAML frontmatter. Returns '' if not found."""
    candidates = [
        _SKILLS_DIR / f"{tool_name}.md",
        *[p for p in _SKILLS_DIR.glob("*.md") if tool_name.startswith(p.stem)],
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()
    return ""


def _build_skill_context(tool_names: List[str]) -> str:
    """Load and deduplicate skill docs for all enabled tools."""
    seen: set[str] = set()
    docs: list[str] = []
    for name in tool_names:
        candidates = [
            _SKILLS_DIR / f"{name}.md",
            *[p for p in _SKILLS_DIR.glob("*.md") if name.startswith(p.stem)],
        ]
        skill_file = next((p for p in candidates if p.exists()), None)
        if not skill_file or skill_file.name in seen:
            continue
        seen.add(skill_file.name)
        doc = _load_skill_doc(name)
        if doc:
            docs.append(doc)
    return "\n\n---\n\n".join(docs)


def _inject_skills(messages: List[Dict], tool_names: List[str]) -> List[Dict]:
    """Append skill docs to the system message (or prepend one if absent)."""
    skill_context = _build_skill_context(tool_names)
    if not skill_context:
        return messages
    messages = list(messages)
    if messages and messages[0]["role"] == "system":
        messages[0] = {
            **messages[0],
            "content": messages[0]["content"].rstrip() + "\n\n" + skill_context,
        }
    else:
        messages.insert(0, {"role": "system", "content": skill_context})
    return messages


class OllamaService:
    def __init__(self, host: str):
        """
        host — Ollama server URL
        """
        api_key = os.getenv("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = Client(host=host, headers=headers)

        self._registry = build_tool_registry()

    # ── Main Chat Runner ────────────────────────────────────────────────────

    def run(
        self,
        messages: List[Dict[str, str]],
        model: str,
        enable_tools: List[str] | None = None,
        think: bool = False,
        max_tool_chars: int = 8000,
    ) -> Dict[str, Any]:

        enabled_tools = [n for n in (enable_tools or []) if n in self._registry]
        enabled_schemas = [self._registry[n].schema for n in enabled_tools]

        messages = _inject_skills(messages, enabled_tools)

        tool_outputs = []

        while True:
            try:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    tools=enabled_schemas,
                    think=think,
                )
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                
                # Handle connection errors with retry
                if "Connection" in error_type or "ECONNREFUSED" in error_str or "ETIMEDOUT" in error_str:
                    logging.warning(f"OllamaService: Connection failed - {error_type}: {error_str[:100]}")
                    raise ConnectionError(f"Failed to connect to Ollama at {self.host}. Please check that Ollama is running.") from e
                
                if enabled_schemas and "error parsing tool call" in error_str:
                    logging.warning(
                        "OllamaService: model produced unparseable tool call "
                        "(likely model incompatibility). Retrying without tools."
                    )
                    enabled_schemas = []
                    response = self.client.chat(
                        model=model, messages=messages, tools=[], think=think
                    )
                else:
                    raise

            messages.append(response.message)

            logging.info(
                "OllamaService: response content=%r, tool_calls=%s",
                response.message.content,
                bool(response.message.tool_calls),
            )
            if not response.message.tool_calls:
                break

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments
                formatted = execute_tool_call(name, args, self._registry, max_tool_chars)
                tool_outputs.append(formatted)
                messages.append(
                    {"role": "tool", "tool_name": name, "content": formatted}
                )

        return {
            "content": response.message.content or "",
            "thinking": getattr(response.message, "thinking", None),
            "tool_results": tool_outputs,
            "messages": messages,
        }