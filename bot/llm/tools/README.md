# Tools — Architecture & Development Guide

This folder contains all bot tools. Each tool is self-contained: one `.py` file
with the callable + schema, and one `skills/*.md` file with the AI-readable skill doc.

## Structure

```
bot/llm/tools/
├── README.md              ← you are here (architecture + how to add tools)
├── registry.py            ← single source of truth: ToolEntry registry
├── web_search.py          ← web_search + web_fetch callable + schema
├── visuals_core.py        ← visuals_core callable + schema
└── skills/
    ├── web_search.md      ← OpenClaw skill doc for web_search / web_fetch
    └── visuals_core.md    ← OpenClaw skill doc for visuals_core
```

## How tools flow

```
config.yaml tools: ["web_search"]
        │
        ▼
registry.py  _ENTRIES["web_search"]  ←── ToolEntry(schema, fn, formatter)
        ├── .schema  ──► sent to model via client.chat(tools=[...])
        ├── .fn      ──► called locally when model requests the tool
        └── .formatter ► formats raw result before sending back as tool message
```

`ollama_service.py` calls `build_tool_registry(client)` — it has **zero hardcoded
tool names**. Adding a tool never requires touching it.

## How to add a new tool

### 1. Create the tool file

`bot/llm/tools/my_tool.py`:

```python
def my_tool(param1: str, param2: int = 0) -> str:
    """Your tool logic here."""
    return str(result)

MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does.",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
            },
            "required": ["param1"],
        },
    },
}
```

> Use only JSON-native types (`string`, `integer`, `boolean`, `array`, `object`).
> Python's `Any` will crash Ollama's introspection.

### 2. Register in registry.py

```python
from .my_tool import my_tool, MY_TOOL_SCHEMA

_ENTRIES: dict[str, ToolEntry] = {
    ...
    "my_tool": ToolEntry(schema=MY_TOOL_SCHEMA, fn=my_tool),
}
```

### 3. Write a skill doc

`bot/llm/tools/skills/my_tool.md` — see existing skill files for the format.
The AI reads this to know when and how to invoke the tool correctly.

### 4. Add to config.yaml

```yaml
models:
  ollama/qwen3:14b:
    tools: ["web_search", "my_tool"]
```

**Done.** No other files need to change.

## ToolEntry dataclass

```python
@dataclass
class ToolEntry:
    schema: dict                       # OpenAI-format schema sent to model
    fn: Callable | None = None         # called when model invokes the tool
    formatter: Callable | None = None  # optional: formatter(result, args) -> str
```

- `fn=None` is used for Ollama-native tools (web_search, web_fetch) whose fn is
  bound to an Ollama Client at runtime by `build_tool_registry(client)`.
- If `formatter` is None, results are cast via `str()`.
