# Tools — Architecture & Development Guide

This folder contains all bot tools. Each tool is self-contained: one `.py` file
with the callable + schema, and one `skills/*.md` file with the AI-readable skill doc.

Tools are **auto-discovered** — no manual registration needed!

## Structure

```
bot/llm/tools/
├── README.md              ← you are here (architecture + how to add tools)
├── __init__.py            ← lazy loading, auto-discovery
├── registry.py            ← tool registry with auto-discovery
├── web_search.py          ← web_search (Brave API) callable + schema
├── visuals_core.py        ← visuals_core callable + schema
├── yahoo_finance.py       ← stock/index prices from Yahoo Finance
├── google_tools.py        ← Gmail + Calendar (Google API)
├── weather.py             ← get_weather (Open-Meteo + Nominatim)
└── skills/
    ├── web_search.md      ← skill doc for web_search
    ├── visuals_core.md    ← skill doc for visuals_core
    ├── yahoo_finance.md   ← skill doc for get_market_prices
    ├── google_tools.md    ← skill doc for google_tools
    └── weather.md         ← skill doc for get_weather
```

## How tools flow

```
config.yaml tools: ["web_search"]
        │
        ▼
registry.py  auto-discovers tools from .py files
        │
        ▼
ToolEntry(schema, fn, formatter)
        ├── .schema  ──► sent to model via client.chat(tools=[...])
        ├── .fn      ──► called locally when model requests the tool
        └── .formatter ► formats raw result before sending back as tool message
```

## How to add a new tool

### 1. Create the tool file

`bot/llm/tools/my_tool.py`:

```python
from bot.llm.tools.registry import ToolEntry

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

# Auto-discovery exports
TOOL_NAME = "my_tool"
TOOL_ENTRY = ToolEntry(
    schema=MY_TOOL_SCHEMA,
    fn=my_tool,
)
```

> Use only JSON-native types (`string`, `integer`, `boolean`, `array`, `object`).
> Python's `Any` will crash Ollama's introspection.

### 2. Write a skill doc (optional but recommended)

`bot/llm/tools/skills/my_tool.md` — see existing skill files for the format.
The AI reads this to know when and how to invoke the tool correctly.

### 3. Add to config.yaml

```yaml
models:
  ollama/qwen3:14b:
    tools: ["web_search", "my_tool"]
```

**Done.** The tool is automatically discovered. No other files need to change.

## ToolEntry dataclass

```python
@dataclass
class ToolEntry:
    schema: dict                       # OpenAI-format schema sent to model
    fn: Callable | None = None         # called when model invokes the tool
    formatter: Callable | None = None  # optional: formatter(result, args) -> str
```

- If `formatter` is None, results are cast via `str()`.
