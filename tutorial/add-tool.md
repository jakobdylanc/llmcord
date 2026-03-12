# Add a New Tool

This guide explains how to add a new tool to the bot.

## Overview

Tools allow the LLM to perform actions like web search, get stock prices, or access external APIs. Tools are executed bot-side - the model requests a tool call, the bot runs it, and feeds the result back.

## Steps

### 1. Create Tool File

Create `bot/llm/tools/my_tool.py`:

```python
# bot/llm/tools/my_tool.py
from typing import Any

# OpenAI-format tool schema
MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What the tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "Description"}
            },
            "required": ["arg1"]
        }
    }
}

def my_tool_function(arg1: str) -> str:
    """Execute the tool."""
    # Your logic here
    return f"Result: {arg1}"

def format_my_tool_result(result: Any) -> str:
    """Format result for LLM."""
    return str(result)
```

### 2. Register in Registry

Add to `bot/llm/tools/registry.py`:

```python
from .my_tool import MY_TOOL_SCHEMA, my_tool_function, format_my_tool_result

_ENTRIES: dict[str, ToolEntry] = {
    # ... existing tools ...
    "my_tool": ToolEntry(
        schema=MY_TOOL_SCHEMA,
        fn=my_tool_function,
        formatter=format_my_tool_result,
    ),
}
```

### 3. Enable in Config

```yaml
models:
  ollama/qwen3:14b:
    tools: ["my_tool"]
```

### 4. Create Skill Doc (Optional)

Create `bot/llm/tools/skills/my_tool.md` in OpenClaw format for better LLM usage.

## ToolEntry Contract

```python
@dataclass
class ToolEntry:
    schema: dict      # OpenAI-format tool schema
    fn: Callable      # Function(args) -> result
    formatter: Callable  # Format result for LLM
```

## Technical Details

For AI agents, see [Tool Spec](../openspec/specs/tools/spec.md) for full API details.