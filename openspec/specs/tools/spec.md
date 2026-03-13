# Tool Spec

## Purpose
Defines the tool system including ToolEntry contract, registry, available tools, and how to add new tools.

## Requirements
### Requirement: ToolEntry contract
Each tool SHALL have schema, fn (function), and formatter components.

#### Scenario: ToolEntry
- **WHEN** tool is registered
- **THEN** it has schema, fn, and formatter properties

### Requirement: Tool registry
The system SHALL maintain a registry of available tools accessible by name.

#### Scenario: Registry
- **WHEN** tool name is requested
- **THEN** registry returns corresponding ToolEntry

### Requirement: Available tools
The system SHALL provide web_search, visuals_core, get_market_prices, and google_tools.

#### Scenario: Available tools
- **WHEN** tools are needed
- **THEN** these tools are registered and callable

### Requirement: Adding new tools
New tools SHALL be added by creating a Python file with schema, function, and formatter.

#### Scenario: Add new tool
- **WHEN** developer creates tool file and registers it
- **THEN** tool is available via get_openai_tools and execute_tool_call

## ToolEntry Contract

```python
@dataclass
class ToolEntry:
    schema: dict      # OpenAI-format tool schema
    fn: Callable      # Function(args) -> result
    formatter: Callable  # Format result for LLM consumption
```

## Registry

Location: `bot/llm/tools/registry.py`

```python
_ENTRIES: dict[str, ToolEntry] = {
    "web_search": ToolEntry(...),
    "visuals_core": ToolEntry(...),
    "get_market_prices": ToolEntry(...),
    "google_tools": ToolEntry(...),
}
```

## Available Tools

| Tool | File | Function | Description |
|------|------|----------|-------------|
| web_search | web_search.py | brave_web_search(query) | Brave API web search |
| visuals_core | visuals_core.py | generate_visualization(viz_type, title, **kwargs) | ASCII/Markdown charts |
| get_market_prices | yahoo_finance.py | get_market_prices(tickers, days) | Yahoo Finance prices |
| google_tools | google_tools.py | google_tools_wrapper(action, count, label_id, message_id, calendar_id) | Gmail + Calendar |

## APIs

### get_openai_tools(tool_names: list[str]) -> list[dict]
Returns OpenAI-format tool schemas for given tool names.

### execute_tool_call(name: str, args: dict, registry: dict, max_chars: int) -> str
Executes tool by name with args, returns formatted result (truncated to max_chars).

### build_tool_registry() -> dict[str, ToolEntry]
Returns full registry with all tools.

## Adding a New Tool

1. Create `bot/llm/tools/my_tool.py` with:
   - `MY_TOOL_SCHEMA` - OpenAI-format schema
   - `my_function(args)` - Tool implementation
   - `format_result(result)` - Optional formatter

2. Register in `bot/llm/tools/registry.py`:
   ```python
   _ENTRIES["my_tool"] = ToolEntry(
       schema=MY_TOOL_SCHEMA,
       fn=my_function,
       formatter=format_result,
   )
   ```

3. Add to config.yaml model:
   ```yaml
   models:
     ollama/qwen3:14b:
       tools: ["my_tool"]
   ```

4. Create skill doc in `bot/llm/tools/skills/my_tool.md`