# tool-system Specification

## Purpose
Defines the tool system including ToolEntry contract, registry, available tools, how to add new tools, and tool-specific requirements.

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
The system SHALL provide web_search, visuals_core, get_market_prices, google_tools, and get_weather.

#### Scenario: Available tools
- **WHEN** tools are needed
- **THEN** these tools are registered and callable

### Requirement: Adding new tools
New tools SHALL be added by creating a Python file with schema, function, and formatter.

#### Scenario: Add new tool
- **WHEN** developer creates tool file and registers it
- **THEN** tool is available via get_openai_tools and execute_tool_call

### Requirement: get_openai_tools returns tool schemas
The system SHALL provide `get_openai_tools(tool_names)` that returns OpenAI-format schemas.

#### Scenario: Get tools
- **WHEN** called with ["web_search", "visuals_core"]
- **THEN** returns list of OpenAI tool schemas

### Requirement: execute_tool_call runs tool by name
The system SHALL provide `execute_tool_call(name, args, registry, max_chars)` that runs the tool.

#### Scenario: Execute
- **WHEN** called with tool name and args
- **THEN** runs entry.fn(**args) and returns formatted result

### Requirement: get_market_prices handles incomplete data gracefully
The `get_market_prices` tool SHALL handle cases where Yahoo Finance returns NaN or missing data without crashing.

#### Scenario: NaN close price
- **WHEN** Yahoo Finance returns NaN for the close price
- **THEN** the tool SHALL output "[unavailable]" instead of crashing
- **AND** continue processing other tickers

#### Scenario: Missing previous day data
- **WHEN** there is only 1 day of trading data available
- **THEN** the tool SHALL display the close price without % change
- **AND** indicate that previous day data is unavailable

#### Scenario: Empty data with retry
- **WHEN** Yahoo Finance returns empty data on first attempt
- **THEN** the tool SHALL retry up to 3 times with exponential backoff
- **AND** if all retries fail, return clear error message with ticker symbol

### Requirement: get_market_prices provides clear error messages
The tool SHALL return human-readable error messages that the LLM can understand.

#### Scenario: Invalid ticker
- **WHEN** an invalid ticker symbol is provided
- **THEN** the tool SHALL return "{ticker}: no data returned (check ticker symbol)"

#### Scenario: Network error
- **WHEN** yfinance encounters a network error
- **THEN** the tool SHALL return "{ticker}: error — {error message}"

### Requirement: get_market_prices schema has correct defaults
The tool SHALL have a schema that prevents insufficient data retrieval.

#### Scenario: Days parameter default
- **WHEN** the LLM calls the tool without specifying days
- **THEN** the tool SHALL default to 10 days of historical data

#### Scenario: Days parameter minimum
- **WHEN** the LLM specifies days less than 5
- **THEN** the tool SHALL reject the value and use 5 instead

### Requirement: get_market_prices returns accurate price data
The tool SHALL return reliable and fresh price data.

#### Scenario: Missing historical data
- **WHEN** yfinance historical data is missing recent trading days
- **THEN** the tool SHALL use the info endpoint to get previous close
- **AND** calculate percentage from the reliable previous close value

#### Scenario: Intraday price during market hours
- **WHEN** Taiwan market is open (9:00-13:30 TW)
- **THEN** the tool SHALL fetch intraday 15-minute data
- **AND** return current price instead of previous close
- **AND** append "*" marker to indicate intraday price

#### Scenario: Market closed
- **WHEN** Taiwan market is closed
- **THEN** the tool SHALL use daily close price from historical data
- **AND** not append any marker

### Requirement: get_market_prices LLM behavior
The tool output SHALL be used exactly as returned without modification.

#### Scenario: Percentage values
- **WHEN** the tool returns percentage values
- **THEN** the LLM SHALL use the exact values without modification
- **AND** not add decimals, recalculate, or alter the values

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
    "get_weather": ToolEntry(...),
}
```

## Available Tools

| Tool | File | Function | Description |
|------|------|----------|-------------|
| web_search | web_search.py | brave_web_search(query) | Brave API web search |
| visuals_core | visuals_core.py | generate_visualization(viz_type, title, **kwargs) | ASCII/Markdown charts |
| get_market_prices | yahoo_finance.py | get_market_prices(tickers, days) | Yahoo Finance prices |
| google_tools | google_tools.py | google_tools_wrapper(action, ...) | Gmail + Calendar |
| get_weather | weather.py | get_weather(location, units) | OpenWeatherMap weather |

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

