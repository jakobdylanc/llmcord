## ADDED Requirements

### Requirement: ToolEntry dataclass defines tool contract
The system SHALL use ToolEntry with:
- `schema`: OpenAI-format tool schema
- `fn`: Callable that executes the tool
- `formatter`: Function to format results for LLM

#### Scenario: ToolEntry
- **WHEN** AI reads registry.py
- **THEN** it understands: ToolEntry(schema, fn, formatter)

### Requirement: Tool registry in registry.py
The system SHALL maintain `_ENTRIES` dict mapping tool name → ToolEntry.

#### Scenario: Registry
- **WHEN** AI reads registry.py
- **THEN** it can find: web_search, visuals_core, get_market_prices, google_tools

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