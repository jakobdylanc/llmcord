# Project Spec

## Purpose
Defines the overall project structure, modules, and data models for the Discord LLM bot.

## Requirements
### Requirement: Module structure defines components
The project SHALL have distinct modules for entry point, config, LLM, tools, and Discord integration.

#### Scenario: Module organization
- **WHEN** codebase is examined
- **THEN** it contains: llmcord.py (entry), bot/config/* (config), bot/llm/* (LLM), bot/llm/tools/* (tools), bot/discord/* (Discord)

### Requirement: Data models define structures
The system SHALL define data models for Config, ToolEntry, and ScheduledTask.

#### Scenario: Data models
- **WHEN** code is parsed
- **THEN** Config, ToolEntry, and ScheduledTask structures are available

## Overview

Discord bot that streams LLM responses, supports multiple providers, tools, personas, and scheduled tasks.

## Module Structure

| Module | File | Purpose |
|--------|------|---------|
| Entry | `llmcord.py` | Main entry point, Discord client, message handling |
| Config | `bot/config/loader.py` | Load and validate config.yaml |
| Config | `bot/config/validator.py` | Validate config structure |
| Config | `bot/config/personas.py` | Load persona files |
| Config | `bot/config/tasks.py` | Load scheduled tasks |
| LLM | `bot/llm/ollama_service.py` | Ollama provider with tool calling |
| Tools | `bot/llm/tools/registry.py` | Tool registry (ToolEntry, get_openai_tools, execute_tool_call) |
| Discord | `bot/discord/errors.py` | Admin error notifications |

## Data Models

### Config (dict)
```python
{
    "bot_token": str,
    "client_id": str,
    "providers": {provider_name: {"base_url": str, "api_key": str}},
    "models": {model_key: {"tools": list, "persona": str, "fallback_models": list}},
    "persona": str,  # optional
    "system_prompt": str,  # optional
    "fallback_models": list,  # optional
    "scheduled_tasks": list,  # optional
    "permissions": dict,  # optional
}
```

### ToolEntry (dataclass)
```python
@dataclass
class ToolEntry:
    schema: dict      # OpenAI-format tool schema
    fn: Callable      # Function to execute
    formatter: Callable  # Format result for LLM
```

### ScheduledTask (dict)
```python
{
    "name": str,
    "cron": str,
    "model": str,
    "prompt": str,
    "channel_id": int,  # optional
    "user_id": int,  # optional
    "tools": list,  # optional
    "persona": str,  # optional
}
```

## Key APIs

| Function | Module | Signature |
|----------|--------|-----------|
| get_config | bot/config/loader.py | `get_config(path) -> dict` |
| validate_config | bot/config/validator.py | `validate_config(cfg, path) -> None` |
| load_persona | bot/config/personas.py | `load_persona(name) -> str` |
| load_scheduled_tasks | bot/config/tasks.py | `load_scheduled_tasks(config) -> list` |
| OllamaService.run | bot/llm/ollama_service.py | `run(messages, model, enable_tools, think, max_tool_chars) -> tuple` |
| build_tool_registry | bot/llm/tools/registry.py | `build_tool_registry() -> dict[str, ToolEntry]` |
| get_openai_tools | bot/llm/tools/registry.py | `get_openai_tools(tool_names) -> list[dict]` |
| execute_tool_call | bot/llm/tools/registry.py | `execute_tool_call(name, args, registry, max_chars) -> str` |
| notify_admin_error | bot/discord/errors.py | `notify_admin_error(bot, config, error, context) -> None` |

## Current State (from TODOLIST.md)

### Done
- Config loader & validator
- Ollama provider with tool calling
- Brave API web search for all providers
- Tool schemas decoupled from tool callables
- ToolEntry dataclass
- SKILLS.md
- Fallback model chain
- Per-model timeout with fallback
- Graceful tool call parse error recovery
- Persona system
- Scheduled tasks with cron
- /model, /clear slash commands
- Streamed responses with embed color
- `<think>` tag stripping
- Admin error notifications
- Multi-turn tool calling for OpenAI/OpenRouter
- /skill, /task, /refresh, /persona commands
- Google Tools (Gmail + Calendar)

### In Progress
- Tool result timeout
- Multi Receivers support in Scheduled Task

### Backlog
- /tools slash command
- Hot-reload tools
- Tool usage logging
- Document env vars
- CONTRIBUTING.md
- get_weather tool
- Per-task tool override validation
- Config validator for tools
- Dockerfile improvements
- Per-user rate limiting
- Conversation export
- .env.example
- Streaming + tool calls for OpenAI
- Metrics dashboard
- Localization