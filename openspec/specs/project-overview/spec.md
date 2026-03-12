# project-overview Specification

## Purpose
TBD - created by archiving change document-llm-discord-bot. Update Purpose after archive.
## Requirements
### Requirement: Project structure follows modular layout
The project SHALL follow this directory structure:
- `llmcord.py` - main entry point
- `bot/config/` - configuration loader, validator, personas, tasks
- `bot/llm/` - LLM services and tools
- `bot/discord/` - Discord-specific error handling

#### Scenario: Project layout
- **WHEN** AI agent reads the project
- **THEN** it can navigate using: llmcord.py, bot/config/, bot/llm/, bot/discord/

### Requirement: Main entry point is llmcord.py
The system SHALL provide `llmcord.py` as the main entry point that starts the Discord bot.

#### Scenario: Run bot
- **WHEN** user runs `python llmcord.py`
- **THEN** the Discord bot starts and connects to Discord

### Requirement: Config loaded from YAML
The system SHALL load configuration from `config.yaml` (or path from CONFIG_PATH env).

#### Scenario: Load config
- **WHEN** bot starts
- **THEN** it loads and validates config.yaml

