# configuration-system Specification

## Purpose
TBD - created by archiving change document-llm-discord-bot. Update Purpose after archive.
## Requirements
### Requirement: Config schema defines required structure
The config SHALL have these top-level keys:
- `providers`: LLM provider configurations
- `models`: Model configurations with provider/model format
- `bot_token`: Discord bot token
- Optional: `persona`, `system_prompt`, `fallback_models`, `scheduled_tasks`, `permissions`

#### Scenario: Config structure
- **WHEN** AI reads config.yaml
- **THEN** it can parse: providers, models, bot_token, persona, system_prompt, fallback_models, scheduled_tasks, permissions

### Requirement: Config validation ensures required fields
The system SHALL validate config and fail startup if required fields are missing.

#### Scenario: Validation
- **WHEN** config is loaded
- **THEN** validator.py checks: providers exists, models exists, each model has valid provider

### Requirement: Persona loaded from file
The system SHALL load personas from `bot/config/personas/<name>.(md|txt|yaml|yml)`.

#### Scenario: Load persona
- **WHEN** persona: bao is configured
- **THEN** loads bot/config/personas/bao.md

