# Config Spec

## Purpose
Defines the configuration file structure for the Discord bot, including providers, models, personas, tasks, and permissions.

## Requirements
### Requirement: Config schema defines required structure
The config SHALL have these top-level keys:
- `bot_token`: Discord bot token
- `client_id`: Discord client ID (optional)
- `providers`: LLM provider configurations
- `models`: Model configurations with provider/model format
- Optional: `persona`, `system_prompt`, `fallback_models`, `scheduled_tasks`, `permissions`, `status_message`, `max_text`, `max_images`, `max_messages`, `use_plain_responses`, `show_embed_color`, `allow_dms`

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

## Config File

Default: `config.yaml` (or path from CONFIG_PATH env)

## Top-Level Keys

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| bot_token | string | Yes | Discord bot token |
| client_id | string | No | Discord client ID (OAuth2) |
| providers | object | Yes | LLM provider configurations |
| models | object | Yes | Model configurations |
| persona | string | No | Default persona name |
| system_prompt | string | No | Global system prompt |
| fallback_models | array | No | Global fallback model list |
| scheduled_tasks | array | No | Inline task definitions |
| permissions | object | No | Access control |
| status_message | string | No | Bot status |
| max_text | number | No | Max chars per message (default: 100000) |
| max_images | number | No | Max images per message (default: 5) |
| max_messages | number | No | Max messages in chain (default: 25) |
| use_plain_responses | boolean | No | Use plaintext instead of embeds |
| show_embed_color | boolean | No | Show green/orange bar |
| allow_dms | boolean | No | Allow direct messages |

## Provider Config

```yaml
providers:
  ollama:
    base_url: "http://localhost:11434"
  openrouter:
    base_url: "https://openrouter.ai"
    api_key: "your-key"
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "your-key"
```

## Model Config

```yaml
models:
  openrouter/openrouter/free:
    persona: bao
    tools: ["web_search"]
    supports_tools: true
    fallback_models:
      - "ollama/qwen3:14b"

  ollama/qwen3:14b:
    tools: ["web_search", "visuals_core"]
    think: true
    system_prompt: "You are helpful."
```

## Persona

- Stored in `bot/config/personas/<name>.(md|txt|yaml|yml)`
- Loaded by `load_persona(name)` in personas.py
- Resolution: model persona → model system_prompt → global persona → global system_prompt

## Scheduled Task

```yaml
scheduled_tasks:
  - name: daily-stock-check
    cron: "0 9 * * *"
    model: openrouter/openrouter/free
    prompt: "Check stock prices for AAPL, GOOGL"
    channel_id: 123456789
    # or user_id: 123456789
```

## Validation

- `bot/config/validator.py` validates:
  - providers exists and has entries
  - models exists and each model has valid provider
  - scheduled_tasks have required fields (name, cron, model, prompt)
  - permissions.users/roles/channels have valid structure
- Raises `ConfigValidationError` on failure