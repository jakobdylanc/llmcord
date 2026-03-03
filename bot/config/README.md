# Configuration Guide

This directory contains configuration files for personas and scheduled tasks.

## Directory Structure

- **`personas/`** – Personality/system prompt files for LLM models
- **`tasks/`** – Scheduled task definitions (cron-based automation)

---

## Personas

Personas are reusable personality/system prompt files that define how the bot behaves.

### Location
`bot/config/personas/`

### File Format
Supported formats: `.md`, `.txt`, `.yaml`, `.yml`

### Example Files
- `lao_pi.md` – A casual, friendly persona
- `stock_market_analyst.md` – A financial analysis persona

### Creating a Persona

Create a new file, e.g., `my_persona.md`:

```markdown
# My Custom Persona

You are a helpful assistant with a specific personality.
- Be concise and direct
- Use emojis occasionally
- Focus on practical advice
```

### Using a Persona

**In `config.yaml` (global or per-model):**
```yaml
# Global persona
persona: lao_pi

# Or per-model
models:
  openrouter/openrouter/free:
    persona: stock_market_analyst
```

**In scheduled tasks:**
```yaml
scheduled_tasks:
  my_task:
    persona: lao_pi
```

### Priority
If both `persona` and `system_prompt` are set, **persona takes priority**.

---

## Scheduled Tasks

Scheduled tasks run automatically on a cron schedule and send results to Discord channels or DMs.

### Location
`bot/config/tasks/`

### File Format
YAML (`.yaml` or `.yml`)

### Example Task File

Create `bot/config/tasks/stock_market_check.yaml`:

```yaml
name: stock_market_check
enabled: true
cron: "30 0 * * 1-5"  # 12:30 AM Mon-Fri
user_id: 123456789012345678
model: "ollama/qwen3:14b"
persona: stock_market_analyst
tools:
  - "web_search"
  - "web_fetch"
  - "visuals_core"
fallback_models:
  - "openrouter/openrouter/free"
prompt: "Summarize overnight moves in Taiwan and US markets."
system_prompt: "Reply in Traditional Chinese. Use web search for current data."
think: true  # Optional: for reasoning models
```

### Required Fields

| Field | Description |
|-------|-------------|
| `name` | Unique task identifier |
| `enabled` | `true` or `false` |
| `cron` | Cron schedule (see below) |
| `model` | Model to use (must exist in `config.yaml`) |
| `prompt` | The message to send to the LLM |
| `channel_id` OR `user_id` | Where to send results (use one, not both) |

### Optional Fields

| Field | Description |
|-------|-------------|
| `persona` | Persona file name (without extension) |
| `system_prompt` | Override system prompt for this task |
| `tools` | List of tools (Ollama only): `web_search`, `web_fetch`, `visuals_core` |
| `think` | Enable reasoning mode (Ollama only) |
| `fallback_models` | Fallback models if primary fails |

### Cron Format

`minute hour day month day_of_week`

**Examples:**
- `"0 9 * * *"` – Every day at 9:00 AM
- `"0 9 * * 1-5"` – Mon-Fri at 9:00 AM
- `"0 */2 * * *"` – Every 2 hours
- `"30 6 * * 0"` – Sunday at 6:30 AM

See [crontab.guru](https://crontab.guru) for help.

### How Tasks Are Loaded

1. All `*.yaml` files in `bot/config/tasks/` are automatically loaded
2. Tasks are merged with any entries under `scheduled_tasks` in `config.yaml`
3. File-based tasks take priority over inline tasks with the same name

### Example: Email Check Task

`bot/config/tasks/email_check.yaml`:
```yaml
name: email_check
enabled: true
cron: "0 9 * * *"  # 9:00 AM daily
channel_id: 987654321098765432
model: "open-webui/gmail-checker"
prompt: "Summarize my recent emails"
fallback_models:
  - "groq/mixtral-8x7b-32768"
```

---

## Tips

- **Keep personas focused** – One clear personality per file
- **Use descriptive task names** – Makes scheduling easier
- **Test cron expressions** – Use [crontab.guru](https://crontab.guru) to verify
- **Set fallback models** – Ensures tasks complete even if primary model fails
- **Use web_search for current data** – Perfect for market updates, news summaries, etc.
