---
name: gpt-discord-bot
description: Discord bot connecting to multiple LLM providers (Ollama, OpenRouter, OpenAI) with tool calling, personas, and scheduled tasks.
homepage: https://github.com/ckw1206/gpt-discord-bot
metadata: {"clawhub":{"emoji":"🤖","requires":{"bins":["python"]}}}
---

# gpt-discord-bot Skills

This file is the skill index for the bot. Each tool has its own skill doc under
`bot/llm/tools/skills/` in OpenClaw format — readable by both AI assistants and humans.

## Available Skills

| Skill | File | Provider | Description |
|-------|------|----------|-------------|
| 🌐 `web_search` / `web_fetch` | [`skills/web_search.md`](bot/llm/tools/skills/web_search.md) | Ollama | Search the web and fetch URLs |
| 📊 `visuals_core` | [`skills/visuals_core.md`](bot/llm/tools/skills/visuals_core.md) | Any | ASCII/Markdown charts, tables, timelines |

## Using skills in config.yaml

```yaml
models:
  ollama/qwen3:14b:
    tools: ["web_search", "web_fetch", "visuals_core"]
```

Tool names must match keys in `bot/llm/tools/registry.py`.

## Adding a new skill

See [`bot/llm/tools/README.md`](bot/llm/tools/README.md) for the full guide.
Short version:
1. Create `bot/llm/tools/my_tool.py` (callable + schema)
2. Register in `bot/llm/tools/registry.py`
3. Write `bot/llm/tools/skills/my_tool.md` (OpenClaw format)
4. Add to `config.yaml` under the model's `tools:` list
