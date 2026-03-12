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
| 🌐 `web_search` | [`skills/web_search.md`](bot/llm/tools/skills/web_search.md) | Any | Search the web (Brave API) |
| 📊 `visuals_core` | [`skills/visuals_core.md`](bot/llm/tools/skills/visuals_core.md) | Any | ASCII/Markdown charts, tables, timelines |
| 📈 `get_market_prices` | [`skills/yahoo_finance.md`](bot/llm/tools/skills/yahoo_finance.md) | Any | Stock/index prices from Yahoo Finance |
| 📧 `google_tools` | [`skills/google_tools.md`](bot/llm/tools/skills/google_tools.md) | Any | Gmail + Calendar (read-only) |

## Using skills in config.yaml

```yaml
models:
  ollama/qwen3:14b:
    tools: ["web_search", "visuals_core"]
```

Tool names are auto-discovered from `bot/llm/tools/` directory.

## Adding a new skill

1. Create `bot/llm/tools/my_tool.py` with:
   ```python
   from bot.llm.tools.registry import ToolEntry

   def my_tool(arg1: str) -> str:
       return f"Result: {arg1}"

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
               "required": ["arg1"],
           },
       },
   }

   TOOL_NAME = "my_tool"
   TOOL_ENTRY = ToolEntry(
       schema=MY_TOOL_SCHEMA,
       fn=my_tool,
   )
   ```

2. Add to `config.yaml` under the model's `tools:` list

That's it! The tool will be automatically discovered.
