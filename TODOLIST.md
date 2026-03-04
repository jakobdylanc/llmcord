# TODO List

Backlog of improvements, bugs, and ideas for the bot.
Check items off as they're completed.

**Reference:** [spec.md](spec.md) (APIs), [report.md](report.md) (status), [SKILLS.md](SKILLS.md) and [bot/llm/tools/README.md](bot/llm/tools/README.md) (tools), [bot/config/README.md](bot/config/README.md) (personas & tasks).

---

## ✅ Done

- [x] Config loader & validator (`bot/config/loader.py`, `bot/config/validator.py`)
- [x] Ollama provider with tool calling
- [x] Brave API web search for all providers (including Ollama)
- [x] Tool schemas decoupled from tool callables — each tool file owns both
- [x] `registry.py` as single source of truth for all tool schemas
- [x] `build_tool_registry()` — single registry; web_search uses Brave for all providers
- [x] `ToolEntry` dataclass (schema + fn + formatter) — uniform tool contract
- [x] `SKILLS.md` — human + AI reference for adding/using tools
- [x] Fallback model chain — primary → model fallbacks → global fallbacks
- [x] Per-model timeout (`RESPONSE_TIMEOUT_SECONDS`) with fallback on expiry
- [x] Graceful tool call parse error recovery (retries without tools on Ollama 500)
- [x] Persona system (`bot/config/personas/`)
- [x] Scheduled tasks with cron (`bot/config/tasks/`)
- [x] `/model` slash command with autocomplete
- [x] `/clear` slash command
- [x] Streamed responses with embed color (green = complete, orange = streaming)
- [x] `<think>` tag stripping for reasoning models
- [x] Admin error notifications via Discord DM
- [x] Multi-turn tool calling for OpenAI/OpenRouter (`run_openai_with_tools` in llmcord.py; tool call → result → continue loop with Brave registry)

---

## 🔧 In Progress / Next Up

- [ ] **Tool result timeout** — individual tool calls can hang;
      add per-tool timeout wrapping `entry.fn(**args)` in `OllamaService.run()`
- [ ] **Multi Recivers support in Scheduled Task** — Support DM to user and send Message to a channel at the same time when both `user_id` and `channel_id` are gave in the task.yaml.
- [ ] **Discord slash cmmands** — Support /skill, /task in Discord slash commands.

---

## 💡 Ideas / Backlog

- [ ] **`/tools` slash command** — list available tools and their descriptions, sourced
      from `SKILLS.md` or `registry.py` at runtime
- [ ] **Hot-reload tools** — watch `registry.py` for changes and reload without restart
- [ ] **Tool usage logging** — log which tools were called per conversation to a file or
      Discord channel for debugging
- [ ] **Document env vars** — single place for CONFIG_PATH, BRAVE_API_KEY, OLLAMA_API_KEY, bot_token, etc. (e.g. README or .env.example)
- [ ] **CONTRIBUTING.md** — PR workflow, code style, how to run tests
- [ ] **`get_weather` tool** — weather lookup (e.g. Open-Meteo or provider API); implement and register in `registry.py` if desired
- [ ] **Per-task tool override in scheduled tasks** — tasks can already set `tools:` but
      there's no validation that the tool names exist; add a check in `bot/config/tasks.py`
- [ ] **Config validator for tools** — `bot/config/validator.py` should check that tool
      names in `config.yaml` match keys in `registry._ENTRIES`
- [ ] **Dockerfile improvements** — pin Python version, add health check
- [ ] **Per-user rate limiting** — limit messages or tool calls per user per minute to avoid abuse
- [ ] **Conversation export** — `/export` or button to export thread as markdown or JSON
- [ ] **.env.example** — committed template listing CONFIG_PATH, BRAVE_API_KEY, OLLAMA_API_KEY, bot_token
- [ ] **Streaming + tool calls for OpenAI** — optional streaming path that still handles tool calls (e.g. stream until tool_calls, then run tools and continue)
- [ ] **Metrics or simple dashboard** — count messages per model, tool usage, errors (log file or Discord summary)
- [ ] **Localization** — user-facing strings (e.g. error replies) in config or locale files instead of hardcoded zh-TW/en

---

## 🐛 Known Issues

- `qwen3:14b` with `think: true` bleeds reasoning tokens into tool call JSON, causing
  Ollama to return HTTP 500. Workaround is already in place (retry without tools), but
  the model is fundamentally unreliable for tool calling. Use `qwen2.5:14b` instead.
