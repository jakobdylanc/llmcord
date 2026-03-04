## gpt-discord-bot

Discord bot that connects to multiple LLM providers (OpenRouter, Ollama, Open WebUI, etc.), supports tools, personas, and scheduled tasks.

### Documentation

| Document | Purpose |
|---------|---------|
| [spec.md](spec.md) | Technical spec: key APIs/functions, module layout, external APIs, architecture diagram. |
| [report.md](report.md) | Development status, phases, current architecture, how to maintain. |
| [TODOLIST.md](TODOLIST.md) | Done / in progress / backlog and links to other docs. |
| [SKILLS.md](SKILLS.md) | Skill index (tools the bot can use); see also `bot/llm/tools/skills/`. |
| [bot/config/README.md](bot/config/README.md) | Personas and scheduled tasks (format, fields, cron). |
| [bot/llm/tools/README.md](bot/llm/tools/README.md) | How to add a new tool. |

### Layout

- `llmcord.py` – main entrypoint (Discord bot, message handling, scheduling).
- `bot/` – modular package:
  - `bot/config/` – loader, validator, personas, tasks.
  - `bot/discord/errors.py` – admin notifications, slash-command error handling.
  - `bot/llm/` – Ollama service, tools registry, skill docs.

Run the bot:

```bash
python llmcord.py
# or
python -m bot.main
```

---

### Config basics (`config.yaml`)

- **Providers**: under `providers`, one key per provider (`openrouter`, `ollama`, `open-webui`, etc.).
- **Models**: under `models`, keys are `"provider/model-name"` (e.g. `openrouter/openrouter/free`).
- **Personas**:
  - Global: `persona: some_name` at top-level.
  - Model-level: `persona: some_name` under a model.
  - Task-level: `persona: some_name` inside a scheduled task.
  - `some_name` maps to a file under `bot/config/personas/some_name.(md|txt|yaml|yml)`.
- **Fallback models**:
  - Global: `fallback_models: [...]` at top-level.
  - Model-level: `fallback_models: [...]` under a model.
  - Task-level: `fallback_models: [...]` inside a task.

Resolution order: model fallbacks → global fallbacks.

For detailed persona and task config, see [`bot/config/README.md`](bot/config/README.md).

---

<h1 align="center">llmcord</h1>
<h3 align="center"><i>Talk to LLMs with your friends!</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

**Note:** This is a fork of [llmcord](https://github.com/jakobdylanc/llmcord) with added tool calling, Brave web search, skill docs, and scheduler support.

---

## Features

### Reply-based chat system
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation
- When DMing the bot, conversations continue automatically (no reply required)
- Branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ)
- Back-to-back messages from the same user are automatically chained together

---

### Tool calling — works with ALL providers

Tools are executed **bot-side**. Any model from any provider (OpenAI, OpenRouter, Ollama, etc.) can use them — the model requests a tool call, the bot runs it locally, and feeds the result back.

#### Available tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web (Brave API; works with all providers) |
| `visuals_core` | Generate ASCII/Markdown charts, tables, timelines, trees |
| `get_market_prices` | Yahoo Finance closing prices for tickers |

Web search uses the Brave API for all providers. Set `BRAVE_API_KEY` in `.env` to enable `web_search`.

Enable tools per model in `config.yaml`:

```yaml
models:
  openrouter/openrouter/free:
    tools: ["web_search", "visuals_core"]

  ollama/qwen3:14b:
    tools: ["web_search", "visuals_core"]
    think: true
```

#### Skill docs

Each tool has a skill doc under `bot/llm/tools/skills/` in [OpenClaw](https://clawhub.sh) format. When a tool is enabled for a model, its skill doc is automatically injected into the system prompt so the model knows exactly when and how to call it — no manual prompting needed.

To add a new tool, see [`bot/llm/tools/README.md`](bot/llm/tools/README.md).

---

### Scheduled tasks
Configure periodic tasks to run on a cron schedule.

- Send results to Discord channels or user DMs
- Per-task model, tools, persona, and system prompt
- Task-level tool/think overrides the model config
- File-based tasks in `bot/config/tasks/*.yaml` take priority over inline config

---

### Model switching with `/model`
![image](https://github.com/user-attachments/assets/568e2f5c-bf32-4b77-ab57-198d9120f3d2)

Supports remote providers: OpenAI, xAI, Google Gemini, Mistral, Groq, OpenRouter

Local providers: Ollama, LM Studio, vLLM — or any OpenAI-compatible API.

---

### Clear conversation with `/clear`
Resets conversation history and message cache. Useful when switching models or starting fresh.

---

### Discord slash commands

| Command | Description | Admin |
|:--------|:------------|:-----:|
| `/model` | Switch to a different model | ✓ |
| `/persona` | View or switch the current persona | ✓ |
| `/skill` | List available skills/tools | ✓ |
| `/task` | List tasks, toggle on/off, or run immediately | ✓ |
| `/clear` | Reset conversation history | ✓ |
| `/refresh` | Reload config, tasks, model, and persona | ✓ |

---

### And more
- Vision model support (image attachments)
- Text file attachments
- Customizable personas (system prompts)
- Streamed responses with completion indicator (green = done, orange = streaming)
- Automatic `<think>` tag stripping for reasoning models
- Per-model response timeout with automatic fallback
- Hot-reloading config (no restart needed)
- Fully asynchronous

---

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/ckw1206/gpt-discord-bot
   cd gpt-discord-bot
   ```

2. Copy and fill in config:
   ```bash
   cp config.yaml.template config.yaml
   ```

3. Create `.env` for secrets (see [TODOLIST.md](TODOLIST.md) for full env list; optional: add `.env.example`):
   ```
   BRAVE_API_KEY=your_key_here
   OLLAMA_API_KEY=optional
   # CONFIG_PATH=config.yaml   # optional override
   ```

4. Run:
   ```bash
   pip install -r requirements.txt
   python llmcord.py
   # or with Docker:
   docker compose up
   ```

---

## Config reference

### Discord settings

| Setting | Description |
|---------|-------------|
| `bot_token` | Discord bot token ([discord.com/developers/applications](https://discord.com/developers/applications)). Enable MESSAGE CONTENT INTENT. |
| `client_id` | Found under OAuth2 tab of your Discord bot. |
| `status_message` | Custom status shown on the bot's profile. Max 128 chars. |
| `max_text` | Max characters per message including attachments. Default: `100000` |
| `max_images` | Max image attachments per message. Default: `5` (vision models only) |
| `max_messages` | Max messages in a reply chain. Default: `25` |
| `use_plain_responses` | Use plaintext instead of embeds. Disables streaming. Default: `false` |
| `show_embed_color` | Show green/orange bar on responses. Default: `true` |
| `allow_dms` | Allow direct messages. Default: `true` |
| `permissions` | `users`, `roles`, `channels` with `allowed_ids`, `blocked_ids`, and `admin_ids`. |

### LLM settings

| Setting | Description |
|---------|-------------|
| `providers` | LLM providers with `base_url` and optional `api_key`. |
| `models` | Models in `provider/model: params` format. First model is the startup default. |
| `fallback_models` | Global fallback list tried when primary model fails. |
| `system_prompt` | Global system prompt. Supports `{date}` and `{time}` tags. |

### Model parameters

```yaml
models:
  openrouter/openrouter/free:
    persona: bao                    # loads bot/config/personas/bao.md
    tools: ["web_search"]           # enable tools
    supports_tools: true            # required for OpenAI-compat streaming fallback
    fallback_models:
      - "ollama/qwen3:14b"

  ollama/qwen3:14b:
    tools: ["web_search", "visuals_core"]
    think: true                     # enable reasoning mode
    system_prompt: "You are helpful."
```

---

## Error handling

- All serious errors are DMed to admins listed in `permissions.users.admin_ids`.
- LLM/API errors: bot retries using fallback models. If all fail, users see a zh-TW message and admin gets a DM.
- Scheduled tasks: same fallback behavior; target channel/user is notified on total failure.
- Slash command errors handled centrally via `bot/discord/errors.py`.

---

## Notes

- **Scheduled task DM issues:** If you get "Cannot send messages to this user", either the user has DMs from bots disabled, hasn't DMed the bot first, or the bot is blocked. Use `channel_id` as a fallback.

- **Thinking tags:** `<think>` blocks are automatically stripped from all responses — users only see the final answer.

- **Tool call parse errors (Ollama):** Models like `qwen3:14b` with `think: true` can bleed reasoning tokens into tool JSON. The bot automatically retries without tools in this case. Use `qwen2.5:14b` or `llama3.1` for reliable tool calling.

- Only OpenAI API and xAI API are user-identity aware (support the `name` message field).

- This is a fork of [llmcord](https://github.com/jakobdylanc/llmcord). PRs welcome :)
