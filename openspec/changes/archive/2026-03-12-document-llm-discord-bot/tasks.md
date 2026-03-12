## 1. Create OpenSpec Project Spec

- [x] 1.1 Create `openspec/specs/project/spec.md` - main project overview
- [x] 1.2 Define module structure (llmcord.py, bot/config, bot/llm, bot/discord)
- [x] 1.3 Define data models (Config, Persona, ScheduledTask, ToolEntry)
- [x] 1.4 Define key APIs with signatures

## 2. Create OpenSpec Provider Spec

- [x] 2.1 Create `openspec/specs/providers/spec.md`
- [x] 2.2 Define provider interface (base_url, api_key, client)
- [x] 2.3 Document supported providers (Ollama, OpenAI, OpenRouter, Google)
- [x] 2.4 Define fallback chain logic

## 3. Create OpenSpec Tool Spec

- [x] 3.1 Create `openspec/specs/tools/spec.md`
- [x] 3.2 Define ToolEntry dataclass contract
- [x] 3.3 Document available tools (web_search, visuals_core, yahoo_finance, google_tools)
- [x] 3.4 Define tool execution interface

## 4. Create OpenSpec Config Spec

- [x] 4.1 Create `openspec/specs/config/spec.md`
- [x] 4.2 Define config schema (providers, models, personas, tasks, permissions)
- [x] 4.3 Define validation rules
- [x] 4.4 Document persona and task file formats

## 5. Create OpenSpec Discord Spec

- [x] 5.1 Create `openspec/specs/discord/spec.md`
- [x] 5.2 Define Discord bot interface
- [x] 5.3 Define message handling contract
- [x] 5.4 Define slash command interface

## 6. Restructure Markdown Docs

- [x] 6.1 Update README.md to link to OpenSpec for technical details
- [x] 6.2 Create `tutorial/` directory with how-to guides
- [x] 6.3 Create `explanation/` directory with rationale docs
- [x] 6.4 Update existing docs to reference OpenSpec