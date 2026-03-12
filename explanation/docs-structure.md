# Documentation Structure

This project uses a hybrid documentation approach: **OpenSpec** for AI/agents and **Markdown** for humans.

## Why Hybrid?

1. **AI needs structure**: AI agents need clear APIs, data models, and current state - not prose
2. **Humans need explanation**: Developers need tutorials, rationale, and context - not just specs
3. **Single source of truth**: OpenSpec is the source; markdown links to it

## Structure

```
openspec/specs/     → AI source of truth
  ├── project/      → Module structure, APIs, data models
  ├── providers/    → LLM provider interfaces
  ├── tools/        → Tool registry & schemas
  ├── config/       → Config schema & validation
  └── discord/      → Discord bot interfaces

*.md (root)         → Human-friendly docs
tutorial/           → How-to guides
explanation/        → Rationale docs
```

## For AI Agents

Read `openspec/specs/` to understand:
- Current project state (what's done, in progress, backlog)
- All APIs and their signatures
- Data models and contracts
- Provider interfaces
- Tool registry

## For Humans

Read markdown files for:
- **Getting started**: [tutorial/getting-started.md](../tutorial/getting-started.md)
- **Configuration**: [tutorial/configure-provider.md](../tutorial/configure-provider.md)
- **Adding tools**: [tutorial/add-tool.md](../tutorial/add-tool.md)
- **Architecture decisions**: explanation/*.md

## How They Connect

- OpenSpec specs are concise and machine-readable
- Markdown tutorials link to OpenSpec for technical details
- When updating: update OpenSpec first, then update markdown

## Current State

See [TODOLIST.md](../TODOLIST.md) for what's done, in progress, and backlog.