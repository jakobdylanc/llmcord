## Context

This is an existing Discord Bot project that streams LLM responses. Current docs are mixed (human + AI). Need to restructure so AI can read OpenSpec to understand current state.

## Goals / Non-Goals

**Goals:**
- OpenSpec specs contain: APIs, modules, data models, interfaces, current state
- Markdown docs contain: tutorials, explanations, rationale
- AI reads OpenSpec; humans read markdown
- Keep existing markdown docs but restructure

**Non-Goals:**
- No code changes
- Not duplicating content - link between OpenSpec and markdown

## Decisions

- **OpenSpec location**: `openspec/specs/<capability>/spec.md`
- **Markdown location**: Root `*.md` files, `tutorial/`, `explanation/`, `reference/`
- **Linking**: Markdown references OpenSpec; OpenSpec points to source code

## Hybrid Structure

```
openspec/
├── specs/                    # AI source of truth
│   ├── project/             # Whole-project overview
│   ├── providers/           # LLM provider interfaces
│   ├── tools/               # Tool registry & schemas
│   ├── config/              # Config schema & validation
│   └── discord/             # Discord bot interfaces
└── changes/                 # Change proposals

*.md                        # Human-friendly docs
tutorial/
explanation/
reference/
```

## Risks / Trade-offs

- [Risk] Keeping docs in sync → Mitigation: OpenSpec is source; markdown links to it
- [Risk] Two places to update → Mitigation: OpenSpec is concise; markdown is explanatory
