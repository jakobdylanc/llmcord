## Why

This project has existing documentation (README.md, spec.md, report.md, TODOLIST.md, SKILLS.md) but it's mixed - some for humans, some for AI, not clearly separated. This makes it hard for AI agents to understand the current state and for humans to get oriented. We need a clear separation: OpenSpec as the source of truth for AI, markdown for human explanation.

## What Changes

- Restructure docs into hybrid format:
  - **OpenSpec** (`openspec/`): Structure, rules, interfaces, current state - AI readable
  - **Markdown** (`*.md` in root): Explanation, tutorials, rationale - human friendly
- Create a whole-project spec in OpenSpec that AI can read to understand current state

## Capabilities

### New Capabilities

- **project-spec**: Whole-project spec in OpenSpec format (APIs, modules, data models, interfaces)
- **hybrid-docs-structure**: Clear separation between OpenSpec (AI) and markdown (human)

### Modified Capabilities

- (None - this is initial restructuring)

## Impact

- OpenSpec specs in `openspec/specs/` and `openspec/changes/document-llm-discord-bot/specs/`
- Markdown docs remain in root for human reference
- AI agents read OpenSpec; humans read markdown