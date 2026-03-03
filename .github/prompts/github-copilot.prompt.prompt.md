---
name: github-copilot.prompt
description: Generate or update workspace instructions file for AI coding agents
argument-hint: Optionally specify a focus area or pattern to document for agents
agent: agent
---
Related skill: `agent-customization`.

Generate or update workspace instructions (`.github/copilot-instructions.md` as first choice, or `AGENTS.md` if it is already present) for guiding AI coding agents in this workspace.

## Required Documentation Review

Before starting work, this agent **MUST** read and understand:
1. **README.md** - Project overview, setup, and purpose
2. **todolist.md** - Current tasks and priorities
3. **spec.md** - Technical specifications and requirements
4. **report.md** - Status updates, completed work, and blockers

After completing work, update all relevant documents with:
* Progress made in **todolist.md** (mark items complete, add new items discovered)
* Technical changes in **spec.md** (if architecture/design changed)
* Status and decisions in **report.md** (what was done, why, any blockers)

---

## Discovery

Search for existing AI conventions using this glob pattern: 
`**/{.github/copilot-instructions.md,AGENT.md,AGENTS.md,CLAUDE.md,.cursorrules,.windsurfrules,.clinerules,.cursor/rules/**,.windsurf/rules/**,.clinerules/**,README.md}`

Then, start a subagent to research essential knowledge. Only include sections the workspace benefits from:

### Suggested Output Template
```markdown
# Project Guidelines

## Code Style
{Language and formatting preferences - reference key files that exemplify patterns}

## Architecture
{Major components, service boundaries, data flows, the "why" behind structural decisions}

## Build and Test
{Commands to install, build, test - agents will attempt to run these automatically}

## Project Conventions
{Patterns that differ from common practices - include specific examples from the codebase}

## Integration Points
{External dependencies and cross-component communication}

## Security
{Sensitive areas and auth patterns}

## Documentation Standards
{How to update README.md, todolist.md, spec.md, report.md - what goes where}

```

---

## Guidelines

* **Merge Intelligently**: If instructions already exist, preserve valuable content while updating outdated sections.
* **Location Priority**: If `AGENTS.md` exists, prefer updating it. For monorepos, use nested files per package.
* **Conciseness**: Write actionable instructions (~20-50 lines) using clean markdown structure.
* **Evidence-Based**: Link specific examples and reference key directories.
* **Avoid Generics**: Skip advice like "write tests." Focus on **this** project's specific approaches.
* **Current State**: Document only discoverable patterns, not aspirational/future practices.

---

## Pre-Work Checklist

* [ ] Read `README.md` for context
* [ ] Check `todolist.md` for blockers and priorities
* [ ] Review `spec.md` for requirements
* [ ] Check `report.md` for recent decisions and status

## Post-Work Checklist

* [ ] Update `todolist.md` (completed items, new discoveries)
* [ ] Update `spec.md` if architecture/design changed
* [ ] Update `report.md` with status and decisions
* [ ] Verify consistency across all four docs

---

**Instruction:** Update `.github/copilot-instructions.md` or `AGENTS.md`, then ask for feedback on unclear or incomplete sections to iterate.

```

### Key Changes Made:
1.  **Fixed Nested Blocks**: Used a `### Suggested Output Template` header so the agent understands the following block is the *target* format.
2.  **Standardized Lists**: Replaced the non-standard characters with standard Markdown bullet points (`*`) and Task Lists (`- [ ]`) for the checklists.
3.  **Visual Hierarchy**: Added horizontal rules (`---`) to separate the meta-instructions from the operational checklists.
4.  **Cleaning Up Prose**: Moved the "Pro tips" and "Key improvements" out of the prompt body itself, as those are usually for your reference rather than the AI's execution instructions.

Would you like me to help you define the specific **Documentation Convention** section for your `spec.md` to match this new workflow?

```