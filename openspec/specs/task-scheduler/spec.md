# task-scheduler Specification

## Purpose
TBD - created by archiving change document-llm-discord-bot. Update Purpose after archive.
## Requirements
### Requirement: APScheduler for cron jobs
The system SHALL use `APScheduler` for scheduling tasks with cron expressions.

#### Scenario: Schedule task
- **WHEN** task has cron: "0 * * * *"
- **THEN** APScheduler runs task every hour

### Requirement: Task config in YAML
The system SHALL load tasks from `bot/config/tasks/*.yaml` and config['scheduled_tasks'].

#### Scenario: Load tasks
- **WHEN** bot starts
- **THEN** loads all task YAML files and inline scheduled_tasks

### Requirement: Task sends to channel or user
The system SHALL send task output to either channel_id or user_id (not both currently).

#### Scenario: Task output
- **WHEN** task runs
- **THEN** sends to channel_id if set, else user_id if set

### Requirement: Task can override model/tools
Each task SHALL support: model, tools, persona, system_prompt overrides.

#### Scenario: Task override
- **WHEN** task specifies tools: ["web_search"]
- **THEN** task uses those tools instead of model's default

