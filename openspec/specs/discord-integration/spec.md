# discord-integration Specification

## Purpose
TBD - created by archiving change document-llm-discord-bot. Update Purpose after archive.
## Requirements
### Requirement: Discord client uses discord.py
The system SHALL use `discord.Client` with `intents` for message content.

#### Scenario: Bot startup
- **WHEN** bot starts
- **THEN** connects to Discord using bot_token from config

### Requirement: Message handling via on_message
The system SHALL process messages in channels where bot has read permissions.

#### Scenario: Process message
- **WHEN** message received and mentions bot or in allowed channel
- **THEN** generate LLM response and send

### Requirement: Slash commands registered
The system SHALL register slash commands: /model, /clear, /persona, /skill, /task, /refresh.

#### Scenario: Slash command
- **WHEN** user runs /model
- **THEN** bot switches to selected model

### Requirement: Error notification to admins
The system SHALL DM errors to admin user IDs in permissions.users.admin_ids.

#### Scenario: Error
- **WHEN** exception occurs
- **THEN** notify_admin_error() sends DM to admins

