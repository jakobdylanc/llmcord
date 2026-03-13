# Discord Spec

## Purpose
Defines Discord bot integration, including message handling, slash commands, and error handling.

## Requirements
### Requirement: Bot authenticates with Discord
The bot SHALL authenticate using the configured bot_token and require MESSAGE CONTENT INTENT.

#### Scenario: Bot authentication
- **WHEN** bot starts with valid token
- **THEN** it connects to Discord and registers slash commands

### Requirement: Message handling with reply system
The bot SHALL respond to messages that mention it or reply to bot messages, building conversation from reply chain.

#### Scenario: Message handling
- **WHEN** user mentions bot or replies to bot message
- **THEN** bot builds conversation and streams LLM response

### Requirement: Slash commands for control
The bot SHALL provide slash commands for model switching, conversation clearing, persona management, skill listing, task management, and config reload.

#### Scenario: Slash commands
- **WHEN** user invokes /model, /clear, /persona, /skill, /task, or /refresh
- **THEN** bot executes the corresponding action

### Requirement: Error handling with admin notifications
The bot SHALL notify admins of errors and handle slash command errors gracefully.

#### Scenario: Error handling
- **WHEN** an error occurs
- **THEN** admin is notified via DM and user sees localized error message

## Bot Setup

- Uses `discord.py` with `Intents.members` and `Intents.message_content`
- Authenticates with `bot_token` from config
- Requires MESSAGE CONTENT INTENT enabled in Discord Developer Portal

## Message Handling

### Flow
1. `on_message` event triggers
2. Check if message mentions bot or in allowed channel/DM
3. Build conversation from reply chain
4. Call LLM with messages
5. Stream response to Discord

### Reply System
- Bot responds to messages that mention it OR reply to bot messages
- Builds conversation from reply chain (up to max_messages)
- DM conversations continue automatically

## Slash Commands

| Command | Description | Admin |
|---------|-------------|-------|
| /model | Switch model | ✓ |
| /clear | Reset conversation | ✓ |
| /persona | View/switch persona | ✓ |
| /skill | List available skills | ✓ |
| /task | List/toggle/run tasks | ✓ |
| /refresh | Reload config, tasks, model, persona | ✓ |

## Error Handling

- `bot/discord/errors.py` provides:
  - `notify_admin_error(bot, config, error, context)` - DM admins
  - `handle_app_command_error(interaction, error, bot, config)` - Slash command errors

- Admin IDs from `permissions.users.admin_ids`
- User-facing errors in zh-TW

## Response Streaming

- Uses Discord typing indicator during generation
- Edits message with chunks as they arrive
- Embed color: green (complete), orange (streaming)
- Strips `<think>` tags from reasoning models