# Discord Spec

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