# bot-status-integration Specification

## Purpose
Enables the web portal to display real-time Discord bot status and reload bot configuration.

## ADDED Requirements

### Requirement: Bot registers with web portal on ready
The Discord bot SHALL register itself with the web portal when the on_ready event fires, providing access to bot state for the status API.

#### Scenario: Bot registers on ready
- **WHEN** Discord bot's on_ready event fires
- **THEN** set_discord_bot() and mark_bot_ready() are called to register bot reference

#### Scenario: Status API returns bot information
- **WHEN** user requests /api/status with valid JWT
- **THEN** response includes: status, online, server_count, uptime_seconds, started_at, user_count, channel_count, user_name, user_id

### Requirement: Bot config can be reloaded via portal
The web portal SHALL be able to trigger a reload of the bot's main configuration without requiring bot restart.

#### Scenario: Reload bot config
- **WHEN** user POSTs to /api/refresh
- **THEN** both portal config AND bot config are reloaded from disk
- **AND** response indicates success for both reloads

### Requirement: Bot config references registered
The bot SHALL register references to its global config (config, curr_model, curr_persona) with the web portal for external reload capability.

#### Scenario: Config refs registered on ready
- **WHEN** Discord bot's on_ready event fires
- **THEN** register_config_refs() is called to store config references