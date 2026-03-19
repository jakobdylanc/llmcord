# server-detail Specification

## Purpose
Provides detailed view of Discord guilds with member/channel lists and permission management.

## ADDED Requirements

### Requirement: Server detail endpoint
The system SHALL provide an endpoint to get detailed information about a specific Discord guild.

#### Scenario: Get server details
- **WHEN** user requests GET /api/servers/{guild_id}
- **THEN** response includes: guild name, icon, member list, channel list, permissions

### Requirement: Display server icon
The system SHALL display the server icon when available.

#### Scenario: Server has icon
- **WHEN** guild has an icon set
- **THEN** icon URL is returned in /api/servers/{guild_id} response

### Requirement: List guild members
The system SHALL provide a list of guild members with username and display name.

#### Scenario: List members
- **WHEN** user views server detail page
- **THEN** member list shows username and display name for each member

### Requirement: List guild channels
The system SHALL provide a list of guild text channels with ID and name.

#### Scenario: List channels
- **WHEN** user views server detail page
- **THEN** channel list shows channel_id and channel name for each text channel

### Requirement: Display bot permissions
The system SHALL display the bot's current permissions in the guild.

#### Scenario: Show permissions
- **WHEN** user views server detail page
- **THEN** current bot permissions are displayed

### Requirement: Permission management
The system SHALL allow granting and revoking bot permissions in a guild.

#### Scenario: Grant permission
- **WHEN** user clicks "Grant" button for a permission
- **THEN** PUT /api/servers/{guild_id}/permissions is called
- **AND** permission is granted to bot

#### Scenario: Revoke permission
- **WHEN** user clicks "Revoke" button for a permission
- **THEN** PUT /api/servers/{guild_id}/permissions is called
- **AND** permission is revoked from bot