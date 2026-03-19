# config-three-buttons Specification

## Purpose
Adds Save, Apply, and Save&Apply buttons to the config page for proper config management.

## ADDED Requirements

### Requirement: Save button writes to config.yaml
The system SHALL provide a Save button that writes configuration changes to config.yaml file without reloading the bot.

#### Scenario: Save config with Chinese characters
- **WHEN** user modifies status_message containing Chinese characters and clicks Save
- **THEN** config.yaml is written with UTF-8 encoding preserving Chinese characters

### Requirement: Apply button reloads config in-memory
The system SHALL provide an Apply button that reloads configuration in the running bot without modifying config.yaml.

#### Scenario: Apply config changes
- **WHEN** user clicks Apply button
- **THEN** bot's in-memory config is reloaded from current config.yaml
- **AND** config.yaml file remains unchanged

### Requirement: Save&Apply performs both actions
The system SHALL provide a Save&Apply button that writes to config.yaml AND reloads in-memory config.

#### Scenario: Save and Apply
- **WHEN** user clicks Save&Apply button
- **THEN** config.yaml is written with changes
- **AND** bot's in-memory config is reloaded

### Requirement: Fix [object Object] display
The system SHALL display complex config fields (models, permissions, providers, tools) in human-readable format.

#### Scenario: Display models field
- **WHEN** config includes models: {ollama: {...}, openai: {...}}
- **THEN** UI displays "ollama, openai" not "[object Object]"

#### Scenario: Display permissions field
- **WHEN** config includes permissions object
- **THEN** UI displays permission key names