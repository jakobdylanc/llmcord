# persona-management Specification

## Purpose
Enables CRUD operations for persona files and live preview editing in the web portal.

## ADDED Requirements

### Requirement: List existing personas
The system SHALL provide an endpoint to list all available personas with their titles and content previews.

#### Scenario: List personas
- **WHEN** user requests GET /api/personas
- **THEN** response includes array of {name, content} for each persona file in bot/config/personas/

### Requirement: Create new persona
The system SHALL allow creating new persona files via POST endpoint.

#### Scenario: Create persona with Chinese content
- **WHEN** POST /api/personas with Chinese characters in content
- **THEN** file is created at bot/config/personas/{name}.md with UTF-8 encoding

### Requirement: Update existing persona
The system SHALL allow updating existing persona files via PUT endpoint.

#### Scenario: Update persona
- **WHEN** PUT /api/personas/{name} with new content
- **THEN** existing file is updated with new content

### Requirement: Delete persona
The system SHALL allow deleting persona files.

#### Scenario: Delete persona
- **WHEN** DELETE /api/personas/{name}
- **THEN** persona file is removed from bot/config/personas/

### Requirement: Persona editor with preview
The system SHALL provide a markdown editor with live preview for persona editing.

#### Scenario: Editor shows preview
- **WHEN** user types in markdown editor
- **THEN** rendered preview updates in real-time

### Requirement: Apply persona changes
The system SHALL reload persona files in-memory when Apply is clicked without saving to file.

#### Scenario: Apply without save
- **WHEN** user clicks Apply (without Save)
- **THEN** bot reloads persona files from disk
- **AND** config.yaml remains unchanged