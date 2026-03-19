# task-management Specification

## Purpose
Enables CRUD operations for scheduled tasks, YAML editing, and manual task execution.

## ADDED Requirements

### Requirement: List existing tasks
The system SHALL provide an endpoint to list all available tasks with their full content.

#### Scenario: List tasks
- **WHEN** user requests GET /api/tasks
- **THEN** response includes array of {name, content, enabled, status} for each task

### Requirement: Create new task
The system SHALL allow creating new task files via POST endpoint.

#### Scenario: Create task with Chinese content
- **WHEN** POST /api/tasks with Chinese characters in content
- **THEN** file is created at bot/config/tasks/{name}.yaml with UTF-8 encoding

### Requirement: Update existing task
The system SHALL allow updating existing task files via PUT endpoint.

#### Scenario: Update task
- **WHEN** PUT /api/tasks/{name} with new content
- **THEN** existing file is updated with new content

### Requirement: Delete task
The system SHALL allow deleting task files.

#### Scenario: Delete task
- **WHEN** DELETE /api/tasks/{name}
- **THEN** task file is removed from bot/config/tasks/

### Requirement: YAML validation before save
The system SHALL validate YAML format before saving task files.

#### Scenario: Invalid YAML rejected
- **WHEN** user submits task with invalid YAML syntax
- **THEN** error is returned and file is not saved

### Requirement: Enable/disable task
The system SHALL allow toggling task enabled status.

#### Scenario: Toggle task enabled
- **WHEN** user clicks enable/disable toggle on task
- **THEN** task's enabled state is updated in config

### Requirement: Manual task execution
The system SHALL allow manual task execution via "Run Now" button.

#### Scenario: Run task manually
- **WHEN** user clicks "Run Now" on a task
- **THEN** POST /api/tasks/{name}/run is called
- **AND** task is queued for execution asynchronously

### Requirement: Apply task changes
The system SHALL reload task files in-memory when Apply is clicked without saving to file.

#### Scenario: Apply without save
- **WHEN** user clicks Apply (without Save)
- **THEN** bot reloads task files from disk
- **AND** config.yaml remains unchanged