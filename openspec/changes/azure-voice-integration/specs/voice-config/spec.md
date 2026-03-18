## ADDED Requirements

### Requirement: Azure-speech config section in config.yaml
The system SHALL support an `azure-speech` section in config.yaml with required and optional fields.

#### Scenario: Valid azure-speech config
- **WHEN** config.yaml contains:
  ```yaml
  azure-speech:
    key: "your-azure-speech-key"
    region: "eastus"
  ```
- **THEN** the config is considered valid

#### Scenario: Missing required fields
- **WHEN** azure-speech section exists but missing `key` or `region`
- **THEN** config validation fails with appropriate error message

#### Scenario: Azure-speech section optional
- **WHEN** azure-speech section is not present in config.yaml
- **THEN** voice features are disabled, bot works without them

### Requirement: Azure-speech config fields defined
The azure-speech config SHALL support these fields:

#### Scenario: Key field
- **WHEN** `key` field is provided in azure-speech config
- **THEN** used for Azure Speech API authentication

#### Scenario: Region field
- **WHEN** `region` field is provided (e.g., "eastus", "westeurope")
- **THEN** used to determine Azure Speech endpoint region

#### Scenario: Optional endpoint field
- **WHEN** `endpoint` field is provided in azure-speech config
- **THEN** used as custom Azure Speech endpoint (optional override)

#### Scenario: Default voice field
- **WHEN** `default_voice` field is provided (e.g., "en-US-JennyNeural")
- **THEN** used as the default voice for TTS when not specified

#### Scenario: Default style field
- **WHEN** `default_style` field is provided (e.g., "cheerful", "sad")
- **THEN** used as the default speaking style for TTS