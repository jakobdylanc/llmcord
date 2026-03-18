# azure-speech-stt Specification

## Purpose
TBD - created by archiving change azure-voice-integration. Update Purpose after archive.
## Requirements
### Requirement: Azure STT service initialized with config
The system SHALL initialize Azure Speech recognizer using configuration from `config.yaml` under the `azure-speech` section.

#### Scenario: STT initialization with valid config
- **WHEN** azure-speech section exists in config with `key` and `region`
- **THEN** STT service initializes successfully

#### Scenario: STT initialization without config
- **WHEN** azure-speech section is missing from config
- **THEN** STT service raises configuration error or is None

### Requirement: STT transcribe converts audio to text
The system SHALL provide a `transcribe(audio_bytes)` method that converts audio to text using Azure STT.

#### Scenario: Transcribe audio bytes
- **WHEN** transcribe(audio_bytes) is called with valid audio data
- **THEN** returns transcribed text string

#### Scenario: Transcribe from file path
- **WHEN** transcribe_file("/path/to/audio.ogg") is called
- **THEN** reads audio file and returns transcribed text

### Requirement: STT handles Discord voice message format
The system SHALL be able to process Discord voice message attachments (.ogg format).

#### Scenario: Transcribe Discord voice message
- **WHEN** voice message attachment is downloaded and passed to transcribe()
- **THEN** returns the spoken text content

### Requirement: STT runs independently of Discord
The STT module SHALL be testable without running the Discord bot.

#### Scenario: STT standalone test
- **WHEN** STT module is imported and transcribe() is called with valid audio and Azure credentials
- **THEN** returns text without requiring Discord client

