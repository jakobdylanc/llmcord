# voice-integration Specification

## Purpose
Defines voice channel integration including /join, /leave commands, voice message processing, STT transcription, and TTS speech synthesis using Azure Speech.

## Requirements
### Requirement: Bot joins voice channel via slash command
The system SHALL provide a `/join` slash command that makes the bot join the user's current voice channel.

#### Scenario: Join command in voice channel
- **WHEN** user is in a voice channel and runs /join
- **THEN** bot joins the same voice channel

#### Scenario: Join command not in voice channel
- **WHEN** user is not in a voice channel and runs /join
- **THEN** bot responds with error message directing user to join a voice channel first

### Requirement: Bot leaves voice channel via slash command
The system SHALL provide a `/leave` slash command that makes the bot leave its current voice channel.

#### Scenario: Leave command when in voice channel
- **WHEN** bot is in a voice channel and /leave is called
- **THEN** bot leaves the voice channel

#### Scenario: Leave command when not in voice channel
- **WHEN** bot is not in a voice channel and /leave is called
- **THEN** bot responds with message indicating it's not in a voice channel

### Requirement: Bot handles voice message attachments in text channels
The system SHALL detect and process voice message attachments in text channels.

#### Scenario: Voice message in text channel
- **WHEN** user sends a voice message in a text channel
- **THEN** bot downloads the audio, transcribes it via Azure STT, and processes as text with the LLM

#### Scenario: Voice message in DM
- **WHEN** user sends a voice message in DM to the bot
- **THEN** bot downloads the audio, transcribes it via Azure STT, and processes as text with the LLM

### Requirement: Bot responds with TTS in voice channels
The system SHALL allow the LLM to use TTS to speak responses in voice channels.

#### Scenario: TTS response in voice channel
- **WHEN** bot is in a voice channel and LLM returns a response with TTS enabled
- **THEN** bot converts text to speech and plays audio in the voice channel

### Requirement: Slash commands registered on bot startup
The voice slash commands SHALL be registered when the bot loads the voice cog.

#### Scenario: Bot loads voice cog
- **WHEN** bot loads the voice cog
- **THEN** /join and /leave commands appear in Discord's slash command list

## Azure Speech STT

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

## Azure Speech TTS

### Requirement: Azure TTS service initialized with config
The system SHALL initialize Azure Speech synthesizer using configuration from `config.yaml` under the `azure-speech` section.

#### Scenario: TTS initialization with valid config
- **WHEN** azure-speech section exists in config with `key` and `region`
- **THEN** TTS service initializes successfully

#### Scenario: TTS initialization without config
- **WHEN** azure-speech section is missing from config
- **THEN** TTS service raises configuration error or is None

### Requirement: TTS speak converts text to audio
The system SHALL provide a `speak(text, voice, style)` method that converts text to audio using Azure TTS.

#### Scenario: Speak with default voice
- **WHEN** speak("Hello world") is called
- **THEN** returns audio bytes in MP3 format

#### Scenario: Speak with custom voice
- **WHEN** speak("Hello", voice="en-US-JennyNeural") is called
- **THEN** returns audio bytes using the specified voice

#### Scenario: Speak with style
- **WHEN** speak("Hello", style="cheerful") is called
- **THEN** returns audio bytes with the specified speaking style

### Requirement: TTS lists available voices
The system SHALL provide a `list_voices()` method that returns a list of available Azure voices.

#### Scenario: List voices
- **WHEN** list_voices() is called
- **THEN** returns list of voice objects with name, locale, gender properties

### Requirement: TTS runs independently of Discord
The TTS module SHALL be testable without running the Discord bot.

#### Scenario: TTS standalone test
- **WHEN** TTS module is imported and speak() is called with valid Azure credentials
- **THEN** returns audio bytes without requiring Discord client

