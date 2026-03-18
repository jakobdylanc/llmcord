## ADDED Requirements

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