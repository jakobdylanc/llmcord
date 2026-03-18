## Why

The bot currently only supports text-based communication. Users want voice capabilities: the ability to send voice messages in text channels that get transcribed, and to have real-time voice conversations with the bot in Discord voice channels. Azure Speech Services provides high-quality TTS and STT that can be integrated as a separate module.

## What Changes

- Add `azure-speech` configuration section to config.yaml
- Create new `bot/voice/` module with TTS and STT wrappers
- Add voice cog with `/join` and `/leave` slash commands
- Handle voice message attachments in text channels and DMs
- Update llmcord.py to load voice cog and handle voice intents

## Capabilities

### New Capabilities

- `azure-speech-tts`: Text-to-speech using Azure Speech Services - converts text to audio
- `azure-speech-stt`: Speech-to-text using Azure Speech Services - converts audio to text
- `voice-integration`: Discord voice channel commands (/join, /leave) and voice message handling
- `voice-config`: Azure speech configuration in config.yaml

### Modified Capabilities

- (None - this is a new feature)

## Impact

- New module: `bot/voice/` with tts.py, stt.py, config.py
- New config section: `azure-speech` in config.yaml
- Modified: llmcord.py (voice intents, load voice cog)
- New dependency: `azure-cognitiveservices-speech`
- Documentation: Update getting-started.md, create azure-speech.md