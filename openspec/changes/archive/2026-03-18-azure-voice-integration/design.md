## Context

The gpt-discord-bot currently supports text-based chat with LLMs via Discord. This design addresses adding voice capabilities:
- Text-to-Speech (TTS) to read responses aloud
- Speech-to-Text (STT) to understand voice messages
- Voice channel integration for real-time conversation

The system will use Azure Speech Services for TTS/STT. The design follows the existing pattern of separate modules in `bot/` (similar to `bot/llm/`, `bot/config/`).

## Goals / Non-Goals

**Goals:**
- Create standalone `bot/voice/` module with TTS and STT that can be tested independently
- Add `/join` and `/leave` slash commands to control bot's voice channel presence
- Handle voice message attachments in text channels and DMs via STT
- Allow LLM to use TTS as a callable tool
- Validate each component before integrating with llmcord.py

**Non-Goals:**
- Real-time voice conversation in voice channels (listening and responding in real-time)
- Push-to-talk or voice activity detection
- Multi-voice channel support (bot in one channel at a time)
- Voice recording storage or history

## Decisions

### 1. Module Structure: Separate voice module
**Decision:** Create `bot/voice/` as a separate module with `tts.py`, `stt.py`, and `config.py`

**Rationale:** 
- Follows existing pattern (`bot/llm/`, `bot/config/`)
- Keeps TTS/STT logic separate from Discord bot logic
- Allows independent testing without running the full bot
- Prevents llmcord.py from becoming bloated

**Alternatives Considered:**
- Adding directly to llmcord.py: Rejected - would mix voice logic with bot logic
- Creating a tool in bot/llm/tools/: Rejected - tools are for LLM callable functions, not Discord voice handling

### 2. Config Location: config.yaml with azure-speech section
**Decision:** Store Azure credentials in `config.yaml` under `azure-speech` section

**Rationale:**
- Follows existing provider configuration pattern in config.yaml
- Keeps all configuration in one place
- Easier to validate with existing validator

**Alternatives Considered:**
- `.env` file: Already has placeholders in `.env.example` but分散 - better to centralize

### 3. Testing Strategy: Phased validation
**Decision:** Validate each phase independently before proceeding

**Rationale:**
- TTS/STT can be tested with simple scripts without Discord
- Voice cog can be tested by loading it in a test bot
- Reduces integration issues later

### 4. Voice Message Handling: Download and transcribe
**Decision:** When a voice message is sent, download the audio file, send to Azure STT, process as text

**Rationale:**
- Discord sends voice messages as `.ogg` attachments
- Azure STT can process audio from bytes
- Follows existing pattern of handling attachments

## Risks / Trade-offs

- [Risk] Azure Speech SDK dependency is large (~100MB) → Mitigation: Make optional, only install if voice features needed
- [Risk] TTS audio playback requires ffmpeg → Mitigation: Document ffmpeg requirement, use discord.FFmpegPCMAudio
- [Risk] Voice messages in Discord are short (60s max) → Mitigation: Document limitation, works within constraints
- [Risk] Latency in STT processing → Mitigation: Process asynchronously, don't block message handling

## Migration Plan

1. Phase 1: Add azure-speech config to config-example.yaml, add validation
2. Phase 2: Implement TTS module, test independently
3. Phase 3: Implement STT module, test independently  
4. Phase 4: Create voice cog, add /join /leave commands, integrate with llmcord.py
5. Phase 5: Update documentation

## Open Questions

- Should TTS output play immediately or queue? (Decided: Play immediately, interrupt if new message)
- Should voice messages be transcribed immediately or wait for LLM? (Decided: Transcribe immediately, send to LLM)