## 1. Phase 1: Config Layer

- [x] 1.1 Add azure-speech section to config-example.yaml (key, region, endpoint, default_voice, default_style)
- [x] 1.2 Add azure-speech validation to bot/config/validator.py
- [x] 1.3 Create bot/voice/ directory structure
- [x] 1.4 Create bot/voice/config.py to parse azure-speech config
- [x] 1.5 Create test/test_voice_config.py to validate config loading

## 2. Phase 2: Azure TTS Module

- [x] 2.1 Add azure-cognitiveservices-speech to requirements.txt
- [x] 2.2 Create bot/voice/tts.py with AzureTTS class
- [x] 2.3 Implement speak(text, voice, style) method returning audio bytes
- [x] 2.4 Implement list_voices() method
- [x] 2.5 Create test/test_azure_tts.py for standalone TTS testing
- [x] 2.6 Validate TTS produces valid audio output (CHECKPOINT 1)

## 3. Phase 3: Azure STT Module

- [x] 3.1 Create bot/voice/stt.py with AzureSTT class
- [x] 3.2 Implement transcribe(audio_bytes) method returning text
- [x] 3.3 Implement transcribe_file(audio_path) method
- [x] 3.4 Create test/test_azure_stt.py for standalone STT testing
- [x] 3.5 Validate STT produces text from audio (CHECKPOINT 2)

## 4. Phase 4: Discord Voice Integration

- [x] 4.1 Update llmcord.py to enable voice_states intent
- [x] 4.2 Create bot/voice/__init__.py with VoiceCog
- [x] 4.3 Implement /join slash command
- [x] 4.4 Implement /leave slash command
- [x] 4.5 Load voice cog in llmcord.py
- [x] 4.6 Add voice message attachment handling in text channels
- [x] 4.7 Create test/test_voice_cog.py
- [x] 4.8 Validate bot can join/leave voice channels (CHECKPOINT 3)

## 5. Phase 5: Documentation

- [x] 5.1 Update config-example.yaml with azure-speech documentation
- [x] 5.2 Update docs/getting-started.md with voice features
- [x] 5.3 Create docs/azure-speech.md configuration guide
- [x] 5.4 Update README.md with voice capabilities
- [x] 5.5 Validate all documentation is complete (CHECKPOINT 4)

## 6. Final Validation

- [ ] 6.1 Run full test suite
- [ ] 6.2 Manual testing of all voice features
- [ ] 6.3 Archive change in OpenSpec