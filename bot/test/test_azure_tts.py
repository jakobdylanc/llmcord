"""
Tests for Azure TTS module.

Tests can run independently without Discord bot.
Note: Tests require valid Azure credentials to actually call the API.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from bot.voice.config import VoiceConfig


class TestAzureTTSImport:
    """Test that the module can be imported."""
    
    def test_import_azure_tts(self):
        """Should import AzureTTS class."""
        from bot.voice.tts import AzureTTS, create_tts
        assert AzureTTS is not None
        assert create_tts is not None
    
    def test_import_voice_dataclass(self):
        """Should import Voice dataclass."""
        from bot.voice.tts import Voice
        assert Voice is not None


class TestAzureTTSCreation:
    """Test AzureTTS instance creation."""
    
    def test_raises_when_config_not_configured(self):
        """Should raise ValueError when config is not configured."""
        from bot.voice.tts import AzureTTS
        
        config = VoiceConfig(key="", region="eastus")
        with pytest.raises(ValueError, match="not configured"):
            AzureTTS(config)
    
    def test_raises_when_region_missing(self):
        """Should raise ValueError when region is missing."""
        from bot.voice.tts import AzureTTS
        
        config = VoiceConfig(key="test-key", region="")
        with pytest.raises(ValueError, match="not configured"):
            AzureTTS(config)


class TestVoiceDataclass:
    """Test Voice dataclass."""
    
    def test_creates_voice_from_info(self):
        """Should create Voice from VoiceInfo."""
        from bot.voice.tts import Voice
        
        # Create a mock with a name attribute (like an enum)
        mock_gender = Mock()
        mock_gender.name = "Female"
        
        mock_info = Mock()
        mock_info.name = "en-US-JennyNeural"
        mock_info.locale = "en-US"
        mock_info.gender = mock_gender
        mock_info.short_name = "Jenny Neural"
        
        voice = Voice.from_voice_info(mock_info)
        
        assert voice.name == "en-US-JennyNeural"
        assert voice.locale == "en-US"
        assert voice.gender == "female"
        assert voice.short_name == "Jenny Neural"
    
    def test_creates_voice_from_string_gender(self):
        """Should create Voice from VoiceInfo with string gender."""
        from bot.voice.tts import Voice
        
        mock_info = Mock()
        mock_info.name = "en-US-JennyNeural"
        mock_info.locale = "en-US"
        mock_info.gender = "Female"  # String instead of enum
        mock_info.short_name = "Jenny Neural"
        
        voice = Voice.from_voice_info(mock_info)
        
        assert voice.gender == "female"


class TestCreateTTSFactory:
    """Test create_tts factory function."""
    
    def test_returns_none_when_config_is_none(self):
        """Should return None when config is None."""
        from bot.voice.tts import create_tts
        
        result = create_tts(None)
        assert result is None
    
    def test_returns_none_when_not_configured(self):
        """Should return None when config is not configured."""
        from bot.voice.tts import create_tts
        
        config = VoiceConfig(key="", region="eastus")
        result = create_tts(config)
        assert result is None
    
    @patch('bot.voice.tts.AzureTTS')
    def test_returns_tts_instance_when_configured(self, mock_tts_class):
        """Should return AzureTTS instance when config is valid."""
        from bot.voice.tts import create_tts
        
        mock_tts_instance = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        config = VoiceConfig(key="test-key", region="eastus")
        result = create_tts(config)
        
        assert result is mock_tts_instance
        mock_tts_class.assert_called_once_with(config)


class TestTTSXML:
    """Test XML escaping in TTS."""
    
    def test_escape_xml_escapes_ampersand(self):
        """Should escape & to &amp;."""
        from bot.voice.tts import AzureTTS
        
        text = "Tom & Jerry"
        result = AzureTTS._escape_xml(text)
        assert "&amp;" in result
    
    def test_escape_xml_escapes_lt_gt(self):
        """Should escape < and >."""
        from bot.voice.tts import AzureTTS
        
        text = "<hello> world"
        result = AzureTTS._escape_xml(text)
        assert "&lt;" in result
        assert "&gt;" in result


class TestTTSIntegration:
    """
    Integration tests that require actual Azure credentials.
    
    These tests are marked with a special marker and will be skipped
    unless AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables
    are set.
    """
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid config from environment or skip."""
        import os
        key = os.environ.get("AZURE_SPEECH_KEY")
        region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
        
        if not key:
            pytest.skip("AZURE_SPEECH_KEY not set")
        
        return VoiceConfig(key=key, region=region, default_voice="en-US-JennyNeural")
    
    def test_tts_can_be_instantiated(self, valid_config):
        """Should be able to create AzureTTS with valid credentials."""
        from bot.voice.tts import AzureTTS
        
        tts = AzureTTS(valid_config)
        assert tts is not None
        assert tts.config == valid_config
    
    def test_speak_returns_bytes(self, valid_config):
        """speak() should return audio bytes."""
        from bot.voice.tts import AzureTTS
        
        tts = AzureTTS(valid_config)
        audio = tts.speak("Hello world")
        
        assert isinstance(audio, bytes)
        assert len(audio) > 0
    
    def test_speak_with_custom_voice(self, valid_config):
        """speak() should work with custom voice."""
        from bot.voice.tts import AzureTTS
        
        tts = AzureTTS(valid_config)
        audio = tts.speak("Hello", voice="en-US-GuyNeural")
        
        assert isinstance(audio, bytes)
        assert len(audio) > 0
    
    def test_list_voices_returns_list(self, valid_config):
        """list_voices() should return a list of voices."""
        from bot.voice.tts import AzureTTS
        
        tts = AzureTTS(valid_config)
        voices = tts.list_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
    
    def test_list_voices_with_locale_filter(self, valid_config):
        """list_voices() should filter by locale."""
        from bot.voice.tts import AzureTTS
        
        tts = AzureTTS(valid_config)
        voices = tts.list_voices(locale="en-US")
        
        assert isinstance(voices, list)
        # All returned voices should match the locale
        for voice in voices:
            assert voice.locale.startswith("en-")