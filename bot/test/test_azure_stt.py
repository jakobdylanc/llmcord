"""
Tests for Azure STT module.

Tests can run independently without Discord bot.
Note: Tests require valid Azure credentials to actually call the API.
"""

import pytest
from unittest.mock import Mock, patch
from bot.voice.config import VoiceConfig


class TestAzureSTTImport:
    """Test that the module can be imported."""
    
    def test_import_azure_stt(self):
        """Should import AzureSTT class."""
        from bot.voice.stt import AzureSTT, create_stt
        assert AzureSTT is not None
        assert create_stt is not None


class TestAzureSTTCreation:
    """Test AzureSTT instance creation."""
    
    def test_raises_when_config_not_configured(self):
        """Should raise ValueError when config is not configured."""
        from bot.voice.stt import AzureSTT
        
        config = VoiceConfig(key="", region="eastus")
        with pytest.raises(ValueError, match="not configured"):
            AzureSTT(config)
    
    def test_raises_when_region_missing(self):
        """Should raise ValueError when region is missing."""
        from bot.voice.stt import AzureSTT
        
        config = VoiceConfig(key="test-key", region="")
        with pytest.raises(ValueError, match="not configured"):
            AzureSTT(config)


class TestCreateSTTFactory:
    """Test create_stt factory function."""
    
    def test_returns_none_when_config_is_none(self):
        """Should return None when config is None."""
        from bot.voice.stt import create_stt
        
        result = create_stt(None)
        assert result is None
    
    def test_returns_none_when_not_configured(self):
        """Should return None when config is not configured."""
        from bot.voice.stt import create_stt
        
        config = VoiceConfig(key="", region="eastus")
        result = create_stt(config)
        assert result is None
    
    @patch('bot.voice.stt.AzureSTT')
    def test_returns_stt_instance_when_configured(self, mock_stt_class):
        """Should return AzureSTT instance when config is valid."""
        from bot.voice.stt import create_stt
        
        mock_stt_instance = Mock()
        mock_stt_class.return_value = mock_stt_instance
        
        config = VoiceConfig(key="test-key", region="eastus")
        result = create_stt(config)
        
        assert result is mock_stt_instance
        mock_stt_class.assert_called_once_with(config)


class TestSTTIntegration:
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
        
        return VoiceConfig(key=key, region=region)
    
    def test_stt_can_be_instantiated(self, valid_config):
        """Should be able to create AzureSTT with valid credentials."""
        from bot.voice.stt import AzureSTT
        
        stt = AzureSTT(valid_config)
        assert stt is not None
        assert stt.config == valid_config
    
    def test_transcribe_with_dummy_audio(self, valid_config):
        """transcribe() should handle audio bytes."""
        from bot.voice.stt import AzureSTT
        
        stt = AzureSTT(valid_config)
        
        # Create minimal WAV header + silence (should return empty or error gracefully)
        # This is just to verify the method signature works
        dummy_audio = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        
        # This will likely fail with no-match since it's not real audio,
        # but should not raise an exception
        try:
            result = stt.transcribe(dummy_audio)
            # Should either return empty string (no match) or raise
            assert isinstance(result, str)
        except RuntimeError as e:
            # Cancellation or no-match is expected for dummy audio
            assert "NoMatch" in str(e) or "cancelled" in str(e).lower() or "canceled" in str(e).lower()
    
    def test_transcribe_file_not_found(self, valid_config):
        """transcribe_file() should raise RuntimeError for missing file."""
        from bot.voice.stt import AzureSTT
        
        stt = AzureSTT(valid_config)
        
        with pytest.raises(RuntimeError, match="Failed to read"):
            stt.transcribe_file("/nonexistent/path/audio.ogg")
    
    def test_transcribe_from_url_invalid(self, valid_config):
        """transcribe_from_url() should handle invalid URLs gracefully."""
        from bot.voice.stt import AzureSTT
        
        stt = AzureSTT(valid_config)
        
        # This should handle the invalid URL gracefully
        # Note: Azure SDK may raise an error for invalid URLs
        with pytest.raises(Exception):
            stt.transcribe_from_url("http://invalid-url-that-does-not-exist.com/audio.wav")