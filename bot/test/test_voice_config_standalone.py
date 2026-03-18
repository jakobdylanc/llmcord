"""
Standalone tests for voice config - no discord dependency.

Run with: python -m pytest bot/test/test_voice_config_standalone.py -v
"""

import pytest
import os
import sys
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Inline VoiceConfig and get_voice_config for testing (copied from bot/voice/config.py)
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VoiceConfig:
    """Configuration for Azure Speech Services."""
    key: str
    region: str
    endpoint: Optional[str] = None
    default_voice: Optional[str] = None
    default_style: Optional[str] = None
    
    @property
    def is_configured(self) -> bool:
        """Check if Azure Speech is properly configured."""
        return bool(self.key and self.region)


def get_voice_config(config: dict[str, Any]) -> Optional[VoiceConfig]:
    """Parse azure-speech section from config with env var overrides."""
    azure_speech = config.get("azure-speech")
    if azure_speech is None:
        return None
    
    if not isinstance(azure_speech, dict):
        return None
    
    # Get region - config.yaml or environment variable
    region = azure_speech.get("region") or os.getenv("AZURE_SPEECH_REGION")
    if not region:
        return None
    
    # Get key - config.yaml or environment variable
    key = azure_speech.get("key") or os.getenv("AZURE_SPEECH_KEY") or ""
    
    # Get optional fields - config.yaml or environment variables
    endpoint = azure_speech.get("endpoint") or os.getenv("AZURE_SPEECH_ENDPOINT") or None
    default_voice = azure_speech.get("default_voice") or os.getenv("AZURE_SPEECH_VOICE") or None
    default_style = azure_speech.get("default_style") or os.getenv("AZURE_SPEECH_STYLE") or None
    
    return VoiceConfig(
        key=key,
        region=region,
        endpoint=endpoint,
        default_voice=default_voice,
        default_style=default_style,
    )


class TestGetVoiceConfig:
    """Tests for get_voice_config function."""
    
    def test_returns_none_when_azure_speech_not_in_config(self):
        """Should return None when azure-speech section is missing."""
        config = {"providers": {}, "models": {}}
        result = get_voice_config(config)
        assert result is None
    
    def test_returns_none_when_azure_speech_is_none(self):
        """Should return None when azure-speech is null."""
        config = {"providers": {}, "models": {}, "azure-speech": None}
        result = get_voice_config(config)
        assert result is None
    
    def test_returns_none_when_azure_speech_is_empty_dict(self):
        """Should return None when azure-speech is empty dict."""
        config = {"providers": {}, "models": {}, "azure-speech": {}}
        result = get_voice_config(config)
        assert result is None
    
    def test_returns_none_when_region_is_missing(self):
        """Should return None when region is not provided."""
        config = {
            "providers": {},
            "models": {},
            "azure-speech": {"key": "test-key"}
        }
        result = get_voice_config(config)
        assert result is None
    
    def test_returns_voice_config_when_valid(self):
        """Should return VoiceConfig when azure-speech has required fields."""
        config = {
            "providers": {},
            "models": {},
            "azure-speech": {
                "key": "test-key-123",
                "region": "eastus"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert isinstance(result, VoiceConfig)
        assert result.key == "test-key-123"
        assert result.region == "eastus"
        assert result.is_configured is True
    
    def test_returns_voice_config_with_all_optional_fields(self):
        """Should include optional fields when provided."""
        config = {
            "providers": {},
            "models": {},
            "azure-speech": {
                "key": "test-key",
                "region": "westeurope",
                "endpoint": "https://custom.endpoint.com",
                "default_voice": "en-US-JennyNeural",
                "default_style": "cheerful"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert result.endpoint == "https://custom.endpoint.com"
        assert result.default_voice == "en-US-JennyNeural"
        assert result.default_style == "cheerful"
    
    def test_returns_voice_config_with_empty_key(self):
        """Should still return config even if key is empty (warns in validator)."""
        config = {
            "providers": {},
            "models": {},
            "azure-speech": {
                "key": "",
                "region": "eastus"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert result.is_configured is False  # Key is empty


class TestVoiceConfig:
    """Tests for VoiceConfig dataclass."""
    
    def test_is_configured_true_when_key_and_region_present(self):
        """is_configured should be True when key and region are present."""
        config = VoiceConfig(key="test", region="eastus")
        assert config.is_configured is True
    
    def test_is_configured_false_when_key_empty(self):
        """is_configured should be False when key is empty."""
        config = VoiceConfig(key="", region="eastus")
        assert config.is_configured is False
    
    def test_is_configured_false_when_region_empty(self):
        """is_configured should be False when region is empty."""
        config = VoiceConfig(key="test", region="")
        assert config.is_configured is False


class TestVoiceConfigEnvOverrides:
    """Tests for environment variable override functionality."""
    
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        """Clear Azure env vars before each test to avoid .env interference."""
        # First clear any existing env vars
        for key in ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT",
                    "AZURE_SPEECH_VOICE", "AZURE_SPEECH_STYLE"]:
            monkeypatch.delenv(key, raising=False)
        yield
    
    def test_env_var_overrides_region(self, monkeypatch):
        """Should use AZURE_SPEECH_REGION env var when region not in config."""
        monkeypatch.setenv("AZURE_SPEECH_REGION", "westeurope")
        monkeypatch.setenv("AZURE_SPEECH_KEY", "env-key")
        config = {"azure-speech": {}}
        result = get_voice_config(config)
        assert result is not None
        assert result.region == "westeurope"
        assert result.key == "env-key"
    
    def test_env_var_overrides_key(self, monkeypatch):
        """Should use AZURE_SPEECH_KEY env var when key not in config."""
        monkeypatch.setenv("AZURE_SPEECH_KEY", "env-key-123")
        config = {
            "azure-speech": {
                "region": "eastus"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert result.key == "env-key-123"
    
    def test_env_var_overrides_voice(self, monkeypatch):
        """Should use AZURE_SPEECH_VOICE env var for default_voice."""
        monkeypatch.setenv("AZURE_SPEECH_VOICE", "en-US-AriaNeural")
        config = {
            "azure-speech": {
                "key": "test",
                "region": "eastus"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert result.default_voice == "en-US-AriaNeural"
    
    def test_env_var_overrides_style(self, monkeypatch):
        """Should use AZURE_SPEECH_STYLE env var for default_style."""
        monkeypatch.setenv("AZURE_SPEECH_STYLE", "sad")
        config = {
            "azure-speech": {
                "key": "test",
                "region": "eastus"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert result.default_style == "sad"
    
    def test_config_overrides_env_when_both_present(self, monkeypatch):
        """Config.yaml values should take precedence over env vars."""
        monkeypatch.setenv("AZURE_SPEECH_REGION", "env-region")
        monkeypatch.setenv("AZURE_SPEECH_KEY", "env-key")
        config = {
            "azure-speech": {
                "key": "config-key",
                "region": "config-region"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert result.key == "config-key"
        assert result.region == "config-region"
    
    def test_env_var_endpoint(self, monkeypatch):
        """Should read AZURE_SPEECH_ENDPOINT from env var."""
        monkeypatch.setenv("AZURE_SPEECH_ENDPOINT", "https://custom.cris.ai")
        config = {
            "azure-speech": {
                "key": "test",
                "region": "eastus"
            }
        }
        result = get_voice_config(config)
        assert result is not None
        assert result.endpoint == "https://custom.cris.ai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])