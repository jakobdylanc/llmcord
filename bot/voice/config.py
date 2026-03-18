"""
Voice module configuration loader.

Parses azure-speech section from config.yaml and supports environment variable overrides.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional


# Pattern to match ${ENV_VAR} in config values
ENV_VAR_PATTERN = re.compile(r'\$\{(\w+)\}')


def _interpolate_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    if not isinstance(value, str):
        return value
    
    def replace_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    return ENV_VAR_PATTERN.sub(replace_var, value)


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
    """
    Parse azure-speech section from config.
    
    Environment variables (AZURE_SPEECH_*) can override config.yaml values:
    - AZURE_SPEECH_KEY: Override the key
    - AZURE_SPEECH_REGION: Override the region
    - AZURE_SPEECH_VOICE: Override default voice
    - AZURE_SPEECH_STYLE: Override default style
    
    Args:
        config: The full config dictionary from config.yaml
        
    Returns:
        VoiceConfig if azure-speech section exists and has region, None otherwise
    """
    azure_speech = config.get("azure-speech")
    if azure_speech is None:
        return None
    
    if not isinstance(azure_speech, dict):
        return None
    
    # Get and interpolate region (supports ${AZURE_SPEECH_REGION})
    region = _interpolate_env_vars(azure_speech.get("region", ""))
    if not region:
        return None
    
    # Get and interpolate key (supports ${AZURE_SPEECH_KEY})
    key = _interpolate_env_vars(azure_speech.get("key", ""))
    if not key:
        return None
    
    # Get optional fields with interpolation
    endpoint = _interpolate_env_vars(azure_speech.get("endpoint", ""))
    endpoint = endpoint if endpoint else None
    default_voice = _interpolate_env_vars(azure_speech.get("default_voice", ""))
    default_voice = default_voice if default_voice else None
    default_style = _interpolate_env_vars(azure_speech.get("default_style", ""))
    default_style = default_style if default_style else None
    
    return VoiceConfig(
        key=key,
        region=region,
        endpoint=endpoint,
        default_voice=default_voice,
        default_style=default_style,
    )