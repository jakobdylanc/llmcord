"""
Azure Text-to-Speech (TTS) service.

Provides text-to-speech conversion using Azure Speech Services.
Can be tested independently without running the Discord bot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    import azure.cognitiveservices.speech as speech
    _AZURE_SPEECH_AVAILABLE = True
except ImportError:
    _AZURE_SPEECH_AVAILABLE = False
    logging.warning("azure-cognitiveservices-speech not installed - TTS disabled")

from .config import VoiceConfig


logger = logging.getLogger(__name__)


@dataclass
class Voice:
    """Represents an Azure TTS voice."""
    name: str
    locale: str
    gender: str
    short_name: str
    
    @classmethod
    def from_voice_info(cls, info: speech.VoiceInfo) -> "Voice":
        # Handle gender - could be an enum or string
        gender = info.gender
        if hasattr(gender, 'name'):
            gender = gender.name.lower()
        elif hasattr(gender, 'lower'):
            gender = gender.lower()
        else:
            gender = str(gender).lower()
        
        return cls(
            name=info.name,
            locale=info.locale,
            gender=gender,
            short_name=info.short_name,
        )


class AzureTTS:
    """
    Azure Text-to-Speech service.
    
    Provides text-to-speech conversion using Azure Speech Services.
    Can be tested independently without running the Discord bot.
    """
    
    def __init__(self, config: VoiceConfig):
        """
        Initialize Azure TTS with the given config.
        
        Args:
            config: VoiceConfig with Azure Speech credentials
            
        Raises:
            ImportError: If azure-cognitiveservices-speech is not installed
            ValueError: If config is not properly configured
        """
        if not _AZURE_SPEECH_AVAILABLE:
            raise ImportError("azure-cognitiveservices-speech not installed")
        
        if not config.is_configured:
            raise ValueError("Azure Speech not configured (missing key or region)")
        
        self.config = config
        self._speech_config = speech.SpeechConfig(
            subscription=config.key,
            region=config.region
        )
        
        if config.endpoint:
            # Use custom endpoint if provided
            self._speech_config.endpoint = config.endpoint
        
        self._default_voice = config.default_voice or "en-US-JennyNeural"
        self._default_style = config.default_style or "neutral"
    
    def speak(self, text: str, voice: Optional[str] = None, style: Optional[str] = None) -> bytes:
        """
        Convert text to speech and return audio as bytes.
        
        Args:
            text: Text to convert to speech
            voice: Voice name (e.g., "en-US-JennyNeural"). Uses default if not specified.
            style: Speaking style (e.g., "cheerful", "sad"). Uses default if not specified.
            
        Returns:
            Audio data as bytes (MP3 format)
            
        Raises:
            RuntimeError: If TTS conversion fails
        """
        voice = voice or self._default_voice
        style = style or self._default_style
        
        # Determine locale from voice name (e.g., en-US-JennyNeural -> en-US, zh-CN-XiaoxiaoNeural -> zh-CN)
        locale = "en-US"
        if "-" in voice:
            locale_parts = voice.split("-")
            if len(locale_parts) >= 2:
                locale = f"{locale_parts[0]}-{locale_parts[1]}"
        
        # Configure audio output to return bytes in memory
        # Passing None gets in-memory bytes without trying to output to speaker
        audio_config = None
        
        # Create synthesizer with custom audio config
        synthesizer = speech.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )
        
        # Set up SSML for voice and style - use dynamic locale based on voice
        ssml = f"""
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{locale}'>
    <voice name='{voice}'>
        <style name='{style}'>{self._escape_xml(text)}</style>
    </voice>
</speak>
"""
        
        # Synchronous synthesis
        result = synthesizer.speak_ssml(ssml)
        
        if result.reason in (speech.ResultReason.SynthesizingAudio, speech.ResultReason.SynthesizingAudioCompleted):
            return result.audio_data
        elif result.reason == speech.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"TTS cancelled: {cancellation.reason}"
            if cancellation.error_details:
                error_msg += f" - {cancellation.error_details}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            raise RuntimeError(f"TTS failed with reason: {result.reason}")
    
    def speak_to_file(self, text: str, file_path: str, voice: Optional[str] = None, style: Optional[str] = None) -> None:
        """
        Convert text to speech and save to file.
        
        Args:
            text: Text to convert to speech
            file_path: Path to save audio file
            voice: Voice name (e.g., "en-US-JennyNeural")
            style: Speaking style (e.g., "cheerful", "sad")
        """
        audio_config = speech.audio.AudioOutputConfig(filename=file_path)
        synthesizer = speech.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )
        
        voice = voice or self._default_voice
        style = style or self._default_style
        
        ssml = f"""
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='{voice}'>
        <style name='{style}'>{self._escape_xml(text)}</style>
    </voice>
</speak>
"""
        
        result = synthesizer.speak_ssml(ssml)
        
        if result.reason != speech.ResultReason.SynthesizingAudio:
            cancellation = result.cancellation_details
            raise RuntimeError(f"TTS failed: {cancellation.reason if cancellation else result.reason}")
    
    def list_voices(self, locale: Optional[str] = None) -> list[Voice]:
        """
        List available Azure voices.
        
        Args:
            locale: Optional locale filter (e.g., "en-US")
            
        Returns:
            List of Voice objects
        """
        # Create a speech synthesizer to get voices
        # Pass None to get in-memory audio without speaker output
        audio_config = None
        synthesizer = speech.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )
        
        # Get voices
        voices = synthesizer.get_voices(locale=locale or "")
        voice_list = []
        
        for voice_info in voices.voices:
            voice_list.append(Voice.from_voice_info(voice_info))
        
        return voice_list
    
    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape special XML characters in text."""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
        )


def create_tts(config: VoiceConfig) -> Optional[AzureTTS]:
    """
    Factory function to create AzureTTS instance.
    
    Args:
        config: VoiceConfig with Azure Speech credentials
        
    Returns:
        AzureTTS instance if config is valid, None otherwise
    """
    if not config or not config.is_configured:
        logger.warning("Azure Speech not configured - TTS disabled")
        return None
    
    try:
        return AzureTTS(config)
    except Exception as e:
        logger.error(f"Failed to initialize Azure TTS: {e}")
        return None