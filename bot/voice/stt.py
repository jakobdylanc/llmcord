"""
Azure Speech-to-Text (STT) service.

Provides speech-to-text conversion using Azure Speech Services.
Can be tested independently without running the Discord bot.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

try:
    import azure.cognitiveservices.speech as speech
    _AZURE_SPEECH_AVAILABLE = True
except ImportError:
    _AZURE_SPEECH_AVAILABLE = False
    logging.warning("azure-cognitiveservices-speech not installed - STT disabled")

from .config import VoiceConfig


logger = logging.getLogger(__name__)


class AzureSTT:
    """
    Azure Speech-to-Text service.
    
    Provides speech-to-text conversion using Azure Speech Services.
    Can be tested independently without running the Discord bot.
    """
    
    def __init__(self, config: VoiceConfig):
        """
        Initialize Azure STT with the given config.
        
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
            self._speech_config.endpoint = config.endpoint
    
    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Audio data to transcribe (supports WAV, OGG, MP3)
            language: Language code (e.g., "en-US"). Auto-detects if not specified.
            
        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If transcription fails
        """
        # Create audio input from bytes
        audio_stream = speech.audio.PushAudioInputStream()
        
        # Write audio data to stream
        audio_stream.write(audio_bytes)
        audio_stream.close()
        
        audio_config = speech.audio.AudioConfig(stream=audio_stream)
        
        # Create recognizer
        recognizer = speech.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )
        
        # Set language if specified
        if language:
            recognizer.speech_recognition_language = language
        
        # Synchronous recognition
        result = recognizer.recognize_once()
        
        if result.reason == speech.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speech.ResultReason.NoMatch:
            logger.warning("No speech recognized in audio")
            return ""
        elif result.reason == speech.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"STT cancelled: {cancellation.reason}"
            if cancellation.error_details:
                error_msg += f" - {cancellation.error_details}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            raise RuntimeError(f"STT failed with reason: {result.reason}")
    
    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            file_path: Path to audio file (supports WAV, OGG, MP3)
            language: Language code (e.g., "en-US"). Auto-detects if not specified.
            
        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If transcription fails or file cannot be read
        """
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
        except IOError as e:
            raise RuntimeError(f"Failed to read audio file: {e}")
        
        return self.transcribe(audio_bytes, language)
    
    def transcribe_from_url(self, url: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio from a URL.
        
        Args:
            url: URL to audio file
            language: Language code (e.g., "en-US"). Auto-detects if not specified.
            
        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If transcription fails
        """
        audio_config = speech.audio.AudioConfig.from_wav_file_input(url)
        
        recognizer = speech.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )
        
        if language:
            recognizer.speech_recognition_language = language
        
        result = recognizer.recognize_once()
        
        if result.reason == speech.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speech.ResultReason.NoMatch:
            logger.warning("No speech recognized in audio from URL")
            return ""
        elif result.reason == speech.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise RuntimeError(f"STT cancelled: {cancellation.reason}")
        else:
            raise RuntimeError(f"STT failed with reason: {result.reason}")


def create_stt(config: VoiceConfig) -> Optional[AzureSTT]:
    """
    Factory function to create AzureSTT instance.
    
    Args:
        config: VoiceConfig with Azure Speech credentials
        
    Returns:
        AzureSTT instance if config is valid, None otherwise
    """
    if not config or not config.is_configured:
        logger.warning("Azure Speech not configured - STT disabled")
        return None
    
    try:
        return AzureSTT(config)
    except Exception as e:
        logger.error(f"Failed to initialize Azure STT: {e}")
        return None