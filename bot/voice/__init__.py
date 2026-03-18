"""
bot/voice - Azure Speech Services integration for Discord bot.

Provides TTS (Text-to-Speech) and STT (Speech-to-Text) capabilities.

Modules:
- config: Voice configuration loading
- tts: Azure Text-to-Speech service
- stt: Azure Speech-to-Text service
"""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from .config import get_voice_config, VoiceConfig
from .tts import AzureTTS, create_tts, Voice
from .stt import AzureSTT, create_stt


logger = logging.getLogger(__name__)


class VoiceCog(commands.Cog):
    """
    Voice cog for Discord bot.
    
    Provides /join and /leave commands for voice channel control,
    and handles TTS/STT for voice interactions.
    """
    
    def __init__(self, bot: commands.Bot, config: dict):
        """
        Initialize the VoiceCog.
        
        Args:
            bot: The Discord bot instance
            config: The full config dictionary
        """
        self.bot = bot
        self.config = config
        self.voice_config = get_voice_config(config)
        
        # Initialize TTS and STT if configured
        self.tts = create_tts(self.voice_config) if self.voice_config else None
        self.stt = create_stt(self.voice_config) if self.voice_config else None
        
        # Track current voice client
        self.voice_client: discord.VoiceClient | None = None
        
        logger.info(f"VoiceCog initialized - TTS: {self.tts is not None}, STT: {self.stt is not None}")
    
    # ── Slash Commands ─────────────────────────────────────────────────────
    
    @app_commands.command(name="join", description="Join your voice channel")
    async def join_command(self, interaction: discord.Interaction):
        """Make the bot join the user's voice channel."""
        # Check if user is in a voice channel
        if not interaction.user.voice:
            await interaction.response.send_message(
                "❌ You need to be in a voice channel first!",
                ephemeral=True
            )
            return
        
        voice_channel = interaction.user.voice.channel
        
        # Check if bot is already in a voice channel
        if interaction.guild.voice_client:
            await interaction.response.send_message(
                f"⚠️ I'm already in a voice channel! Use /leave first.",
                ephemeral=True
            )
            return
        
        # Join the voice channel
        try:
            await interaction.response.send_message(
                f"🎤 Joining {voice_channel.mention}...",
                ephemeral=True
            )
            vc = await voice_channel.connect()
            self.voice_client = vc
            logger.info(f"Bot joined voice channel: {voice_channel.name} (guild: {interaction.guild.name})")
            
            await interaction.followup.send(
                f"✅ Joined {voice_channel.mention}!",
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Failed to join voice channel: {e}")
            await interaction.followup.send(
                f"❌ Failed to join voice channel: {e}",
                ephemeral=True
            )
    
    @app_commands.command(name="leave", description="Leave the current voice channel")
    async def leave_command(self, interaction: discord.Interaction):
        """Make the bot leave its current voice channel."""
        # Check if bot is in a voice channel
        voice_client = interaction.guild.voice_client
        
        if not voice_client:
            await interaction.response.send_message(
                "❌ I'm not in a voice channel!",
                ephemeral=True
            )
            return
        
        # Disconnect from voice channel
        try:
            channel_name = voice_client.channel.name if voice_client.channel else "voice channel"
            await voice_client.disconnect()
            self.voice_client = None
            logger.info(f"Bot left voice channel: {channel_name} (guild: {interaction.guild.name})")
            
            await interaction.response.send_message(
                f"✅ Left {channel_name}!",
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Failed to leave voice channel: {e}")
            await interaction.response.send_message(
                f"❌ Failed to leave voice channel: {e}",
                ephemeral=True
            )
    
    # ── TTS Command ────────────────────────────────────────────────────────
    
    @app_commands.command(name="speak", description="Make the bot speak in voice chat")
    @app_commands.describe(text="Text to speak")
    async def speak_command(self, interaction: discord.Interaction, text: str):
        """Make the bot speak the given text in voice channel."""
        # Check if bot is in a voice channel
        voice_client = interaction.guild.voice_client
        
        if not voice_client:
            await interaction.response.send_message(
                "❌ I'm not in a voice channel! Use /join first.",
                ephemeral=True
            )
            return
        
        # Check if TTS is configured
        if not self.tts:
            await interaction.response.send_message(
                "❌ TTS is not configured. Add Azure Speech credentials to config.yaml.",
                ephemeral=True
            )
            return
        
        await interaction.response.send_message(
            "🎤 Speaking...",
            ephemeral=True
        )
        
        try:
            # Generate TTS audio (MP3 bytes)
            audio_bytes = self.tts.speak(text)
            
            # Save to temp file - FFmpegOpusAudio needs a file path
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            try:
                # Create audio source from temp file
                audio_source = discord.FFmpegOpusAudio(source=temp_path, bitrate=128)
                
                # Play audio
                voice_client.play(audio_source)
                
                await interaction.followup.send(
                    f"✅ Speaking: {text[:100]}...",
                    ephemeral=True
                )
            finally:
                # Clean up temp file after playback starts
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            await interaction.followup.send(
                f"❌ TTS failed: {e}",
                ephemeral=True
            )


async def setup_voice_cog(bot: commands.Bot, config: dict) -> VoiceCog:
    """
    Set up and add the voice cog to the bot.
    
    Args:
        bot: The Discord bot instance
        config: The full config dictionary
        
    Returns:
        The added VoiceCog instance
    """
    cog = VoiceCog(bot, config)
    await bot.add_cog(cog)
    logger.info("VoiceCog loaded")
    return cog


# Export for external use
__all__ = [
    "get_voice_config",
    "VoiceConfig",
    "AzureTTS",
    "create_tts",
    "Voice",
    "AzureSTT",
    "create_stt",
    "VoiceCog",
    "setup_voice_cog",
]