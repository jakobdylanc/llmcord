import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import re
import sys
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

# Discord Actions System Prompt Extension
DISCORD_ACTIONS_PROMPT = """

You have access to advanced Discord capabilities. When the user asks you to perform Discord operations, you can use special action commands that will be automatically executed.

To perform a Discord action, include it in your response using this format: [[ACTION:ACTION_NAME:param1:param2:...]]

**Available Actions:**

Channel Operations:
- [[ACTION:LIST_CHANNELS]] or [[ACTION:LIST_CHANNELS:text]] - List all channels (filter by: all/text/voice/forum/thread)
- [[ACTION:READ_CHANNEL:channel_name:limit]] - Read messages from a channel (limit is optional, default 20)
- [[ACTION:CHANNEL_INFO:channel_name]] - Get detailed info about a channel
- [[ACTION:FIND_CHANNEL_MENTIONS:channel_name]] - Find all channel mentions/tags in a channel
- [[ACTION:BROWSE_MENTIONED_CHANNELS:channel_name]] - Find channel mentions and read from each one

Server Operations:
- [[ACTION:SERVER_INFO]] - Get server information
- [[ACTION:LIST_ROLES]] - List all roles
- [[ACTION:LIST_MEMBERS]] or [[ACTION:LIST_MEMBERS:role_name]] - List members (optionally filter by role)
- [[ACTION:VOICE_STATUS]] - See who's in voice channels
- [[ACTION:LIST_EMOJIS]] - List custom emojis
- [[ACTION:SCHEDULED_EVENTS]] - View scheduled events

Message Operations:
- [[ACTION:SEARCH_MESSAGES:query:channel_name]] - Search for messages containing query
- [[ACTION:GET_PINNED:channel_name]] - Get pinned messages
- [[ACTION:MESSAGE_INFO:message_id]] - Get details about a specific message
- [[ACTION:ADD_REACTION:message_id:emoji]] - Add a reaction to a message
- [[ACTION:REMOVE_REACTION:message_id:emoji]] - Remove a reaction

Thread Operations:
- [[ACTION:LIST_THREADS:channel_name]] or [[ACTION:LIST_THREADS:channel_name:true]] - List threads (add true for archived)
- [[ACTION:CREATE_THREAD:message_id:thread_name]] - Create a thread from a message

User Operations:
- [[ACTION:USER_INFO:username_or_id]] - Get information about a user

**Examples:**
- User asks "what channels are in this server?" → Use [[ACTION:LIST_CHANNELS]]
- User asks "read the last 10 messages from #general" → Use [[ACTION:READ_CHANNEL:general:10]]
- User asks "go to #announcements and find all the channels mentioned there" → Use [[ACTION:FIND_CHANNEL_MENTIONS:announcements]]
- User asks "browse through all the channels tagged in #resources" → Use [[ACTION:BROWSE_MENTIONED_CHANNELS:resources]]
- User asks "who's online in voice?" → Use [[ACTION:VOICE_STATUS]]
- User asks "search for 'bug report' in #support" → Use [[ACTION:SEARCH_MESSAGES:bug report:support]]

When you use an action, briefly explain what you're doing. The action results will appear in a follow-up message.
You can use multiple actions in a single response if needed.
"""

# Event to signal when Discord bot is ready
bot_ready_event = asyncio.Event()


def clean_config_content(raw_bytes: bytes) -> str:
    """Clean config content by removing problematic characters."""
    # Try multiple encodings
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Last resort: decode with replacement
        decoded = raw_bytes.decode("utf-8", errors="replace")

    # Remove C1 control characters (0x80-0x9F) and other problematic chars
    # Keep printable chars, newlines, tabs, and common Unicode
    cleaned_chars = []
    for char in decoded:
        code = ord(char)
        # Skip C1 control characters (0x80-0x9F) - these cause YAML errors
        if 0x80 <= code <= 0x9F:
            continue
        # Skip replacement character
        if code == 0xFFFD:
            continue
        # Keep printable, newlines, tabs
        if char.isprintable() or char in '\n\r\t':
            cleaned_chars.append(char)

    return ''.join(cleaned_chars)


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Load config with robust encoding handling to prevent Unicode errors."""
    try:
        with open(filename, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        # If UTF-8 or YAML parsing fails, try cleaning the content
        logging.warning(f"Config file has issues, attempting to clean: {e}")
        with open(filename, "rb") as file:
            raw = file.read()
        cleaned = clean_config_content(raw)
        try:
            return yaml.safe_load(cleaned)
        except yaml.YAMLError as e2:
            logging.error(f"Failed to parse config even after cleaning: {e2}")
            raise


config = get_config()
if not config.get("models"):
    raise ValueError("No models configured in config.yaml. Please add at least one model.")
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
cli_conversation: list[dict[str, str]] = []  # Conversation history for CLI mode
cli_guild_id: Optional[int] = None  # Current server for CLI Discord actions
cli_channel_id: Optional[int] = None  # Current channel for CLI Discord actions

intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # Required for member listing functionality
intents.presences = True  # Required for online status in member listing
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()

# =============================================================================
# DISCORD ACTION SYSTEM - Advanced Discord Operations for LLM
# =============================================================================

# Action pattern for LLM to request Discord operations
# Format: [[ACTION:action_name:param1:param2:...]]
ACTION_PATTERN = re.compile(r'\[\[ACTION:([A-Z_]+)(?::([^\]]*))?\]\]')

class DiscordActions:
    """Handler for advanced Discord operations that the LLM can request."""

    @staticmethod
    async def list_channels(guild: discord.Guild, channel_type: str = "all") -> str:
        """List all channels in the server."""
        if not guild:
            return "Error: Not in a server context."

        result = [f"**Channels in {guild.name}:**\n"]

        # Group by category
        categories = {}
        no_category = []

        for channel in guild.channels:
            if isinstance(channel, discord.CategoryChannel):
                continue

            # Filter by type if specified
            if channel_type != "all":
                if channel_type == "text" and not isinstance(channel, discord.TextChannel):
                    continue
                elif channel_type == "voice" and not isinstance(channel, discord.VoiceChannel):
                    continue
                elif channel_type == "forum" and not isinstance(channel, discord.ForumChannel):
                    continue
                elif channel_type == "thread" and not isinstance(channel, discord.Thread):
                    continue

            category_name = channel.category.name if channel.category else None
            if category_name:
                if category_name not in categories:
                    categories[category_name] = []
                categories[category_name].append(channel)
            else:
                no_category.append(channel)

        def format_channel(ch):
            type_icon = "💬" if isinstance(ch, discord.TextChannel) else \
                       "🔊" if isinstance(ch, discord.VoiceChannel) else \
                       "📋" if isinstance(ch, discord.ForumChannel) else \
                       "🧵" if isinstance(ch, discord.Thread) else "📁"
            return f"  {type_icon} #{ch.name} (ID: {ch.id})"

        if no_category:
            result.append("**No Category:**")
            for ch in sorted(no_category, key=lambda x: x.position):
                result.append(format_channel(ch))

        for cat_name in sorted(categories.keys()):
            result.append(f"\n**{cat_name}:**")
            for ch in sorted(categories[cat_name], key=lambda x: x.position):
                result.append(format_channel(ch))

        return "\n".join(result)

    @staticmethod
    async def read_channel(channel: discord.TextChannel, limit: int = 20, before_id: int = None) -> str:
        """Read messages from a channel."""
        if not channel:
            return "Error: Channel not found."

        try:
            before = discord.Object(id=before_id) if before_id else None
            messages = [msg async for msg in channel.history(limit=min(limit, 50), before=before)]

            if not messages:
                return f"No messages found in #{channel.name}."

            result = [f"**Messages from #{channel.name}** (showing {len(messages)} messages):\n"]

            for msg in reversed(messages):
                timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M")
                author = msg.author.display_name
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content

                # Handle attachments
                attachments = ""
                if msg.attachments:
                    attachments = f" [📎 {len(msg.attachments)} attachment(s)]"

                # Handle embeds
                embeds = ""
                if msg.embeds:
                    embeds = f" [📋 {len(msg.embeds)} embed(s)]"

                # Handle reactions
                reactions = ""
                if msg.reactions:
                    reactions = " " + " ".join([f"{r.emoji}×{r.count}" for r in msg.reactions[:5]])

                result.append(f"[{timestamp}] **{author}**: {content}{attachments}{embeds}{reactions}")
                result.append(f"  └─ Message ID: {msg.id}")

            return "\n".join(result)
        except discord.Forbidden:
            return f"Error: No permission to read #{channel.name}."
        except Exception as e:
            return f"Error reading channel: {str(e)}"

    @staticmethod
    async def get_channel_info(channel: discord.abc.GuildChannel) -> str:
        """Get detailed information about a channel."""
        if not channel:
            return "Error: Channel not found."

        result = [f"**Channel Info: #{channel.name}**\n"]
        result.append(f"• ID: {channel.id}")
        result.append(f"• Type: {channel.type.name}")
        result.append(f"• Category: {channel.category.name if channel.category else 'None'}")
        result.append(f"• Position: {channel.position}")
        result.append(f"• Created: {channel.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if isinstance(channel, discord.TextChannel):
            result.append(f"• Topic: {channel.topic or 'No topic set'}")
            result.append(f"• NSFW: {channel.nsfw}")
            result.append(f"• Slowmode: {channel.slowmode_delay}s")

            # Count threads
            threads = channel.threads
            result.append(f"• Active Threads: {len(threads)}")

        elif isinstance(channel, discord.VoiceChannel):
            result.append(f"• Bitrate: {channel.bitrate // 1000}kbps")
            result.append(f"• User Limit: {channel.user_limit or 'Unlimited'}")
            result.append(f"• Members Connected: {len(channel.members)}")
            if channel.members:
                result.append(f"• Connected Users: {', '.join([m.display_name for m in channel.members[:10]])}")

        elif isinstance(channel, discord.ForumChannel):
            result.append(f"• Topic: {channel.topic or 'No topic set'}")
            result.append(f"• Available Tags: {', '.join([t.name for t in channel.available_tags])}")

        return "\n".join(result)

    @staticmethod
    async def find_channel_mentions(channel: discord.TextChannel, limit: int = 50) -> str:
        """Find all channel mentions in a channel and return info about them."""
        if not channel:
            return "Error: Channel not found."

        try:
            messages = [msg async for msg in channel.history(limit=limit)]

            # Pattern to find channel mentions: <#channel_id>
            channel_pattern = re.compile(r'<#(\d+)>')
            mentioned_channels = {}

            for msg in messages:
                matches = channel_pattern.findall(msg.content)
                for channel_id in matches:
                    if channel_id not in mentioned_channels:
                        mentioned_channels[channel_id] = {
                            'count': 0,
                            'channel': channel.guild.get_channel(int(channel_id))
                        }
                    mentioned_channels[channel_id]['count'] += 1

            if not mentioned_channels:
                return f"No channel mentions found in the last {limit} messages of #{channel.name}."

            result = [f"**Channel Mentions in #{channel.name}** (last {limit} messages):\n"]

            for ch_id, data in sorted(mentioned_channels.items(), key=lambda x: x[1]['count'], reverse=True):
                ch = data['channel']
                if ch:
                    result.append(f"• <#{ch.id}> ({ch.name}) - mentioned {data['count']} time(s)")
                else:
                    result.append(f"• Unknown/Deleted Channel (ID: {ch_id}) - mentioned {data['count']} time(s)")

            return "\n".join(result)
        except discord.Forbidden:
            return f"Error: No permission to read #{channel.name}."
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    async def browse_mentioned_channels(source_channel: discord.TextChannel, message_limit: int = 50, read_limit: int = 10) -> str:
        """Find channel mentions in a channel, then read recent messages from each mentioned channel."""
        if not source_channel:
            return "Error: Source channel not found."

        try:
            messages = [msg async for msg in source_channel.history(limit=message_limit)]

            channel_pattern = re.compile(r'<#(\d+)>')
            mentioned_channel_ids = set()

            for msg in messages:
                matches = channel_pattern.findall(msg.content)
                mentioned_channel_ids.update(matches)

            if not mentioned_channel_ids:
                return f"No channel mentions found in #{source_channel.name}."

            result = [f"**Browsing channels mentioned in #{source_channel.name}:**\n"]

            for ch_id in mentioned_channel_ids:
                ch = source_channel.guild.get_channel(int(ch_id))
                if ch and isinstance(ch, discord.TextChannel):
                    result.append(f"\n{'='*40}")
                    result.append(f"**#{ch.name}** (ID: {ch.id})")
                    result.append(f"Topic: {ch.topic or 'No topic'}")
                    result.append("-" * 40)

                    try:
                        ch_messages = [m async for m in ch.history(limit=read_limit)]
                        for msg in reversed(ch_messages):
                            author = msg.author.display_name
                            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                            result.append(f"  [{msg.created_at.strftime('%H:%M')}] {author}: {content}")
                    except discord.Forbidden:
                        result.append("  (No permission to read this channel)")

            return "\n".join(result)
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    async def get_server_info(guild: discord.Guild) -> str:
        """Get detailed server information."""
        if not guild:
            return "Error: Not in a server context."

        result = [f"**Server: {guild.name}**\n"]
        result.append(f"• ID: {guild.id}")
        result.append(f"• Owner: {guild.owner.display_name if guild.owner else 'Unknown'}")
        result.append(f"• Created: {guild.created_at.strftime('%Y-%m-%d')}")
        result.append(f"• Members: {guild.member_count}")
        result.append(f"• Channels: {len(guild.channels)}")
        result.append(f"• Roles: {len(guild.roles)}")
        result.append(f"• Emojis: {len(guild.emojis)}")
        result.append(f"• Boost Level: {guild.premium_tier}")
        result.append(f"• Boost Count: {guild.premium_subscription_count}")
        result.append(f"• Verification Level: {guild.verification_level.name}")

        if guild.description:
            result.append(f"• Description: {guild.description}")

        if guild.icon:
            result.append(f"• Icon URL: {guild.icon.url}")

        return "\n".join(result)

    @staticmethod
    async def list_roles(guild: discord.Guild) -> str:
        """List all roles in the server."""
        if not guild:
            return "Error: Not in a server context."

        result = [f"**Roles in {guild.name}** ({len(guild.roles)} total):\n"]

        for role in sorted(guild.roles, key=lambda r: r.position, reverse=True):
            if role.name == "@everyone":
                continue

            color = f"#{role.color.value:06x}" if role.color.value else "No color"
            members = len(role.members)
            perms = []
            if role.permissions.administrator:
                perms.append("Admin")
            if role.permissions.manage_guild:
                perms.append("Manage Server")
            if role.permissions.manage_channels:
                perms.append("Manage Channels")
            if role.permissions.manage_messages:
                perms.append("Manage Messages")

            perms_str = f" [{', '.join(perms)}]" if perms else ""
            result.append(f"• {role.name} ({color}) - {members} members{perms_str}")
            result.append(f"  └─ ID: {role.id}")

        return "\n".join(result)

    @staticmethod
    async def list_members(guild: discord.Guild, role_filter: str = None, limit: int = 50) -> str:
        """List members in the server, optionally filtered by role."""
        if not guild:
            return "Error: Not in a server context."

        members = guild.members

        if role_filter:
            role = discord.utils.find(lambda r: r.name.lower() == role_filter.lower() or str(r.id) == role_filter, guild.roles)
            if role:
                members = [m for m in members if role in m.roles]
            else:
                return f"Role '{role_filter}' not found."

        members = members[:limit]

        result = [f"**Members in {guild.name}** (showing {len(members)}):\n"]

        for member in sorted(members, key=lambda m: m.display_name.lower()):
            status = "🟢" if member.status == discord.Status.online else \
                    "🟡" if member.status == discord.Status.idle else \
                    "🔴" if member.status == discord.Status.dnd else "⚫"

            roles = [r.name for r in member.roles if r.name != "@everyone"][:3]
            roles_str = f" [{', '.join(roles)}]" if roles else ""

            result.append(f"{status} {member.display_name} (@{member.name}){roles_str}")
            result.append(f"  └─ ID: {member.id}, Joined: {member.joined_at.strftime('%Y-%m-%d') if member.joined_at else 'Unknown'}")

        return "\n".join(result)

    @staticmethod
    async def get_user_info(member: discord.Member) -> str:
        """Get detailed information about a user."""
        if not member:
            return "Error: User not found."

        result = [f"**User Info: {member.display_name}**\n"]
        result.append(f"• Username: @{member.name}")
        result.append(f"• ID: {member.id}")
        result.append(f"• Display Name: {member.display_name}")
        result.append(f"• Status: {member.status.name}")
        result.append(f"• Bot: {'Yes' if member.bot else 'No'}")
        result.append(f"• Account Created: {member.created_at.strftime('%Y-%m-%d %H:%M')}")
        result.append(f"• Joined Server: {member.joined_at.strftime('%Y-%m-%d %H:%M') if member.joined_at else 'Unknown'}")

        if member.premium_since:
            result.append(f"• Boosting Since: {member.premium_since.strftime('%Y-%m-%d')}")

        roles = [r.name for r in member.roles if r.name != "@everyone"]
        if roles:
            result.append(f"• Roles: {', '.join(roles)}")

        if member.activity:
            result.append(f"• Activity: {member.activity.name}")

        if member.avatar:
            result.append(f"• Avatar URL: {member.avatar.url}")

        return "\n".join(result)

    @staticmethod
    async def search_messages(channel: discord.TextChannel, query: str, limit: int = 50) -> str:
        """Search for messages containing a query in a channel."""
        if not channel:
            return "Error: Channel not found."

        try:
            messages = [msg async for msg in channel.history(limit=limit)]
            matches = [msg for msg in messages if query.lower() in msg.content.lower()]

            if not matches:
                return f"No messages containing '{query}' found in #{channel.name}."

            result = [f"**Search Results for '{query}' in #{channel.name}** ({len(matches)} found):\n"]

            for msg in matches[:20]:
                timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M")
                author = msg.author.display_name
                content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                result.append(f"[{timestamp}] **{author}**: {content}")
                result.append(f"  └─ Message ID: {msg.id}")

            return "\n".join(result)
        except discord.Forbidden:
            return f"Error: No permission to read #{channel.name}."
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    async def get_pinned_messages(channel: discord.TextChannel) -> str:
        """Get pinned messages from a channel."""
        if not channel:
            return "Error: Channel not found."

        try:
            pins = await channel.pins()

            if not pins:
                return f"No pinned messages in #{channel.name}."

            result = [f"**Pinned Messages in #{channel.name}** ({len(pins)} pins):\n"]

            for msg in pins:
                timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M")
                author = msg.author.display_name
                content = msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
                result.append(f"📌 [{timestamp}] **{author}**: {content}")
                result.append(f"  └─ Message ID: {msg.id}")

            return "\n".join(result)
        except discord.Forbidden:
            return f"Error: No permission to read pins in #{channel.name}."
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    async def list_threads(channel: discord.TextChannel, include_archived: bool = False) -> str:
        """List threads in a channel."""
        if not channel:
            return "Error: Channel not found."

        try:
            threads = channel.threads

            if include_archived:
                archived = [t async for t in channel.archived_threads(limit=25)]
                threads = list(threads) + archived

            if not threads:
                return f"No threads found in #{channel.name}."

            result = [f"**Threads in #{channel.name}** ({len(threads)} threads):\n"]

            for thread in threads:
                status = "🟢 Active" if not thread.archived else "📁 Archived"
                result.append(f"• {thread.name} - {status}")
                result.append(f"  └─ ID: {thread.id}, Messages: {thread.message_count}, Members: {thread.member_count}")

            return "\n".join(result)
        except discord.Forbidden:
            return f"Error: No permission to view threads in #{channel.name}."
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    async def get_voice_status(guild: discord.Guild) -> str:
        """Get information about who's in voice channels."""
        if not guild:
            return "Error: Not in a server context."

        result = [f"**Voice Channel Status in {guild.name}:**\n"]

        voice_channels = [ch for ch in guild.channels if isinstance(ch, discord.VoiceChannel)]
        has_members = False

        for vc in sorted(voice_channels, key=lambda x: x.position):
            if vc.members:
                has_members = True
                result.append(f"🔊 **{vc.name}** ({len(vc.members)}/{vc.user_limit or '∞'}):")
                for member in vc.members:
                    status = ""
                    if member.voice:
                        if member.voice.self_mute:
                            status += "🔇"
                        if member.voice.self_deaf:
                            status += "🔕"
                        if member.voice.self_stream:
                            status += "🎥"
                    result.append(f"  • {member.display_name} {status}")

        if not has_members:
            result.append("No members in voice channels.")

        return "\n".join(result)

    @staticmethod
    async def list_emojis(guild: discord.Guild) -> str:
        """List all custom emojis in the server."""
        if not guild:
            return "Error: Not in a server context."

        if not guild.emojis:
            return f"No custom emojis in {guild.name}."

        result = [f"**Custom Emojis in {guild.name}** ({len(guild.emojis)} emojis):\n"]

        animated = [e for e in guild.emojis if e.animated]
        static = [e for e in guild.emojis if not e.animated]

        if static:
            result.append("**Static Emojis:**")
            result.append(" ".join([f"{e}" for e in static]))

        if animated:
            result.append("\n**Animated Emojis:**")
            result.append(" ".join([f"{e}" for e in animated]))

        return "\n".join(result)

    @staticmethod
    async def get_scheduled_events(guild: discord.Guild) -> str:
        """Get scheduled events in the server."""
        if not guild:
            return "Error: Not in a server context."

        events = guild.scheduled_events

        if not events:
            return f"No scheduled events in {guild.name}."

        result = [f"**Scheduled Events in {guild.name}:**\n"]

        for event in sorted(events, key=lambda e: e.start_time):
            status = "🟢 Active" if event.status == discord.EventStatus.active else \
                    "📅 Scheduled" if event.status == discord.EventStatus.scheduled else \
                    "✅ Completed"

            result.append(f"• **{event.name}** - {status}")
            result.append(f"  Start: {event.start_time.strftime('%Y-%m-%d %H:%M')}")
            if event.end_time:
                result.append(f"  End: {event.end_time.strftime('%Y-%m-%d %H:%M')}")
            if event.description:
                result.append(f"  Description: {event.description[:200]}")
            result.append(f"  Interested: {event.user_count} users")

        return "\n".join(result)

    @staticmethod
    async def add_reaction(message: discord.Message, emoji: str) -> str:
        """Add a reaction to a message."""
        if not message:
            return "Error: Message not found."

        try:
            await message.add_reaction(emoji)
            return f"Added reaction {emoji} to message."
        except discord.HTTPException as e:
            return f"Error adding reaction: {str(e)}"

    @staticmethod
    async def remove_reaction(message: discord.Message, emoji: str) -> str:
        """Remove bot's reaction from a message."""
        if not message:
            return "Error: Message not found."

        try:
            await message.remove_reaction(emoji, message.guild.me)
            return f"Removed reaction {emoji} from message."
        except discord.HTTPException as e:
            return f"Error removing reaction: {str(e)}"

    @staticmethod
    async def get_message_info(message: discord.Message) -> str:
        """Get detailed information about a specific message."""
        if not message:
            return "Error: Message not found."

        result = [f"**Message Info:**\n"]
        result.append(f"• ID: {message.id}")
        result.append(f"• Author: {message.author.display_name} (@{message.author.name})")
        result.append(f"• Channel: #{message.channel.name}")
        result.append(f"• Created: {message.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if message.edited_at:
            result.append(f"• Edited: {message.edited_at.strftime('%Y-%m-%d %H:%M:%S')}")

        result.append(f"• Pinned: {'Yes' if message.pinned else 'No'}")
        result.append(f"• Type: {message.type.name}")

        if message.content:
            result.append(f"\n**Content:**\n{message.content}")

        if message.attachments:
            result.append(f"\n**Attachments ({len(message.attachments)}):**")
            for att in message.attachments:
                result.append(f"  • {att.filename} ({att.content_type})")

        if message.embeds:
            result.append(f"\n**Embeds ({len(message.embeds)}):**")
            for embed in message.embeds:
                if embed.title:
                    result.append(f"  • Title: {embed.title}")
                if embed.description:
                    result.append(f"    Description: {embed.description[:200]}")

        if message.reactions:
            result.append(f"\n**Reactions:**")
            result.append(" ".join([f"{r.emoji}×{r.count}" for r in message.reactions]))

        if message.reference:
            result.append(f"\n**Reply to:** Message ID {message.reference.message_id}")

        return "\n".join(result)

    @staticmethod
    async def create_thread(message: discord.Message, name: str, auto_archive: int = 1440) -> str:
        """Create a thread from a message."""
        if not message:
            return "Error: Message not found."

        try:
            thread = await message.create_thread(name=name, auto_archive_duration=auto_archive)
            return f"Created thread '{thread.name}' (ID: {thread.id})"
        except discord.HTTPException as e:
            return f"Error creating thread: {str(e)}"

    @staticmethod
    async def fetch_message_by_id(channel: discord.TextChannel, message_id: int) -> discord.Message:
        """Fetch a specific message by ID."""
        try:
            return await channel.fetch_message(message_id)
        except (discord.NotFound, discord.HTTPException):
            return None

    @staticmethod
    async def get_channel_by_name_or_id(guild: discord.Guild, identifier: str) -> discord.abc.GuildChannel:
        """Get a channel by name or ID."""
        if not guild:
            return None

        # Try as ID first
        try:
            channel_id = int(identifier.strip('<#>'))
            channel = guild.get_channel(channel_id)
            if channel:
                return channel
        except ValueError:
            pass

        # Try as name
        identifier_lower = identifier.lower().strip('#')
        for channel in guild.channels:
            if channel.name.lower() == identifier_lower:
                return channel

        return None

    @staticmethod
    async def get_user_by_name_or_id(guild: discord.Guild, identifier: str) -> discord.Member:
        """Get a member by name, display name, or ID."""
        if not guild:
            return None

        # Try as ID first
        try:
            user_id = int(identifier.strip('<@!>'))
            member = guild.get_member(user_id)
            if member:
                return member
        except ValueError:
            pass

        # Try as name
        identifier_lower = identifier.lower().strip('@')
        for member in guild.members:
            if member.name.lower() == identifier_lower or member.display_name.lower() == identifier_lower:
                return member

        return None


async def execute_discord_action(action: str, params: str, context_msg: discord.Message) -> str:
    """Execute a Discord action and return the result."""
    guild = context_msg.guild
    channel = context_msg.channel

    params_list = params.split(':') if params else []

    action_map = {
        'LIST_CHANNELS': lambda: DiscordActions.list_channels(guild, params_list[0] if params_list else "all"),
        'READ_CHANNEL': lambda: read_channel_action(guild, params_list),
        'CHANNEL_INFO': lambda: channel_info_action(guild, params_list),
        'FIND_CHANNEL_MENTIONS': lambda: find_mentions_action(guild, params_list),
        'BROWSE_MENTIONED_CHANNELS': lambda: browse_mentions_action(guild, params_list),
        'SERVER_INFO': lambda: DiscordActions.get_server_info(guild),
        'LIST_ROLES': lambda: DiscordActions.list_roles(guild),
        'LIST_MEMBERS': lambda: DiscordActions.list_members(guild, params_list[0] if params_list else None),
        'USER_INFO': lambda: user_info_action(guild, params_list),
        'SEARCH_MESSAGES': lambda: search_action(guild, channel, params_list),
        'GET_PINNED': lambda: pinned_action(guild, channel, params_list),
        'LIST_THREADS': lambda: threads_action(guild, channel, params_list),
        'VOICE_STATUS': lambda: DiscordActions.get_voice_status(guild),
        'LIST_EMOJIS': lambda: DiscordActions.list_emojis(guild),
        'SCHEDULED_EVENTS': lambda: DiscordActions.get_scheduled_events(guild),
        'ADD_REACTION': lambda: reaction_action(channel, params_list, add=True),
        'REMOVE_REACTION': lambda: reaction_action(channel, params_list, add=False),
        'MESSAGE_INFO': lambda: message_info_action(channel, params_list),
        'CREATE_THREAD': lambda: create_thread_action(channel, params_list),
    }

    if action not in action_map:
        return f"Unknown action: {action}"

    return await action_map[action]()


async def read_channel_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    limit = int(params[1]) if len(params) > 1 else 20
    return await DiscordActions.read_channel(ch, limit)


async def channel_info_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    return await DiscordActions.get_channel_info(ch)


async def find_mentions_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    limit = int(params[1]) if len(params) > 1 else 50
    return await DiscordActions.find_channel_mentions(ch, limit)


async def browse_mentions_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    msg_limit = int(params[1]) if len(params) > 1 else 50
    read_limit = int(params[2]) if len(params) > 2 else 10
    return await DiscordActions.browse_mentioned_channels(ch, msg_limit, read_limit)


async def user_info_action(guild, params):
    if not params:
        return "Error: User name/ID required."
    member = await DiscordActions.get_user_by_name_or_id(guild, params[0])
    return await DiscordActions.get_user_info(member)


async def search_action(guild, channel, params):
    if len(params) < 1:
        return "Error: Search query required."
    query = params[0]
    if len(params) > 1:
        ch = await DiscordActions.get_channel_by_name_or_id(guild, params[1])
    else:
        ch = channel
    limit = int(params[2]) if len(params) > 2 else 50
    return await DiscordActions.search_messages(ch, query, limit)


async def pinned_action(guild, channel, params):
    if params:
        ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    else:
        ch = channel
    return await DiscordActions.get_pinned_messages(ch)


async def threads_action(guild, channel, params):
    if params:
        ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    else:
        ch = channel
    include_archived = len(params) > 1 and params[1].lower() == 'true'
    return await DiscordActions.list_threads(ch, include_archived)


async def reaction_action(channel, params, add=True):
    if len(params) < 2:
        return "Error: Message ID and emoji required."
    msg = await DiscordActions.fetch_message_by_id(channel, int(params[0]))
    if add:
        return await DiscordActions.add_reaction(msg, params[1])
    return await DiscordActions.remove_reaction(msg, params[1])


async def message_info_action(channel, params):
    if not params:
        return "Error: Message ID required."
    msg = await DiscordActions.fetch_message_by_id(channel, int(params[0]))
    return await DiscordActions.get_message_info(msg)


async def create_thread_action(channel, params):
    if len(params) < 2:
        return "Error: Message ID and thread name required."
    msg = await DiscordActions.fetch_message_by_id(channel, int(params[0]))
    return await DiscordActions.create_thread(msg, params[1])


# =============================================================================
# CLI DISCORD ACTION SUPPORT
# =============================================================================

class CLIActionContext:
    """Mock message context for CLI Discord actions."""
    def __init__(self, guild: discord.Guild, channel: discord.TextChannel):
        self.guild = guild
        self.channel = channel


async def execute_cli_discord_action(action: str, params: str) -> str:
    """Execute a Discord action from CLI context."""
    global cli_guild_id, cli_channel_id

    if cli_guild_id is None:
        return "Error: No server selected. Use /server to select a server first."

    guild = discord_bot.get_guild(cli_guild_id)
    if not guild:
        return f"Error: Could not find server with ID {cli_guild_id}. Use /servers to list available servers."

    channel = None
    if cli_channel_id:
        channel = guild.get_channel(cli_channel_id)

    # Create a mock context for the action
    context = CLIActionContext(guild, channel)

    params_list = params.split(':') if params else []

    # Map actions to their handlers
    action_map = {
        'LIST_CHANNELS': lambda: DiscordActions.list_channels(guild, params_list[0] if params_list else "all"),
        'READ_CHANNEL': lambda: cli_read_channel_action(guild, params_list),
        'CHANNEL_INFO': lambda: cli_channel_info_action(guild, params_list),
        'FIND_CHANNEL_MENTIONS': lambda: cli_find_mentions_action(guild, params_list),
        'BROWSE_MENTIONED_CHANNELS': lambda: cli_browse_mentions_action(guild, params_list),
        'SERVER_INFO': lambda: DiscordActions.get_server_info(guild),
        'LIST_ROLES': lambda: DiscordActions.list_roles(guild),
        'LIST_MEMBERS': lambda: DiscordActions.list_members(guild, params_list[0] if params_list else None),
        'USER_INFO': lambda: cli_user_info_action(guild, params_list),
        'SEARCH_MESSAGES': lambda: cli_search_action(guild, channel, params_list),
        'GET_PINNED': lambda: cli_pinned_action(guild, channel, params_list),
        'LIST_THREADS': lambda: cli_threads_action(guild, channel, params_list),
        'VOICE_STATUS': lambda: DiscordActions.get_voice_status(guild),
        'LIST_EMOJIS': lambda: DiscordActions.list_emojis(guild),
        'SCHEDULED_EVENTS': lambda: DiscordActions.get_scheduled_events(guild),
        'ADD_REACTION': lambda: cli_reaction_action(channel, params_list, add=True),
        'REMOVE_REACTION': lambda: cli_reaction_action(channel, params_list, add=False),
        'MESSAGE_INFO': lambda: cli_message_info_action(channel, params_list),
        'CREATE_THREAD': lambda: cli_create_thread_action(channel, params_list),
    }

    if action not in action_map:
        return f"Unknown action: {action}"

    return await action_map[action]()


# CLI-specific action helpers (use current channel from CLI context)
async def cli_read_channel_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    limit = int(params[1]) if len(params) > 1 else 20
    return await DiscordActions.read_channel(ch, limit)


async def cli_channel_info_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    return await DiscordActions.get_channel_info(ch)


async def cli_find_mentions_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    limit = int(params[1]) if len(params) > 1 else 50
    return await DiscordActions.find_channel_mentions(ch, limit)


async def cli_browse_mentions_action(guild, params):
    if not params:
        return "Error: Channel name/ID required."
    ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    msg_limit = int(params[1]) if len(params) > 1 else 50
    read_limit = int(params[2]) if len(params) > 2 else 10
    return await DiscordActions.browse_mentioned_channels(ch, msg_limit, read_limit)


async def cli_user_info_action(guild, params):
    if not params:
        return "Error: User name/ID required."
    member = await DiscordActions.get_user_by_name_or_id(guild, params[0])
    return await DiscordActions.get_user_info(member)


async def cli_search_action(guild, channel, params):
    if len(params) < 1:
        return "Error: Search query required."
    query = params[0]
    if len(params) > 1:
        ch = await DiscordActions.get_channel_by_name_or_id(guild, params[1])
    elif channel:
        ch = channel
    else:
        return "Error: No channel specified. Use /channel to set one or provide channel name."
    limit = int(params[2]) if len(params) > 2 else 50
    return await DiscordActions.search_messages(ch, query, limit)


async def cli_pinned_action(guild, channel, params):
    if params:
        ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    elif channel:
        ch = channel
    else:
        return "Error: No channel specified."
    return await DiscordActions.get_pinned_messages(ch)


async def cli_threads_action(guild, channel, params):
    if params:
        ch = await DiscordActions.get_channel_by_name_or_id(guild, params[0])
    elif channel:
        ch = channel
    else:
        return "Error: No channel specified."
    include_archived = len(params) > 1 and params[1].lower() == 'true'
    return await DiscordActions.list_threads(ch, include_archived)


async def cli_reaction_action(channel, params, add=True):
    if not channel:
        return "Error: No channel set. Use /channel first."
    if len(params) < 2:
        return "Error: Message ID and emoji required."
    msg = await DiscordActions.fetch_message_by_id(channel, int(params[0]))
    if add:
        return await DiscordActions.add_reaction(msg, params[1])
    return await DiscordActions.remove_reaction(msg, params[1])


async def cli_message_info_action(channel, params):
    if not channel:
        return "Error: No channel set. Use /channel first."
    if not params:
        return "Error: Message ID required."
    msg = await DiscordActions.fetch_message_by_id(channel, int(params[0]))
    return await DiscordActions.get_message_info(msg)


async def cli_create_thread_action(channel, params):
    if not channel:
        return "Error: No channel set. Use /channel first."
    if len(params) < 2:
        return "Error: Message ID and thread name required."
    msg = await DiscordActions.fetch_message_by_id(channel, int(params[0]))
    return await DiscordActions.create_thread(msg, params[1])


# =============================================================================
# END DISCORD ACTION SYSTEM
# =============================================================================


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


# =============================================================================
# ADVANCED DISCORD SLASH COMMANDS
# =============================================================================

@discord_bot.tree.command(name="channels", description="List all channels in the server")
async def channels_command(interaction: discord.Interaction, channel_type: str = "all") -> None:
    """List channels in the server. channel_type can be: all, text, voice, forum, thread"""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.list_channels(interaction.guild, channel_type)
    # Split if too long
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="read", description="Read recent messages from a channel")
async def read_command(interaction: discord.Interaction, channel: discord.TextChannel, limit: int = 20) -> None:
    """Read messages from a specific channel."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.read_channel(channel, min(limit, 50))
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="channelinfo", description="Get detailed information about a channel")
async def channelinfo_command(interaction: discord.Interaction, channel: discord.abc.GuildChannel) -> None:
    """Get detailed info about a channel."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.get_channel_info(channel)
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="mentions", description="Find all channel mentions in a channel")
async def mentions_command(interaction: discord.Interaction, channel: discord.TextChannel, limit: int = 50) -> None:
    """Find channel mentions in the specified channel."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.find_channel_mentions(channel, limit)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="browse", description="Browse through all channels mentioned in a channel")
async def browse_command(interaction: discord.Interaction, channel: discord.TextChannel, message_limit: int = 50, read_limit: int = 10) -> None:
    """Find channel mentions and read from each mentioned channel."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.browse_mentioned_channels(channel, message_limit, read_limit)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="serverinfo", description="Get detailed server information")
async def serverinfo_command(interaction: discord.Interaction) -> None:
    """Get server info."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.get_server_info(interaction.guild)
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="roles", description="List all roles in the server")
async def roles_command(interaction: discord.Interaction) -> None:
    """List all roles."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.list_roles(interaction.guild)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="members", description="List members in the server")
async def members_command(interaction: discord.Interaction, role: discord.Role = None, limit: int = 50) -> None:
    """List members, optionally filtered by role."""
    await interaction.response.defer(ephemeral=True)
    role_filter = role.name if role else None
    result = await DiscordActions.list_members(interaction.guild, role_filter, limit)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="userinfo", description="Get detailed information about a user")
async def userinfo_command(interaction: discord.Interaction, user: discord.Member) -> None:
    """Get detailed user info."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.get_user_info(user)
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="search", description="Search for messages in a channel")
async def search_command(interaction: discord.Interaction, query: str, channel: discord.TextChannel = None, limit: int = 50) -> None:
    """Search messages in a channel."""
    await interaction.response.defer(ephemeral=True)
    search_channel = channel or interaction.channel
    result = await DiscordActions.search_messages(search_channel, query, limit)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="pinned", description="Get pinned messages from a channel")
async def pinned_command(interaction: discord.Interaction, channel: discord.TextChannel = None) -> None:
    """Get pinned messages."""
    await interaction.response.defer(ephemeral=True)
    target_channel = channel or interaction.channel
    result = await DiscordActions.get_pinned_messages(target_channel)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="threads", description="List threads in a channel")
async def threads_command(interaction: discord.Interaction, channel: discord.TextChannel = None, include_archived: bool = False) -> None:
    """List threads in a channel."""
    await interaction.response.defer(ephemeral=True)
    target_channel = channel or interaction.channel
    result = await DiscordActions.list_threads(target_channel, include_archived)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="voice", description="See who's in voice channels")
async def voice_command(interaction: discord.Interaction) -> None:
    """Get voice channel status."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.get_voice_status(interaction.guild)
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="emojis", description="List all custom emojis in the server")
async def emojis_command(interaction: discord.Interaction) -> None:
    """List custom emojis."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.list_emojis(interaction.guild)
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="events", description="View scheduled events")
async def events_command(interaction: discord.Interaction) -> None:
    """Get scheduled events."""
    await interaction.response.defer(ephemeral=True)
    result = await DiscordActions.get_scheduled_events(interaction.guild)
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="react", description="Add a reaction to a message")
async def react_command(interaction: discord.Interaction, message_id: str, emoji: str) -> None:
    """Add reaction to a message."""
    await interaction.response.defer(ephemeral=True)
    try:
        msg = await interaction.channel.fetch_message(int(message_id))
        result = await DiscordActions.add_reaction(msg, emoji)
    except (ValueError, discord.NotFound):
        result = "Error: Message not found."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="messageinfo", description="Get detailed info about a message")
async def messageinfo_command(interaction: discord.Interaction, message_id: str) -> None:
    """Get message details."""
    await interaction.response.defer(ephemeral=True)
    try:
        msg = await interaction.channel.fetch_message(int(message_id))
        result = await DiscordActions.get_message_info(msg)
    except (ValueError, discord.NotFound):
        result = "Error: Message not found."
    if len(result) > 2000:
        result = result[:1997] + "..."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="createthread", description="Create a thread from a message")
async def createthread_command(interaction: discord.Interaction, message_id: str, name: str) -> None:
    """Create a thread from a message."""
    await interaction.response.defer(ephemeral=True)
    try:
        msg = await interaction.channel.fetch_message(int(message_id))
        result = await DiscordActions.create_thread(msg, name)
    except (ValueError, discord.NotFound):
        result = "Error: Message not found."
    await interaction.followup.send(result, ephemeral=True)


@discord_bot.tree.command(name="help_discord", description="Show available Discord action commands")
async def help_discord_command(interaction: discord.Interaction) -> None:
    """Show help for Discord actions."""
    help_text = """**Advanced Discord Commands:**

**Channel Operations:**
• `/channels [type]` - List all channels (type: all/text/voice/forum/thread)
• `/read <channel> [limit]` - Read messages from a channel
• `/channelinfo <channel>` - Get channel details
• `/mentions <channel>` - Find channel mentions in a channel
• `/browse <channel>` - Browse all mentioned channels

**Server Info:**
• `/serverinfo` - Get server information
• `/roles` - List all roles
• `/members [role] [limit]` - List members
• `/emojis` - List custom emojis
• `/events` - View scheduled events
• `/voice` - See voice channel status

**Message Operations:**
• `/search <query> [channel]` - Search messages
• `/pinned [channel]` - Get pinned messages
• `/messageinfo <id>` - Get message details
• `/react <id> <emoji>` - Add reaction

**Thread Operations:**
• `/threads [channel] [archived]` - List threads
• `/createthread <message_id> <name>` - Create thread

**User Info:**
• `/userinfo <user>` - Get user details

**LLM Actions:**
When chatting, I can also perform these actions automatically when asked!
Just tell me what you want to do in natural language.
"""
    await interaction.response.send_message(help_text, ephemeral=True)


# =============================================================================
# END ADVANCED DISCORD SLASH COMMANDS
# =============================================================================


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()
    
    # Signal that bot is ready
    bot_ready_event.set()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    # If local_only_mode is enabled, ignore all Discord messages (CLI only)
    if config.get("local_only_mode", False):
        logging.debug(f"Ignoring Discord message from {new_msg.author.id} (local_only_mode enabled)")
        return

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    model_without_suffix = provider_slash_model.removesuffix(":vision")
    if "/" not in model_without_suffix:
        logging.error(f"Invalid model format: '{provider_slash_model}'. Expected format: 'provider/model'")
        return
    provider, model = model_without_suffix.split("/", 1)

    if provider not in config.get("providers", {}):
        logging.error(f"Provider '{provider}' not found in config. Available providers: {list(config.get('providers', {}).keys())}")
        return
    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference is None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference is None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    system_prompt = config.get("system_prompt", "")
    now = datetime.now().astimezone()

    if system_prompt:
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

    if accept_usernames:
        system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

    # Add Discord actions capabilities if enabled and in a server context
    if not is_dm and config.get("enable_discord_actions", True):
        system_prompt += DISCORD_ACTIONS_PROMPT

    if system_prompt.strip():
        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason is not None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason is not None or msg_split_incoming
                    is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(max(0, EDIT_DELAY_SECONDS - time_delta))
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Process any Discord actions in the response
    full_response = "".join(response_contents)
    if not is_dm and config.get("enable_discord_actions", True):
        action_matches = ACTION_PATTERN.findall(full_response)
        if action_matches:
            action_results = []
            for action, params in action_matches:
                try:
                    result = await execute_discord_action(action, params, new_msg)
                    action_results.append(f"**{action}:**\n{result}")
                    logging.info(f"Executed Discord action: {action} with params: {params}")
                except Exception as e:
                    action_results.append(f"**{action}:** Error - {str(e)}")
                    logging.exception(f"Error executing Discord action: {action}")

            if action_results:
                # Send action results as a follow-up message
                action_output = "\n\n".join(action_results)
                if len(action_output) > 4000:
                    action_output = action_output[:3997] + "..."

                if use_plain_responses:
                    await response_msgs[-1].reply(view=LayoutView().add_item(TextDisplay(content=action_output)), silent=True)
                else:
                    action_embed = discord.Embed(description=action_output, color=EMBED_COLOR_COMPLETE)
                    action_embed.set_footer(text="Discord Action Results")
                    await response_msgs[-1].reply(embed=action_embed, silent=True)

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def cli_prompt(user_input: str) -> str:
    """Send a prompt to the LLM from CLI and return the response."""
    global curr_model, cli_conversation

    config = get_config()

    provider_slash_model = curr_model
    model_without_suffix = provider_slash_model.removesuffix(":vision")
    if "/" not in model_without_suffix:
        return f"Error: Invalid model format: '{provider_slash_model}'. Expected format: 'provider/model'"
    provider, model = model_without_suffix.split("/", 1)

    if provider not in config.get("providers", {}):
        return f"Error: Provider '{provider}' not found. Available: {list(config.get('providers', {}).keys())}"

    provider_config = config["providers"][provider]
    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    max_messages = config.get("max_messages", 25)

    # Add user message to conversation
    cli_conversation.append(dict(role="user", content=user_input))

    # Build messages list with system prompt
    messages = []
    system_prompt = config.get("system_prompt", "")
    now = datetime.now().astimezone()

    if system_prompt:
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

    # Add Discord actions prompt if enabled and server is selected
    discord_actions_enabled = config.get("enable_discord_actions", True) and cli_guild_id is not None
    if discord_actions_enabled:
        guild = discord_bot.get_guild(cli_guild_id)
        if guild:
            server_context = f"\n\nYou are connected to Discord server: {guild.name} (ID: {guild.id})"
            if cli_channel_id:
                channel = guild.get_channel(cli_channel_id)
                if channel:
                    server_context += f"\nCurrent channel: #{channel.name}"
            system_prompt += server_context + DISCORD_ACTIONS_PROMPT

    if system_prompt.strip():
        messages.append(dict(role="system", content=system_prompt))

    # Add conversation history (limited to max_messages)
    messages.extend(cli_conversation[-max_messages:])

    openai_kwargs = dict(
        model=model,
        messages=messages,
        stream=True,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body
    )

    response_text = ""
    try:
        async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
            if not (choice := chunk.choices[0] if chunk.choices else None):
                continue
            if choice.finish_reason is not None:
                break
            if content := choice.delta.content:
                response_text += content
                print(content, end="", flush=True)
        print()  # Newline after response
    except Exception as e:
        return f"Error: {e}"

    # Add assistant response to conversation
    if response_text:
        cli_conversation.append(dict(role="assistant", content=response_text))

    # Execute any Discord actions in the response
    if discord_actions_enabled and response_text:
        action_matches = ACTION_PATTERN.findall(response_text)
        if action_matches:
            print("\n--- Discord Action Results ---")
            for action, params in action_matches:
                try:
                    result = await execute_cli_discord_action(action, params)
                    print(f"\n[{action}]:\n{result}")
                    logging.info(f"CLI executed Discord action: {action} with params: {params}")
                except Exception as e:
                    print(f"\n[{action}]: Error - {str(e)}")
                    logging.exception(f"Error executing CLI Discord action: {action}")
            print("------------------------------\n")

    return response_text


async def cli_loop() -> None:
    """Interactive CLI loop for prompting the bot directly."""
    global curr_model, cli_conversation, config, cli_guild_id, cli_channel_id

    # Wait for Discord bot to be ready before starting CLI
    await bot_ready_event.wait()

    # Small delay to let any remaining Discord logs flush
    await asyncio.sleep(0.5)

    # Auto-select first server if only one is available
    if len(discord_bot.guilds) == 1:
        cli_guild_id = discord_bot.guilds[0].id
        print(f"\nAuto-selected server: {discord_bot.guilds[0].name}")

    print("\n" + "=" * 60)
    print("CLI Mode - Prompt the bot directly")
    print("=" * 60)
    print(f"Current model: {curr_model}")
    if cli_guild_id:
        guild = discord_bot.get_guild(cli_guild_id)
        print(f"Connected to server: {guild.name if guild else 'Unknown'}")
    print("\nCommands:")
    print("  /model [name]   - View or switch model")
    print("  /servers        - List available Discord servers")
    print("  /server [id]    - Select a Discord server for actions")
    print("  /channels       - List channels in current server")
    print("  /channel [name] - Select a channel for actions")
    print("  /discord [cmd]  - Execute Discord action directly")
    print("  /clear          - Clear conversation history")
    print("  /history        - Show conversation history")
    print("  /quit           - Exit CLI mode")
    print("=" * 60 + "\n")

    while True:
        try:
            # Show context in prompt
            prompt_prefix = "You"
            if cli_guild_id:
                guild = discord_bot.get_guild(cli_guild_id)
                if guild:
                    prompt_prefix = f"[{guild.name[:15]}]"
                    if cli_channel_id:
                        channel = guild.get_channel(cli_channel_id)
                        if channel:
                            prompt_prefix += f" #{channel.name[:15]}"

            user_input = await asyncio.to_thread(input, f"{prompt_prefix}: ")
            user_input = user_input.strip()

            if not user_input:
                continue

            # Handle CLI commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd == "quit" or cmd == "exit":
                    print("Exiting CLI mode...")
                    break

                elif cmd == "clear":
                    cli_conversation.clear()
                    print("Conversation history cleared.")
                    continue

                elif cmd == "history":
                    if not cli_conversation:
                        print("No conversation history.")
                    else:
                        print("\n--- Conversation History ---")
                        for msg in cli_conversation:
                            role = "You" if msg["role"] == "user" else "Bot"
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"{role}: {content}")
                        print("----------------------------\n")
                    continue

                elif cmd == "model":
                    config = get_config()
                    if arg:
                        if arg in config["models"]:
                            curr_model = arg
                            print(f"Model switched to: {curr_model}")
                        else:
                            print(f"Model '{arg}' not found. Available models:")
                            for m in config["models"]:
                                marker = "◉" if m == curr_model else "○"
                                print(f"  {marker} {m}")
                    else:
                        print(f"Current model: {curr_model}")
                        print("Available models:")
                        for m in config["models"]:
                            marker = "◉" if m == curr_model else "○"
                            print(f"  {marker} {m}")
                    continue

                elif cmd == "servers":
                    if not discord_bot.guilds:
                        print("Bot is not in any servers.")
                    else:
                        print("\nAvailable Discord Servers:")
                        for guild in discord_bot.guilds:
                            marker = "◉" if guild.id == cli_guild_id else "○"
                            print(f"  {marker} {guild.name} (ID: {guild.id})")
                        print("\nUse /server <id> to select a server")
                    continue

                elif cmd == "server":
                    if not arg:
                        if cli_guild_id:
                            guild = discord_bot.get_guild(cli_guild_id)
                            print(f"Current server: {guild.name if guild else 'Unknown'} (ID: {cli_guild_id})")
                        else:
                            print("No server selected. Use /servers to list available servers.")
                    else:
                        try:
                            guild_id = int(arg)
                            guild = discord_bot.get_guild(guild_id)
                            if guild:
                                cli_guild_id = guild_id
                                cli_channel_id = None  # Reset channel when switching servers
                                print(f"Switched to server: {guild.name}")
                                print("Discord actions are now enabled for this server!")
                            else:
                                print(f"Server with ID {guild_id} not found.")
                        except ValueError:
                            # Try to find by name
                            found = None
                            for guild in discord_bot.guilds:
                                if arg.lower() in guild.name.lower():
                                    found = guild
                                    break
                            if found:
                                cli_guild_id = found.id
                                cli_channel_id = None
                                print(f"Switched to server: {found.name}")
                                print("Discord actions are now enabled for this server!")
                            else:
                                print(f"Server '{arg}' not found. Use /servers to list available servers.")
                    continue

                elif cmd == "channels":
                    if not cli_guild_id:
                        print("No server selected. Use /server first.")
                    else:
                        guild = discord_bot.get_guild(cli_guild_id)
                        if guild:
                            result = await DiscordActions.list_channels(guild, "text")
                            print(f"\n{result}\n")
                        else:
                            print("Server not found.")
                    continue

                elif cmd == "channel":
                    if not cli_guild_id:
                        print("No server selected. Use /server first.")
                    elif not arg:
                        if cli_channel_id:
                            guild = discord_bot.get_guild(cli_guild_id)
                            if guild:
                                channel = guild.get_channel(cli_channel_id)
                                print(f"Current channel: #{channel.name if channel else 'Unknown'}")
                        else:
                            print("No channel selected. Use /channels to list available channels.")
                    else:
                        guild = discord_bot.get_guild(cli_guild_id)
                        if guild:
                            channel = await DiscordActions.get_channel_by_name_or_id(guild, arg)
                            if channel:
                                cli_channel_id = channel.id
                                print(f"Switched to channel: #{channel.name}")
                            else:
                                print(f"Channel '{arg}' not found.")
                        else:
                            print("Server not found.")
                    continue

                elif cmd == "discord":
                    # Direct Discord action execution
                    if not cli_guild_id:
                        print("No server selected. Use /server first.")
                    elif not arg:
                        print("Usage: /discord <ACTION:params>")
                        print("Example: /discord LIST_CHANNELS")
                        print("Example: /discord READ_CHANNEL:general:10")
                    else:
                        # Parse the action
                        if ':' in arg:
                            action_name, params = arg.split(':', 1)
                        else:
                            action_name, params = arg, ""
                        try:
                            result = await execute_cli_discord_action(action_name.upper(), params)
                            print(f"\n{result}\n")
                        except Exception as e:
                            print(f"Error: {str(e)}")
                    continue

                else:
                    print(f"Unknown command: /{cmd}")
                    print("Use /quit to exit, /servers to list servers, /help for more commands")
                    continue

            # Send prompt to LLM
            print("Bot: ", end="", flush=True)
            await cli_prompt(user_input)

        except EOFError:
            print("\nExiting CLI mode...")
            break
        except KeyboardInterrupt:
            print("\nExiting CLI mode...")
            break


async def main() -> None:
    """Main entry point - runs Discord bot and optionally CLI."""
    config = get_config()
    cli_enabled = config.get("cli_enabled", False)
    local_only = config.get("local_only_mode", False)

    if local_only and not cli_enabled:
        logging.warning("local_only_mode is enabled but cli_enabled is false. Enabling CLI automatically.")
        cli_enabled = True

    if local_only:
        logging.info("Running in local-only mode (Discord connected but ignores prompts)")

    if cli_enabled:
        # Run both Discord bot and CLI together
        logging.info("CLI mode enabled")
        await asyncio.gather(
            discord_bot.start(config["bot_token"]),
            cli_loop()
        )
    else:
        # Original behavior - just Discord bot
        await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
