import logging
from base64 import b64encode
from typing import List, Dict, Any, Optional, Tuple

import discord
import httpx

from config import Config
from models import MsgNode, ConversationWarnings


async def extract_message_content(msg: discord.Message, msg_node: MsgNode, httpx_client: httpx.AsyncClient, config: Config) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Extract text and image content from a Discord message.
    Returns a tuple of (text_content, image_content, has_bad_attachments)
    """
    # Clean up the message content (remove bot mention)
    cleaned_content = msg.content
    if msg.guild and msg.guild.me in msg.mentions:
        cleaned_content = cleaned_content.replace(f"<@{msg.guild.me.id}>", "").lstrip()
    elif msg.mentions:
        # In DMs or where guild.me isn't available, check each mention
        for mentioned_user in msg.mentions:
            if mentioned_user.bot:
                cleaned_content = cleaned_content.replace(f"<@{mentioned_user.id}>", "").lstrip()
    
    # Process attachments by type
    good_attachments = {
        file_type: [att for att in msg.attachments if att.content_type and file_type in att.content_type]
        for file_type in config.ALLOWED_FILE_TYPES
    }
    
    # Extract text from content, embeds, and text attachments
    text_parts = []
    if cleaned_content:
        text_parts.append(cleaned_content)
    
    # Add embed descriptions
    for embed in msg.embeds:
        if embed.description:
            text_parts.append(embed.description)
    
    # Add text from text attachments
    for att in good_attachments.get("text", []):
        try:
            response = await httpx_client.get(att.url)
            if response.status_code == 200:
                text_parts.append(response.text)
        except Exception as e:
            logging.warning(f"Failed to fetch text attachment: {e}")
    
    # Create images array for vision models
    images = []
    for att in good_attachments.get("image", []):
        try:
            response = await httpx_client.get(att.url)
            if response.status_code == 200:
                image_data = f"data:{att.content_type};base64,{b64encode(response.content).decode('utf-8')}"
                images.append({"type": "image_url", "image_url": {"url": image_data}})
        except Exception as e:
            logging.warning(f"Failed to fetch image attachment: {e}")
    
    # Check if there are any unsupported attachments
    has_bad_attachments = len(msg.attachments) > sum(len(att_list) for att_list in good_attachments.values())
    
    return "\n".join(text_parts), images, has_bad_attachments


async def find_parent_message(msg: discord.Message) -> Optional[discord.Message]:
    """
    Find the parent message of the given message.
    Handles replies, threads, and back-to-back messages.
    """
    try:
        # Case 1: Direct reply
        if msg.reference and msg.reference.message_id:
            # Try to get from cache first
            if msg.reference.cached_message:
                return msg.reference.cached_message
            # Otherwise fetch from API
            return await msg.channel.fetch_message(msg.reference.message_id)
        
        # Case 2: Thread starter message
        if msg.channel.type == discord.ChannelType.public_thread and not msg.reference:
            if msg.channel.starter_message:
                return msg.channel.starter_message
            return await msg.channel.parent.fetch_message(msg.channel.id)
        
        # Case 3: Back-to-back messages in same channel (for DMs or from same author)
        # Only if not already mentioning the bot (which would start a new conversation)
        if not msg.reference:
            # Access the client through the _state attribute, which is more reliable
            bot_user = msg.guild.me if msg.guild else None
            bot_mentioned = False
            
            if bot_user and f"<@{bot_user.id}>" in msg.content:
                bot_mentioned = True
            
            if not bot_mentioned:
                # Get the previous message in the channel
                async for prev_msg in msg.channel.history(before=msg, limit=1):
                    # Check if it's a normal message and from the same author (or bot in DMs)
                    if prev_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                        is_dm = msg.channel.type == discord.ChannelType.private
                        if is_dm:
                            # In DMs, check if previous message is from the bot
                            if prev_msg.author.bot and not msg.author.bot:
                                return prev_msg
                        else:
                            # In regular channels, check if previous message is from the same author
                            if prev_msg.author == msg.author:
                                return prev_msg
                    
        return None
        
    except (discord.NotFound, discord.HTTPException) as e:
        logging.warning(f"Failed to fetch parent message: {e}")
        return None


def check_permissions(msg: discord.Message, config: Config) -> bool:
    """
    Check if the user has permission to use the bot in this context.
    Returns True if permitted, False otherwise.
    """
    is_dm = msg.channel.type == discord.ChannelType.private
    
    # Get all relevant IDs
    user_id = msg.author.id
    role_ids = set(role.id for role in getattr(msg.author, "roles", ()))
    channel_ids = set(id for id in (
        msg.channel.id, 
        getattr(msg.channel, "parent_id", None), 
        getattr(msg.channel, "category_id", None)
    ) if id)
    
    permissions = config.permissions
    
    # Extract permission lists
    allowed_user_ids = permissions.get("users", {}).get("allowed_ids", [])
    blocked_user_ids = permissions.get("users", {}).get("blocked_ids", [])
    
    allowed_role_ids = permissions.get("roles", {}).get("allowed_ids", [])
    blocked_role_ids = permissions.get("roles", {}).get("blocked_ids", [])
    
    allowed_channel_ids = permissions.get("channels", {}).get("allowed_ids", [])
    blocked_channel_ids = permissions.get("channels", {}).get("blocked_ids", [])
    
    # Check if DMs are allowed
    if is_dm and not config.allow_dms:
        return False
    
    # Determine if all users are allowed (no specific allowed list)
    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    
    # Check if user is explicitly allowed or blocked
    is_good_user = allow_all_users or user_id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or user_id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)
    
    # Check if channel is allowed or blocked
    allow_all_channels = not allowed_channel_ids
    is_good_channel = config.allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)
    
    # User must be good and channel must be good
    return (not is_bad_user) and (not is_bad_channel)


def create_embed_for_warnings(warnings: ConversationWarnings) -> discord.Embed:
    """Create a Discord embed with warnings."""
    embed = discord.Embed()
    for warning in warnings.get_sorted():
        embed.add_field(name=warning, value="", inline=False)
    return embed


def truncate_messages(messages: List[Dict[str, Any]], max_length: int) -> List[str]:
    """
    Split a message into chunks that don't exceed max_length.
    Returns a list of message content strings.
    """
    if not messages:
        return []
        
    chunks = []
    current_chunk = ""
    
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            if len(current_chunk) + len(content) > max_length:
                # Need to start a new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle content that's longer than max_length
                while len(content) > max_length:
                    chunks.append(content[:max_length])
                    content = content[max_length:]
                
                current_chunk = content
            else:
                current_chunk += content
        
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks