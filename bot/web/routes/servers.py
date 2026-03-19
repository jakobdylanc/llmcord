"""Servers API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from bot.web.auth import CurrentUser, get_current_user
from bot.web.routes.status import bot_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["servers"])


# Response models
class TextChannel(BaseModel):
    """Text channel info."""
    id: str  # Use string to avoid JavaScript number precision loss
    name: str


class Server(BaseModel):
    """Server (guild) info."""
    id: str  # Use string to avoid JavaScript number precision loss
    name: str
    icon: Optional[str] = None
    member_count: int
    channel_count: int
    owner_id: Optional[str] = None  # Use string to avoid JavaScript number precision loss
    text_channels: list[TextChannel]
    
    # Serialize large integers as strings to avoid JS precision loss
    model_config = {"json_encoders": {int: lambda v: str(v)}}


class ServersResponse(BaseModel):
    """List of servers response."""
    servers: list[Server]


# Server detail response models
class GuildMember(BaseModel):
    """Guild member info."""
    id: str  # Use string to avoid JavaScript number precision loss
    username: str
    display_name: Optional[str] = None
    is_owner: bool = False


class GuildChannel(BaseModel):
    """Guild channel info."""
    id: str  # Use string to avoid JavaScript number precision loss
    name: str
    type: str


class GuildPermissions(BaseModel):
    """Bot permissions in a guild."""
    can_send_messages: bool = False
    can_embed_links: bool = False
    can_attach_files: bool = False
    can_use_external_emojis: bool = False
    can_manage_messages: bool = False
    can_manage_channels: bool = False
    can_kick_members: bool = False
    can_ban_members: bool = False
    can_manage_guild: bool = False


class ServerDetailResponse(BaseModel):
    """Server detail response."""
    id: str  # Use string to avoid JavaScript number precision loss
    name: str
    icon: Optional[str] = None
    owner_id: Optional[str] = None  # Use string to avoid JavaScript number precision loss
    member_count: int
    channel_count: int
    members: list[GuildMember]
    channels: list[GuildChannel]
    permissions: GuildPermissions


class PermissionUpdateRequest(BaseModel):
    """Request to update bot permissions."""
    action: str  # "grant" or "revoke"
    permission: str  # permission name


class PermissionUpdateResponse(BaseModel):
    """Response for permission update."""
    success: bool
    message: str


# API endpoints
@router.get("/servers", response_model=ServersResponse)
async def get_servers(current_user: CurrentUser = Depends(get_current_user)) -> ServersResponse:
    """Get list of connected servers and their text channels."""
    servers = []
    
    bot = bot_state._discord_bot
    if bot and hasattr(bot, 'guilds'):
        for guild in bot.guilds:
            # Get text channels - iterate guild.channels which returns all channels
            text_channels = []
            if hasattr(guild, 'channels'):
                for channel in guild.channels:
                    # Check for text channels - Discord.py uses ChannelType enum
                    # channel.type could be a string or ChannelType enum
                    channel_type = getattr(channel, 'type', None)
                    if channel_type is not None:
                        # Handle both enum and string comparison
                        type_name = str(channel_type).lower() if hasattr(channel_type, 'name') else str(channel_type).lower()
                        if 'text' in type_name or channel_type == 0:  # 0 is TextChannel in Discord API
                            text_channels.append(TextChannel(
                                id=str(channel.id),  # Convert to string to avoid JS precision loss
                                name=channel.name,
                            ))
            
            # Get member count - iterate guild.members (which is a lazy collection)
            member_count = 0
            if hasattr(guild, 'members'):
                # guild.members is a lazy iterator, iterate to count
                member_count = sum(1 for _ in guild.members)
            
            # Get channel count - iterate guild.channels
            channel_count = 0
            if hasattr(guild, 'channels'):
                channel_count = sum(1 for _ in guild.channels)
            
            # Get server icon URL
            icon_url = None
            if hasattr(guild, 'icon') and guild.icon:
                # guild.icon returns an IconAsset object with .url property
                icon_url = str(guild.icon.url) if hasattr(guild.icon, 'url') else None
            
            servers.append(Server(
                id=str(guild.id),  # Convert to string to avoid JS precision loss
                name=guild.name,
                icon=icon_url,
                member_count=member_count,
                channel_count=channel_count,
                owner_id=str(getattr(guild, 'owner_id', None)) if getattr(guild, 'owner_id', None) else None,
                text_channels=text_channels,
            ))
    
    return ServersResponse(servers=servers)


@router.get("/servers/{guild_id}", response_model=ServerDetailResponse)
async def get_server_detail(
    guild_id: str,  # Accept as string to avoid JS precision loss
    current_user: CurrentUser = Depends(get_current_user),
) -> ServerDetailResponse:
    """Get detailed information about a specific server (guild)."""
    bot = bot_state._discord_bot
    
    if not bot or not hasattr(bot, 'guilds'):
        raise HTTPException(status_code=404, detail="Bot not connected to any servers")
    
    # Find the guild - convert string back to int for comparison
    try:
        guild_id_int = int(guild_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid guild ID")
    
    guild = None
    for g in bot.guilds:
        if g.id == guild_id_int:
            guild = g
            break
    
    if not guild:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # Get members
    members = []
    owner_id = getattr(guild, 'owner_id', None)
    owner_id_str = str(owner_id) if owner_id else None
    if hasattr(guild, 'members'):
        for member in guild.members:
            # Get display name (nickname if set, otherwise username)
            display_name = getattr(member, 'nick', None) or getattr(member, 'display_name', None)
            if display_name == member.name:
                display_name = None  # Don't show if same as username
            
            members.append(GuildMember(
                id=str(member.id),  # Convert to string to avoid JS precision loss
                username=member.name,
                display_name=display_name,
                is_owner=str(member.id) == owner_id_str if owner_id_str else False,
            ))
    
    # Get channels
    channels = []
    if hasattr(guild, 'channels'):
        for channel in guild.channels:
            # Get channel type as string
            channel_type = getattr(channel, 'type', None)
            if channel_type is not None:
                if hasattr(channel_type, 'name'):
                    type_name = channel_type.name
                else:
                    type_name = str(channel_type)
            else:
                type_name = "unknown"
            
            channels.append(GuildChannel(
                id=str(channel.id),  # Convert to string to avoid JS precision loss
                name=channel.name,
                type=type_name,
            ))
    
    # Get bot permissions in the guild
    permissions = GuildPermissions()
    if hasattr(guild, 'me') and guild.me:
        perms = guild.me.guild_permissions
        if perms:
            permissions = GuildPermissions(
                can_send_messages=perms.send_messages,
                can_embed_links=perms.embed_links,
                can_attach_files=perms.attach_files,
                can_use_external_emojis=perms.external_emojis,
                can_manage_messages=perms.manage_messages,
                can_manage_channels=perms.manage_channels,
                can_kick_members=perms.kick_members,
                can_ban_members=perms.ban_members,
                can_manage_guild=perms.manage_guild,
            )
    
    # Get server icon
    icon_url = None
    if hasattr(guild, 'icon') and guild.icon:
        icon_url = str(guild.icon.url) if hasattr(guild.icon, 'url') else None
    
    # Count members and channels
    member_count = len(members)
    channel_count = len(channels)
    
    return ServerDetailResponse(
        id=str(guild.id),  # Convert to string to avoid JS precision loss
        name=guild.name,
        icon=icon_url,
        owner_id=owner_id_str,
        member_count=member_count,
        channel_count=channel_count,
        members=members,
        channels=channels,
        permissions=permissions,
    )


# Discord permission values for reference
DISCORD_PERMISSIONS = {
    "send_messages": "Send Messages",
    "embed_links": "Embed Links",
    "attach_files": "Attach Files",
    "external_emojis": "Use External Emoji",
    "manage_messages": "Manage Messages",
    "manage_channels": "Manage Channels",
    "kick_members": "Kick Members",
    "ban_members": "Ban Members",
    "manage_guild": "Manage Server",
}


@router.put("/servers/{guild_id}/permissions", response_model=PermissionUpdateResponse)
async def update_server_permissions(
    guild_id: str,  # Accept as string to avoid JS precision loss
    request: PermissionUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> PermissionUpdateResponse:
    """Update bot permissions in a guild (grant/revoke)."""
    bot = bot_state._discord_bot
    
    if not bot or not hasattr(bot, 'guilds'):
        raise HTTPException(status_code=404, detail="Bot not connected to any servers")
    
    # Find the guild - convert string back to int for comparison
    try:
        guild_id_int = int(guild_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid guild ID")
    
    guild = None
    for g in bot.guilds:
        if g.id == guild_id_int:
            guild = g
            break
    
    if not guild:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # Validate action
    if request.action not in ["grant", "revoke"]:
        return PermissionUpdateResponse(
            success=False,
            message=f"Invalid action: {request.action}. Must be 'grant' or 'revoke'."
        )
    
    # Validate permission
    permission_map = {
        "send_messages": "send_messages",
        "embed_links": "embed_links",
        "attach_files": "attach_files",
        "external_emojis": "external_emojis",
        "manage_messages": "manage_messages",
        "manage_channels": "manage_channels",
        "kick_members": "kick_members",
        "ban_members": "ban_members",
        "manage_guild": "manage_guild",
    }
    
    perm_key = permission_map.get(request.permission)
    if not perm_key:
        valid_perms = ", ".join(permission_map.keys())
        return PermissionUpdateResponse(
            success=False,
            message=f"Invalid permission: {request.permission}. Valid: {valid_perms}"
        )
    
    try:
        # Get the bot's current role in the guild
        bot_member = guild.me if hasattr(guild, 'me') else None
        if not bot_member:
            return PermissionUpdateResponse(
                success=False,
                message="Cannot find bot in this server"
            )
        
        # Note: Actually modifying Discord permissions requires adminitrator
        # For now, we'll just return a success message indicating the action
        # Real permission management would require Discord's permission edit API
        action_verb = "granted" if request.action == "grant" else "revoked"
        
        logger.info(f"Permission {request.permission} would be {action_verb} in {guild.name} (guild_id: {guild_id})")
        
        return PermissionUpdateResponse(
            success=True,
            message=f"Permission '{request.permission}' {action_verb} (simulated - Discord API permission edit not implemented)"
        )
    except Exception as e:
        logger.error(f"Error updating permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))