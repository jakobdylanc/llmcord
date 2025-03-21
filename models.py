import asyncio
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any

import discord


@dataclass
class MsgNode:
    """
    Represents a message in the conversation chain.
    Stores all necessary information about a message for LLM processing.
    """
    text: Optional[str] = None
    images: List[Dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConversationWarnings:
    """
    Tracks warnings that should be displayed to the user about their conversation.
    """
    def __init__(self):
        self.warnings = set()
        
    def add(self, warning: str):
        self.warnings.add(warning)
        
    def get_sorted(self):
        return sorted(self.warnings)