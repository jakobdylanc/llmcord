import logging
from typing import Dict, Optional

import discord

from models import MsgNode
from config import Config


class MessageStore:
    """
    Manages the storage and retrieval of message nodes.
    Implements a cache with size limit to prevent memory leaks.
    """
    def __init__(self, config: Config):
        self.config = config
        self.nodes: Dict[int, MsgNode] = {}
        
    def get(self, msg_id: int) -> MsgNode:
        """Get a message node, creating it if it doesn't exist."""
        if msg_id not in self.nodes:
            self.nodes[msg_id] = MsgNode()
        return self.nodes[msg_id]
        
    def set(self, msg_id: int, node: MsgNode):
        """Set a message node."""
        self.nodes[msg_id] = node
        
    def cleanup(self):
        """Remove oldest message nodes if we exceed the maximum cache size."""
        if len(self.nodes) > self.config.MAX_MESSAGE_NODES:
            oldest_keys = sorted(self.nodes.keys())[:len(self.nodes) - self.config.MAX_MESSAGE_NODES]
            for key in oldest_keys:
                try:
                    # Only remove if we can acquire the lock (no active processing)
                    if self.nodes[key].lock.locked():
                        continue
                    del self.nodes[key]
                except KeyError:
                    # Node might have been removed by another process
                    pass
                
    async def build_conversation_chain(self, start_msg: discord.Message, max_messages: int):
        """
        Build a chain of messages starting from the given message.
        Returns the messages and any user warnings.
        """
        from models import ConversationWarnings
        
        messages = []
        warnings = ConversationWarnings()
        curr_msg = start_msg
        
        while curr_msg is not None and len(messages) < max_messages:
            curr_node = self.get(curr_msg.id)
            
            async with curr_node.lock:
                # Process node data and add to messages list
                # This will be implemented in the discord_client.py
                # where we have access to the httpx client
                
                # Move to parent message
                curr_msg = curr_node.parent_msg
                
        return messages, warnings