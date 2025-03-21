import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import discord
import httpx

from config import Config
from llm_client import LLMClient
from message_store import MessageStore
from database import Database  # Import the new database module
from models import MsgNode, ConversationWarnings
from utils import (
    extract_message_content, 
    find_parent_message,
    check_permissions, 
    create_embed_for_warnings
)


class LLMCordClient(discord.Client):
    """
    Discord client for the LLMCord bot.
    Handles Discord events and message processing.
    """
    def __init__(self, config: Config):
        self.config = config
        
        # Setup Discord client with proper intents
        intents = discord.Intents.default()
        intents.message_content = True
        activity = discord.CustomActivity(name=(self.config.status_message or "github.com/jakobdylanc/llmcord")[:128])
        
        super().__init__(intents=intents, activity=activity)
        
        # Initialize other components
        self.http_client = httpx.AsyncClient()
        self.llm_client = LLMClient(config)
        self.message_store = MessageStore(config)
        self.db = Database()  # Initialize the database
        self.last_task_time = 0
        
    async def on_ready(self):
        """Handle bot ready event."""
        logging.info(f"Logged in as {self.user} (ID: {self.user.id})")
        
        if self.config.client_id:
            logging.info(
                f"\n\nBOT INVITE URL:\n"
                f"https://discord.com/api/oauth2/authorize?client_id={self.config.client_id}"
                f"&permissions=412317273088&scope=bot\n"
            )
            
            
    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        # Skip bot messages
        if message.author.bot:
            return
            
        # Check if the bot was mentioned or if it's a DM
        is_dm = message.channel.type == discord.ChannelType.private
        is_mentioned = self.user in message.mentions
        
        if not (is_dm or is_mentioned):
            return
            
        # Update config in case it was changed
        self.config.reload()
        
        # Check permissions
        if not check_permissions(message, self.config):
            return
        
        # Check for reset command
        if "reset" in message.content and self.user.mentioned_in(message):
            success = self.db.reset_user_history(message.author.id)
            if success:
                await message.reply("Your conversation history has been reset. Starting fresh!")
            else:
                await message.reply("There was an error resetting your conversation history.")
            return
            
        # Process the message chain and generate a response
        await self.process_message_chain(message)
        
    async def process_message_node(self, msg: discord.Message, node: MsgNode):
        """Process a message and update its node with content and metadata."""
        if node.text is None:
            # Extract content and update node
            text, images, has_bad_attachments = await extract_message_content(
                msg, node, self.http_client, self.config
            )
            
            node.text = text
            node.images = images
            node.has_bad_attachments = has_bad_attachments
            node.role = "assistant" if msg.author == self.user else "user"
            node.user_id = msg.author.id if node.role == "user" else None
            
            # Try to find parent message if not already set
            if node.parent_msg is None:
                try:
                    node.parent_msg = await find_parent_message(msg)
                    if node.parent_msg is None and msg.reference:
                        node.fetch_parent_failed = True
                except Exception as e:
                    logging.warning(f"Error finding parent message: {str(e)}")
                    node.fetch_parent_failed = True
                    
    async def build_message_chain(self, start_msg: discord.Message) -> Tuple[List[Dict[str, Any]], ConversationWarnings]:
        """
        Build a chain of messages starting from the given message.
        Returns the messages formatted for the LLM API and any warnings.
        """
        messages = []
        warnings = ConversationWarnings()
        curr_msg = start_msg
        max_messages = self.config.max_messages
        provider, model = self.config.model.split("/", 1)
        
        # Determine model capabilities
        accept_images = self.llm_client.model_supports_images(model)
        accept_usernames = self.llm_client.provider_supports_usernames(provider)
        max_images = self.config.max_images if accept_images else 0
        
        # Check if we should include message history
        include_history = True
        if self.user.mentioned_in(start_msg) and not start_msg.reference:
            # If the bot is directly mentioned without a reply, we start a new conversation
            include_history = False
        
        # First, process the immediate message chain
        immediate_messages = []
        while curr_msg is not None and len(immediate_messages) < max_messages:
            # Get or create message node
            curr_node = self.message_store.get(curr_msg.id)
            
            async with curr_node.lock:
                # Process the message if not already processed
                await self.process_message_node(curr_msg, curr_node)
                
                # Prepare content for the LLM
                content = None
                if curr_node.images[:max_images]:
                    # For vision models with images
                    text_content = curr_node.text[:self.config.max_text] if curr_node.text else ""
                    content = []
                    if text_content:
                        content.append({"type": "text", "text": text_content})
                    content.extend(curr_node.images[:max_images])
                else:
                    # Text only
                    content = curr_node.text[:self.config.max_text] if curr_node.text else ""
                
                if content and (isinstance(content, str) and content != "" or isinstance(content, list) and len(content) > 0):
                    message = {"content": content, "role": curr_node.role}
                    if accept_usernames and curr_node.user_id is not None:
                        message["name"] = str(curr_node.user_id)
                    immediate_messages.append(message)
                
                # Check for warnings
                if curr_node.text and len(curr_node.text) > self.config.max_text:
                    warnings.add(f"⚠️ Max {self.config.max_text:,} characters per message")
                if len(curr_node.images) > max_images:
                    if max_images > 0:
                        warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message")
                    else:
                        warnings.add("⚠️ Can't see images")
                if curr_node.has_bad_attachments:
                    warnings.add("⚠️ Unsupported attachments")
                if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(immediate_messages) == max_messages):
                    warnings.add(f"⚠️ Only using last {len(immediate_messages)} message{'' if len(immediate_messages) == 1 else 's'}")
            
            # Move to parent message
            curr_msg = curr_node.parent_msg
        
        # Add immediate messages to the final messages list
        messages.extend(immediate_messages)
        
        # Include conversation history if needed and we have room for more messages
        if include_history and len(messages) < max_messages:
            user_id = start_msg.author.id
            guild_id = start_msg.guild.id if start_msg.guild else None
            
            # Get active conversation or create a new one if none exists
            conversation_id = self.db.get_active_conversation(user_id)
            if not conversation_id:
                conversation_id = self.db.create_conversation(user_id, guild_id, start_msg.channel.id)
            
            # Get previous messages from this conversation
            history_limit = max_messages - len(messages)
            history_messages = self.db.get_conversation_messages(conversation_id, limit=history_limit)
            
            # Format the conversation as a clear chronological sequence
            if history_messages:
                # First, store the current messages in the database
                for msg in reversed(immediate_messages):
                    role = msg["role"]
                    content = msg["content"]
                    has_images = isinstance(content, list) and any(item.get("type") == "image_url" for item in content)
                    if has_images:
                        # For messages with images, extract just the text part
                        text_content = next((item["text"] for item in content if item.get("type") == "text"), "")
                        self.db.add_message(conversation_id, role, text_content, 
                                         discord_message_id=start_msg.id, has_images=True)
                    else:
                        self.db.add_message(conversation_id, role, content, discord_message_id=start_msg.id)
                
                # Format the conversation history in a clear way
                conversation_summary = []
                current_role = None
                current_messages = []
                
                # Group consecutive messages by the same role
                for msg in history_messages:
                    if msg["role"] != current_role:
                        if current_role and current_messages:
                            speaker = "User" if current_role == "user" else "You"
                            combined_content = "\n".join(current_messages)
                            conversation_summary.append(f"{speaker}: {combined_content}")
                        current_role = msg["role"]
                        current_messages = [msg["content"]]
                    else:
                        current_messages.append(msg["content"])
                
                # Add the last group
                if current_role and current_messages:
                    speaker = "User" if current_role == "user" else "You"
                    combined_content = "\n".join(current_messages)
                    conversation_summary.append(f"{speaker}: {combined_content}")
                
                # Add a system message to introduce the conversation history
                conversation_intro = {
                    "role": "system",
                    "content": "Previous conversation history with this user:\n\n" + "\n\n".join(conversation_summary)
                }
                
                # Create the formatted messages array
                formatted_messages = [conversation_intro]
                
                # Add a transition message between history and current exchange
                if history_messages:
                    transition_message = {
                        "role": "system",
                        "content": "The conversation is now continuing. Please respond appropriately based on this history and the user's current message."
                    }
                    formatted_messages.append(transition_message)
                
                # Add the current messages
                formatted_messages.extend(messages)
                messages = formatted_messages
                
                # Add a warning about using conversation history
                warnings.add(f"⚠️ Including {len(history_messages)} message(s) from previous conversation")
            else:
                # No history, just store the current messages
                for msg in reversed(immediate_messages):
                    role = msg["role"]
                    content = msg["content"]
                    has_images = isinstance(content, list) and any(item.get("type") == "image_url" for item in content)
                    if has_images:
                        text_content = next((item["text"] for item in content if item.get("type") == "text"), "")
                        self.db.add_message(conversation_id, role, text_content, 
                                         discord_message_id=start_msg.id, has_images=True)
                    else:
                        self.db.add_message(conversation_id, role, content, discord_message_id=start_msg.id)
        else:
            # Start a new conversation
            user_id = start_msg.author.id
            guild_id = start_msg.guild.id if start_msg.guild else None
            conversation_id = self.db.create_conversation(user_id, guild_id, start_msg.channel.id)
            
            # Store the current messages
            for msg in reversed(immediate_messages):
                role = msg["role"]
                content = msg["content"]
                has_images = isinstance(content, list) and any(item.get("type") == "image_url" for item in content)
                if has_images:
                    text_content = next((item["text"] for item in content if item.get("type") == "text"), "")
                    self.db.add_message(conversation_id, role, text_content, 
                                     discord_message_id=start_msg.id, has_images=True)
                else:
                    self.db.add_message(conversation_id, role, content, discord_message_id=start_msg.id)
        
        # Log the complete message chain for debugging
        logging.info(f"Complete message chain being sent to LLM:")
        for idx, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # For content that's a list (like with images), extract just the text
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content = " | ".join(text_parts) + " [+ images]"
            
            # Truncate very long messages in the log
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
                
            logging.info(f"  [{idx}] {role}: {content}")
            
        return messages, warnings
    
    async def process_message_chain(self, message: discord.Message):
        """Process a message chain and generate a response."""
        # Log the incoming message
        logging.info(
            f"Message received (user ID: {message.author.id}, "
            f"attachments: {len(message.attachments)}):\n{message.content}"
        )
        
        try:
            # Build the message chain
            messages, warnings = await self.build_message_chain(message)
            
            # If message chain is empty, don't bother responding
            if not messages:
                return
                
            # Determine response format parameters
            use_plain_responses = self.config.use_plain_responses
            max_message_length = 2000 if use_plain_responses else (4096 - len(self.config.STREAMING_INDICATOR))
            
            # Generate and send the response
            await self.send_llm_response(message, messages, warnings, use_plain_responses, max_message_length)
            
            # Clean up old message nodes
            self.message_store.cleanup()
            
        except Exception as e:
            logging.exception(f"Error processing message chain: {e}")
            try:
                await message.reply(f"An error occurred: {str(e)}")
            except:
                pass
                
    async def send_llm_response(
        self, 
        original_msg: discord.Message, 
        messages: List[Dict[str, Any]],
        warnings: ConversationWarnings,
        use_plain_responses: bool,
        max_message_length: int
    ):
        """Generate an LLM response and send it to Discord."""
        curr_content = None
        finish_reason = None
        edit_task = None
        response_msgs = []
        response_contents = []
        
        # Create embed with warnings if any
        warnings_embed = create_embed_for_warnings(warnings)
        
        try:
            async with original_msg.channel.typing():
                async for content_delta, curr_finish_reason in self.llm_client.generate_response(messages):
                    if finish_reason is not None:
                        break
                        
                    finish_reason = curr_finish_reason
                    
                    prev_content = curr_content or ""
                    curr_content = content_delta
                    
                    new_content = prev_content if finish_reason is None else (prev_content + curr_content)
                    
                    if response_contents == [] and new_content == "":
                        continue
                        
                    if start_next_msg := (response_contents == [] or 
                        len(response_contents[-1] + new_content) > max_message_length):
                        response_contents.append("")
                        
                    response_contents[-1] += new_content
                    
                    if not use_plain_responses:
                        # Check if we need to edit the message
                        ready_to_edit = (edit_task is None or edit_task.done()) and \
                            datetime.now().timestamp() - self.last_task_time >= self.config.EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason is None and \
                            len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason is not None or msg_split_incoming
                        is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")
                        
                        if start_next_msg or ready_to_edit or is_final_edit:
                            if edit_task is not None:
                                await edit_task
                                
                            # Update embed with current content
                            embed = discord.Embed()
                            for field in warnings_embed.fields:
                                embed.add_field(name=field.name, value=field.value, inline=field.inline)
                                
                            embed.description = response_contents[-1] if is_final_edit else \
                                (response_contents[-1] + self.config.STREAMING_INDICATOR)
                            embed.color = self.config.EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else \
                                self.config.EMBED_COLOR_INCOMPLETE
                                
                            if start_next_msg:
                                # Send a new message
                                reply_to_msg = original_msg if not response_msgs else response_msgs[-1]
                                response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                                response_msgs.append(response_msg)
                                
                                # Create node for the new message
                                new_node = MsgNode(parent_msg=original_msg)
                                self.message_store.set(response_msg.id, new_node)
                                await new_node.lock.acquire()
                            else:
                                # Edit existing message
                                edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                
                            self.last_task_time = datetime.now().timestamp()
                
                # For plain responses, send all content at the end
                if use_plain_responses and finish_reason is not None:
                    for content in response_contents:
                        if content:
                            reply_to_msg = original_msg if not response_msgs else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                            response_msgs.append(response_msg)
                            
                            # Create node for the new message
                            new_node = MsgNode(parent_msg=original_msg)
                            self.message_store.set(response_msg.id, new_node)
                            await new_node.lock.acquire()
                            
        except Exception as e:
            logging.exception(f"Error generating response: {e}")
            try:
                await original_msg.reply("Sorry, I encountered an error while generating a response.")
            except:
                pass
                
        # Update message nodes with response content
        for response_msg in response_msgs:
            node = self.message_store.get(response_msg.id)
            node.text = "".join(response_contents)
            if node.lock.locked():
                node.lock.release()