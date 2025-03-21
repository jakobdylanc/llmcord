import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

class Database:
    """
    Handles storage and retrieval of message history for the bot.
    Uses SQLite for persistence.
    """
    def __init__(self, db_path: str = "message_history.db"):
        self.db_path = db_path
        self._create_tables()
        
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER,
                channel_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
            ''')
            
            # Create messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                discord_message_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                has_images BOOLEAN DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
            ''')
            
            conn.commit()
        except Exception as e:
            logging.error(f"Error creating database tables: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def create_conversation(self, user_id: int, guild_id: Optional[int], channel_id: int) -> int:
        """
        Create a new conversation for a user.
        Returns the conversation ID.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # First, mark any active conversations for this user as inactive
            cursor.execute('''
            UPDATE conversations 
            SET is_active = 0, updated_at = ?
            WHERE user_id = ? AND is_active = 1
            ''', (datetime.now(), user_id))
            
            # Create a new conversation
            cursor.execute('''
            INSERT INTO conversations (user_id, guild_id, channel_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (user_id, guild_id, channel_id, datetime.now(), datetime.now()))
            
            conversation_id = cursor.lastrowid
            conn.commit()
            return conversation_id
        except Exception as e:
            logging.error(f"Error creating conversation: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()
    
    def add_message(self, conversation_id: int, role: str, content: str, 
                   discord_message_id: Optional[int] = None, has_images: bool = False) -> bool:
        """
        Add a message to a conversation.
        Returns True if successful, False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Get the current timestamp
            current_time = datetime.now()
            
            # Add the message
            cursor.execute('''
            INSERT INTO messages (conversation_id, discord_message_id, role, content, timestamp, has_images)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, discord_message_id, role, content, current_time, has_images))
            
            # Update the conversation's updated_at timestamp
            cursor.execute('''
            UPDATE conversations 
            SET updated_at = ?
            WHERE conversation_id = ?
            ''', (current_time, conversation_id))
            
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error adding message: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_active_conversation(self, user_id: int) -> Optional[int]:
        """
        Get the active conversation ID for a user.
        Returns None if no active conversation exists.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT conversation_id FROM conversations
            WHERE user_id = ? AND is_active = 1
            ORDER BY updated_at DESC
            LIMIT 1
            ''', (user_id,))
            
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logging.error(f"Error getting active conversation: {e}")
            return None
        finally:
            conn.close()
    
    def get_conversation_messages(self, conversation_id: int, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Get the messages for a conversation.
        Returns a list of message dictionaries in chronological order (oldest first).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT role, content, has_images, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
            ''', (conversation_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                role, content, has_images, timestamp = row
                message = {
                    "role": role,
                    "content": content
                }
                messages.append(message)
                
            return messages
        except Exception as e:
            logging.error(f"Error getting conversation messages: {e}")
            return []
        finally:
            conn.close()
    
    def reset_user_history(self, user_id: int) -> bool:
        """
        Reset a user's conversation history by marking all conversations as inactive.
        Returns True if successful, False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE conversations 
            SET is_active = 0, updated_at = ?
            WHERE user_id = ?
            ''', (datetime.now(), user_id))
            
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error resetting user history: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
            
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get usage statistics for a user.
        Returns a dictionary with statistics.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Get total number of messages
            cursor.execute('''
            SELECT COUNT(*) FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.user_id = ?
            ''', (user_id,))
            total_messages = cursor.fetchone()[0]
            
            # Get total number of conversations
            cursor.execute('''
            SELECT COUNT(*) FROM conversations
            WHERE user_id = ?
            ''', (user_id,))
            total_conversations = cursor.fetchone()[0]
            
            # Get first conversation date
            cursor.execute('''
            SELECT MIN(created_at) FROM conversations
            WHERE user_id = ?
            ''', (user_id,))
            first_conversation = cursor.fetchone()[0]
            
            return {
                "total_messages": total_messages,
                "total_conversations": total_conversations,
                "first_conversation": first_conversation
            }
        except Exception as e:
            logging.error(f"Error getting user stats: {e}")
            return {
                "total_messages": 0,
                "total_conversations": 0,
                "first_conversation": None
            }
        finally:
            conn.close()