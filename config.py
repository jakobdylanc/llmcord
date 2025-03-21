import yaml
import logging
from typing import Dict, Any


class Config:
    """
    Handles loading and accessing configuration data.
    """
    def __init__(self, filename: str = "config.yaml"):
        self.filename = filename
        self.data = self.reload()
        
        # Constants
        self.VISION_MODEL_TAGS = (
            "gpt-4", "claude-3", "gemini", "gemma", "pixtral", 
            "mistral-small", "llava", "vision", "vl"
        )
        self.PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
        self.ALLOWED_FILE_TYPES = ("image", "text")
        self.EMBED_COLOR_COMPLETE = 0x006400  # dark_green
        self.EMBED_COLOR_INCOMPLETE = 0xFFA500  # orange
        self.STREAMING_INDICATOR = " ⚪"
        self.EDIT_DELAY_SECONDS = 1
        self.MAX_MESSAGE_NODES = 100

    def reload(self) -> Dict[str, Any]:
        """Reloads the configuration from file."""
        try:
            with open(self.filename, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return {}
            
    def get(self, key: str, default=None):
        """Get a configuration value with an optional default."""
        return self.data.get(key, default)
        
    def __getitem__(self, key: str):
        """Allow dict-like access to configuration."""
        return self.data[key]
        
    @property
    def bot_token(self) -> str:
        return self.data.get("bot_token", "")
        
    @property
    def client_id(self) -> str:
        return self.data.get("client_id", "")
        
    @property
    def status_message(self) -> str:
        return self.data.get("status_message", "github.com/jakobdylanc/llmcord")
        
    @property
    def max_text(self) -> int:
        return self.data.get("max_text", 100000)
        
    @property
    def max_images(self) -> int:
        return self.data.get("max_images", 5)
        
    @property
    def max_messages(self) -> int:
        return self.data.get("max_messages", 25)
        
    @property
    def use_plain_responses(self) -> bool:
        return self.data.get("use_plain_responses", False)
        
    @property
    def allow_dms(self) -> bool:
        return self.data.get("allow_dms", True)
        
    @property
    def permissions(self) -> Dict[str, Any]:
        return self.data.get("permissions", {
            "users": {"allowed_ids": [], "blocked_ids": []},
            "roles": {"allowed_ids": [], "blocked_ids": []},
            "channels": {"allowed_ids": [], "blocked_ids": []}
        })
        
    @property
    def system_prompt(self) -> str:
        return self.data.get("system_prompt", "")
        
    @property
    def providers(self) -> Dict[str, Dict[str, str]]:
        return self.data.get("providers", {})
        
    @property
    def model(self) -> str:
        return self.data.get("model", "")
        
    @property
    def extra_api_parameters(self) -> Dict[str, Any]:
        return self.data.get("extra_api_parameters", {})