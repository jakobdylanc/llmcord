import logging
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Tuple, Optional

from openai import AsyncOpenAI

from config import Config


class LLMClient:
    """
    Handles communication with the LLM API.
    Supports different providers and models.
    """
    def __init__(self, config: Config):
        self.config = config
        
    def get_client(self, provider: str) -> AsyncOpenAI:
        """Create an OpenAI client for the specified provider."""
        base_url = self.config.providers[provider]["base_url"]
        api_key = self.config.providers[provider].get("api_key", "sk-no-key-required")
        return AsyncOpenAI(base_url=base_url, api_key=api_key)
        
    def model_supports_images(self, model: str) -> bool:
        """Check if the model supports image inputs."""
        return any(tag in model.lower() for tag in self.config.VISION_MODEL_TAGS)
        
    def provider_supports_usernames(self, provider: str) -> bool:
        """Check if the provider supports the username/name parameter."""
        return any(p in provider.lower() for p in self.config.PROVIDERS_SUPPORTING_USERNAMES)
        
    def prepare_system_message(self, model: str, provider: str) -> Dict[str, str]:
        """Prepare the system message with appropriate context."""
        if not self.config.system_prompt:
            return {}
            
        system_prompt_extras = [f"Today's date: {datetime.now().strftime('%B %d %Y')}."]
        
        if self.provider_supports_usernames(provider):
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")
            
        full_system_prompt = "\n".join([self.config.system_prompt] + system_prompt_extras)
        return {"role": "system", "content": full_system_prompt}
        
    async def generate_response(self, messages: List[Dict[str, Any]]) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """
        Generate a response from the LLM.
        Yields tuples of (content_delta, finish_reason).
        """
        provider, model = self.config.model.split("/", 1)
        client = self.get_client(provider)
        
        # Add system message if configured
        system_message = self.prepare_system_message(model, provider)
        if system_message:
            messages.append(system_message)
            
        # Reverse messages to have them in chronological order
        messages = messages[::-1]
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "stream": True,
                "extra_body": self.config.extra_api_parameters
            }
            
            async for chunk in await client.chat.completions.create(**kwargs):
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                content = delta.content or ""
                
                yield content, finish_reason
                
                if finish_reason is not None:
                    break
                    
        except Exception as e:
            logging.exception(f"Error generating LLM response: {e}")
            yield f"Error generating response: {str(e)}", "error"