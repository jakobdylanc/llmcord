#!/usr/bin/env python3
import asyncio
import logging
import os

from config import Config
from discord_client import LLMCordClient


async def main():
    """Main entry point for the LLMCord bot."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = Config(config_path)
    
    if not config.bot_token:
        logging.error("Bot token not found in configuration. Please set 'bot_token' in your config file.")
        return
        
    # Initialize and run the Discord client
    client = LLMCordClient(config)
    
    try:
        await client.start(config.bot_token)
    except KeyboardInterrupt:
        logging.info("Bot shutting down...")
        await client.close()
    except Exception as e:
        logging.exception(f"Error starting bot: {e}")


if __name__ == "__main__":
    asyncio.run(main())