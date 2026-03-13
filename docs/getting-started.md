# Getting Started

This guide walks you through setting up the gpt-discord-bot.

## Prerequisites

- Python 3.10+
- Discord bot token
- At least one LLM provider (Ollama, OpenAI, OpenRouter, etc.)

## Quick Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/ckw1206/gpt-discord-bot
   cd gpt-discord-bot
   pip install -r requirements.txt
   ```

2. **Create config**
   ```bash
   cp config-example.yaml config.yaml
   ```

3. **Configure your bot**
   - Get bot token from [Discord Developer Portal](https://discord.com/developers/applications)
   - Enable MESSAGE CONTENT INTENT
   - Add token to config.yaml

4. **Run the bot**
   ```bash
   python llmcord.py
   ```

## Next Steps

- [Configure LLM Providers](configure-provider.md)
- [Add Tools](add-tool.md)
- [Set Up Personas](personas.md)

## For AI Agents

See [OpenSpec](../openspec/specs/) for technical details, APIs, and data models.