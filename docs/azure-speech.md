# Azure Speech Configuration

This guide explains how to enable voice features (TTS and STT) using Azure Speech Services.

## Prerequisites

- An Azure subscription
- A Speech resource created in [Azure Portal](https://portal.azure.com/#create/Microsoft.CognitiveServicesSpeechServices)

## Setup

1. **Get your Azure Speech credentials**
   - Go to [Azure Portal](https://portal.azure.com) → Your Speech resource
   - Copy the **Key** (either Key1 or Key2)
   - Note your **Location/Region** (e.g., `eastus`, `westeurope`, `japaneast`)

2. **Update config.yaml**
   ```yaml
   azure-speech:
     key: "your-azure-speech-key"
     region: "eastus"
   ```

## Configuration Options

| Field | Required | Description |
|-------|----------|-------------|
| `key` | Yes | Your Azure Speech API key |
| `region` | Yes | Your Azure region (e.g., eastus, westeurope) |
| `endpoint` | No | Custom endpoint (rarely needed) |
| `default_voice` | No | Default voice for TTS (e.g., `en-US-JennyNeural`) |
| `default_style` | No | Default speaking style (e.g., `cheerful`, `sad`, `neutral`) |

## Available Voices

You can find all available Azure voices on the [Microsoft documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/text-to-speech).

Popular English voices:
- `en-US-JennyNeural` - Female, conversational
- `en-US-GuyNeural` - Male, professional
- `en-GB-SoniaNeural` - Female, British
- `en-AU-NatashaNeural` - Female, Australian

## Speaking Styles

Available styles vary by voice, but common ones include:
- `cheerful` - Happy, positive
- `sad` - Depressed, somber
- `angry` - Frustrated, upset
- `neutral` - Default, neutral tone
- `excited` - Enthusiastic
- `friendly` - Warm, approachable
- `whispering` - Quiet, soft

## Voice Commands

Once configured, the following commands are available:

| Command | Description |
|---------|-------------|
| `/join` | Join your current voice channel |
| `/leave` | Leave the current voice channel |
| `/speak <text>` | Make the bot speak the text in voice channel |

## Voice Messages

When Azure Speech is configured, the bot can also:
- **Transcribe voice messages** sent in text channels or DMs
- Process audio attachments automatically

## Testing

Test your TTS configuration by:
1. Joining a voice channel
2. Running `/speak Hello world`

The bot should speak the text aloud in the voice channel.

## Troubleshooting

- **"TTS is not configured"**: Ensure `key` and `region` are set in config.yaml
- **Authentication errors**: Verify your Azure key is correct
- **Region mismatch**: Ensure the region matches your Speech resource's location
- **Bot can't join voice**: Check that the bot has "Connect" and "Speak" permissions in the voice channel