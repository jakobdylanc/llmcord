import asyncio
from base64 import b64encode, b64decode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional
import io

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
EMBED_COLOR_ERROR = discord.Color.red()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()

# Fix: Provide default values if keys are missing
if "models" not in config:
    # Create models structure from llm.model if it exists
    llm_model = config.get("llm", {}).get("model")
    if llm_model:
        config["models"] = {llm_model: {}}
    else:
        config["models"] = {}

# Create a proper permissions structure based on your existing config
if "permissions" not in config:
    config["permissions"] = {
        "users": {
            "admin_ids": config.get("admin_user_ids", []),
            "allowed_ids": [],
            "blocked_ids": []
        },
        "roles": {
            "allowed_ids": [],
            "blocked_ids": []
        },
        "channels": {
            "allowed_ids": config.get("allowed_channel_ids", []),
            "blocked_ids": []
        }
    }

# Track current provider and model - initialize properly at module level
current_provider = None
current_model = None
current_image_provider = None

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Image generation function for local providers
async def generate_image(prompt: str, negative_prompt: str, provider_config: dict, parameters: dict = None) -> Optional[str]:
    """Generate an image using a local provider (like Stable Diffusion WebUI Forge)"""
    try:
        forge_url = provider_config.get("forge_url")
        if not forge_url:
            logging.error("Forge URL not configured for image generation")
            return None
            
        # Default parameters from config
        default_params = provider_config.get("default_params", {})
        default_model = provider_config.get("default_model", "novaMatureXL_v35")
        default_size = provider_config.get("default_size", "768x768")
        
        # Parse size
        width, height = map(int, default_size.split('x'))
        
        # Merge parameters with defaults
        final_params = default_params.copy()
        if parameters:
            final_params.update(parameters)
        
        # Apply overrides from function parameters
        if "negative_prompt" in final_params:
            final_params["negative_prompt"] = negative_prompt or final_params["negative_prompt"]
        else:
            final_params["negative_prompt"] = negative_prompt or ""
            
        # Set dimensions from parameters or defaults
        final_params["width"] = final_params.get("width", width)
        final_params["height"] = final_params.get("height", height)
        
        # Prepare the payload - corrected for WebUI Forge API
        payload = {
            "prompt": prompt,
            "negative_prompt": final_params["negative_prompt"],
            "steps": final_params.get("steps", 40),
            "cfg_scale": final_params.get("cfg_scale", 7.0),
            "sampler_name": final_params.get("sampler_name", "DPM++ 2M Karras"),
            "restore_faces": final_params.get("restore_faces", False),
            "width": final_params["width"],
            "height": final_params["height"],
            "override_settings": {
                "sd_model_checkpoint": default_model
            }
        }
        
        # Add any additional parameters from config
        for key, value in final_params.items():
            if key not in ["negative_prompt", "steps", "cfg_scale", "sampler_name", "restore_faces", "width", "height"]:
                payload[key] = value
        
        headers = {"Content-Type": "application/json"}
        
        logging.info(f"Sending image generation request to {forge_url}")
        logging.info(f"Payload: {payload}")
        
        async with httpx.AsyncClient(timeout=provider_config.get("timeout", 720.0)) as client:
            response = await client.post(
                f"{forge_url}/sdapi/v1/txt2img",
                json=payload,
                headers=headers
            )
            
            logging.info(f"Response status: {response.status_code}")
            logging.info(f"Response text (first 200 chars): {response.text[:200]}...")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logging.info(f"Response data keys: {list(data.keys())}")
                    if "images" in data and len(data["images"]) > 0:
                        # Return the first image (base64 encoded)
                        return data["images"][0]
                    else:
                        logging.error("No images returned from image generation")
                        logging.error(f"Full response: {data}")
                        return None
                except Exception as e:
                    logging.exception(f"Error parsing JSON response: {e}")
                    logging.error(f"Raw response: {response.text}")
                    return None
            else:
                logging.error(f"Image generation failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logging.exception(f"Error generating image: {e}")
        return None


# Image generation command that uses a simpler approach
@discord_bot.tree.command(name="image", description="Generate an image from a prompt")
async def image_command(interaction: discord.Interaction, 
                       prompt: str,
                       negative_prompt: Optional[str] = None) -> None:
    """Generate an image based on a text prompt"""
    
    # Defer response to show we're working
    await interaction.response.defer(ephemeral=False)
    
    # Check if image generation is configured
    provider_config = config.get("llm", {}).get("image_generator", {})
    if not provider_config:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    # Generate the image with default parameters
    image_data = await generate_image(prompt, negative_prompt or "", provider_config, {})
    
    if image_data:
        try:
            # Handle base64 image data - remove the "data:image/png;base64," prefix if present
            if "," in image_data:
                _, image_data = image_data.split(",", 1)
            
            # Decode base64 image data
            image_bytes = b64decode(image_data)
            
            # Create a Discord file object
            file = discord.File(io.BytesIO(image_bytes), filename="generated_image.png")
            
            # Send the image
            embed = discord.Embed(title=f"Generated Image for: {prompt[:50]}...", color=EMBED_COLOR_COMPLETE)
            if negative_prompt:
                embed.description = f"Negative prompt: {negative_prompt[:50]}..."
            await interaction.followup.send(embed=embed, file=file)
            
        except Exception as e:
            logging.exception(f"Error sending generated image: {e}")
            await interaction.followup.send("Failed to send generated image.", ephemeral=True)
    else:
        await interaction.followup.send("Failed to generate image. Please check the logs for details.", ephemeral=True)


# Alternative approach: Use a simple button-based workflow instead of modal
@discord_bot.tree.command(name="image_advanced", description="Generate an image with advanced parameters")
async def image_advanced_command(interaction: discord.Interaction, 
                                prompt: str,
                                negative_prompt: Optional[str] = None) -> None:
    """Generate an image with advanced options"""
    
    # Simple approach - just defer and process immediately
    await interaction.response.defer(ephemeral=False)
    
    # Check if image generation is configured
    provider_config = config.get("llm", {}).get("image_generator", {})
    if not provider_config:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    # Generate the image with default parameters
    image_data = await generate_image(prompt, negative_prompt or "", provider_config, {})
    
    if image_data:
        try:
            # Handle base64 image data - remove the "data:image/png;base64," prefix if present
            if "," in image_data:
                _, image_data = image_data.split(",", 1)
            
            # Decode base64 image data
            image_bytes = b64decode(image_data)
            
            # Create a Discord file object
            file = discord.File(io.BytesIO(image_bytes), filename="generated_image.png")
            
            # Send the image
            embed = discord.Embed(title=f"Generated Image for: {prompt[:50]}...", color=EMBED_COLOR_COMPLETE)
            if negative_prompt:
                embed.description = f"Negative prompt: {negative_prompt[:50]}..."
            await interaction.followup.send(embed=embed, file=file)
            
        except Exception as e:
            logging.exception(f"Error sending generated image: {e}")
            await interaction.followup.send("Failed to send generated image.", ephemeral=True)
    else:
        await interaction.followup.send("Failed to generate image. Please check the logs for details.", ephemeral=True)


# New /providers command to switch between different endpoints
@discord_bot.tree.command(name="providers", description="Switch between different providers/endpoints")
async def providers_command(interaction: discord.Interaction, provider: str) -> None:
    """Switch to a different provider"""
    global current_provider, current_model
    
    # Get available providers from config
    llm_config = config.get("llm", {})
    providers = llm_config.get("providers", {})
    
    if not providers:
        await interaction.response.send_message("No providers configured.", ephemeral=True)
        return
        
    if provider not in providers:
        available_providers = list(providers.keys())
        await interaction.response.send_message(
            f"Provider '{provider}' not found. Available providers: {', '.join(available_providers)}", 
            ephemeral=True
        )
        return
    
    # Set the new provider and model
    current_provider = provider
    provider_config = providers[provider]
    
    # Try to get the default model for this provider
    # For Ollama, we'll use the main model from config or set a default
    default_model = provider_config.get("model") or "default"
    
    # If no specific model is set, try to find one from the models section
    if default_model == "default":
        # Check if there's a model in the main llm section that matches this provider
        llm_model = config.get("llm", {}).get("model", "")
        if llm_model and provider in llm_model:
            default_model = llm_model.split("/", 1)[1] if "/" in llm_model else llm_model
        else:
            # Fallback to a reasonable default for ollama - let's make it more flexible
            # We'll just use a generic name since we don't know the actual model names
            default_model = "qwen3"  # This will be overridden by the actual model name
    
    current_model = default_model
    output = f"Switched to provider: `{provider}` with model: `{default_model}`"
        
    logging.info(output)
    await interaction.response.send_message(output, ephemeral=True)


@providers_command.autocomplete("provider")
async def provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    """Autocomplete for provider names"""
    llm_config = config.get("llm", {})
    providers = llm_config.get("providers", {})
    
    if not providers:
        return []
        
    # Filter providers based on search string
    filtered_providers = [p for p in providers.keys() if curr_str.lower() in p.lower()]
    
    # Return up to 25 choices
    choices = [Choice(name=p, value=p) for p in filtered_providers[:25]]
    return choices


# New /image_providers command to switch between different image generation endpoints
@discord_bot.tree.command(name="image_providers", description="Switch between different image generation providers")
async def image_providers_command(interaction: discord.Interaction, provider: str) -> None:
    """Switch to a different image generation provider"""
    global current_image_provider
    
    # Get available image providers from config
    image_generators = config.get("llm", {}).get("image_generators", {})
    
    if not image_generators:
        await interaction.response.send_message("No image generators configured.", ephemeral=True)
        return
        
    if provider not in image_generators:
        available_providers = list(image_generators.keys())
        await interaction.response.send_message(
            f"Image provider '{provider}' not found. Available providers: {', '.join(available_providers)}", 
            ephemeral=True
        )
        return
    
    # Set the new image provider
    current_image_provider = provider
    output = f"Switched to image provider: `{provider}`"
        
    logging.info(output)
    await interaction.response.send_message(output, ephemeral=True)


@image_providers_command.autocomplete("provider")
async def image_provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    """Autocomplete for image provider names"""
    image_generators = config.get("llm", {}).get("image_generators", {})
    
    if not image_generators:
        return []
        
    # Filter providers based on search string
    filtered_providers = [p for p in image_generators.keys() if curr_str.lower() in p.lower()]
    
    # Return up to 25 choices
    choices = [Choice(name=p, value=p) for p in filtered_providers[:25]]
    return choices


# Vision command - analyze images with the bot
@discord_bot.tree.command(name="analyze", description="Analyze an image with the bot (mention the bot in your message)")
async def analyze_command(interaction: discord.Interaction, prompt: str) -> None:
    """Analyze an image with a specific prompt"""
    await interaction.response.defer(ephemeral=False)
    
    # This command is now deprecated since we handle vision via mentions
    await interaction.followup.send("Use @bot mention with an image attachment and a question about it instead.", ephemeral=True)


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    # Sync commands when bot is ready
    try:
        synced = await discord_bot.tree.sync()
        logging.info(f"Synced {len(synced)} command(s)")
        for cmd in synced:
            logging.info(f"  - {cmd.name}")
    except Exception as e:
        logging.error(f"Failed to sync commands: {e}")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time, current_provider, current_model, current_image_provider

    is_dm = new_msg.channel.type == discord.ChannelType.private

    # Skip if it's a bot message
    if new_msg.author.bot:
        return

    # The key fix: Allow processing of all non-empty user messages
    should_process = False
    
    # Always process DMs and non-empty messages in channels
    if is_dm or new_msg.content.strip() != "":
        should_process = True

    if not should_process:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    # Fix: Safely access permissions with fallback
    permissions = config.get("permissions", {
        "users": {
            "admin_ids": config.get("admin_user_ids", []),
            "allowed_ids": [],
            "blocked_ids": []
        },
        "roles": {
            "allowed_ids": [],
            "blocked_ids": []
        },
        "channels": {
            "allowed_ids": config.get("allowed_channel_ids", []),
            "blocked_ids": []
        }
    })

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    # Use the current provider/model setup - FIXED: Properly handle global variables
    llm_config = config.get("llm", {})
    
    # If no provider set yet, try to get from config
    if current_provider is None:
        providers = llm_config.get("providers", {})
        if providers:
            # Get first available provider
            current_provider = list(providers.keys())[0]
            provider_config = providers[current_provider]
            
            # Try to get model from main llm config
            default_model = config.get("llm", {}).get("model")
            if default_model:
                # Extract model name if it's in format "provider/model"
                if "/" in default_model:
                    current_model = default_model.split("/", 1)[1]
                else:
                    current_model = default_model
            else:
                current_model = "qwen3"  # Default fallback
        else:
            logging.error("No providers configured!")
            return
    
    # Get provider configuration
    provider_config = llm_config.get("providers", {}).get(current_provider, {})
    
    if not provider_config:
        logging.error(f"Provider configuration not found for: {current_provider}")
        return

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Fix: Handle models properly - use the llm.model value if needed
    model_parameters = {}
    # Only access models if they exist and are properly structured
    if "models" in config and current_model:
        model_key = f"{current_provider}/{current_model}"
        model_parameters = config["models"].get(model_key, {}) or {}

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    # Check if current model supports vision by checking the model name directly
    model_name_lower = (current_model or "").lower()
    accept_images = any(tag in model_name_lower for tag in VISION_MODEL_TAGS) if current_model else False
    accept_usernames = any(current_provider.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = 5 if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = new_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in new_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in new_msg.embeds]
                    + [component.content for component in new_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if new_msg.author == discord_bot.user else "user"

                curr_node.user_id = new_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(new_msg.attachments) > len(good_attachments)

                try:
                    if (
                        new_msg.reference == None
                        and discord_bot.user.mention not in new_msg.content
                        and (prev_msg_in_channel := ([m async for m in new_msg.channel.history(before=new_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if new_msg.channel.type == discord.ChannelType.private else new_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = new_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and new_msg.reference == None and new_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := new_msg.channel.id if parent_is_thread_start else getattr(new_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = new_msg.channel.starter_message or await new_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = new_msg.reference.cached_message or await new_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            # Check if we have images to process
            has_images = bool(curr_node.images[:max_images])
            
            if has_images:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            
            # Only add image warnings when there are actual images to process
            if len(curr_node.images) > 0:
                if len(curr_node.images) > max_images:
                    user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message")
                elif not accept_images:
                    user_warnings.add("⚠️ Can't see images")
            elif curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    # Check if this message is a vision request (mentions bot with an image)
    has_image_attachments = any(att.content_type and att.content_type.startswith("image") for att in new_msg.attachments)
    is_vision_request = (
        discord_bot.user.mention in new_msg.content and 
        has_image_attachments
    )

    # If it's a vision request, we'll process it differently
    if is_vision_request:
        # Extract the question from the message after mentioning the bot
        question = new_msg.content.removeprefix(discord_bot.user.mention).strip()
        
        # If no question provided, use a default one
        if not question:
            question = "What is in this image?"
            
        # Add the question as a text part to the content
        if messages and messages[-1]["role"] == "user":
            # Modify the last user message to include the question
            if isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append({"type": "text", "text": question})
            else:
                messages[-1]["content"] = [{"type": "text", "text": question}]
        else:
            # Create a new user message with the question
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            })
            
        # Ensure we have images in the message
        if not any(isinstance(c, dict) and c.get("type") == "image_url" for msg in messages for c in msg.get("content", [])):
            # If no images found in existing messages, add the ones from this message
            image_content = []
            for att in new_msg.attachments:
                if att.content_type and att.content_type.startswith("image"):
                    try:
                        response = await httpx_client.get(att.url)
                        image_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{att.content_type};base64,{b64encode(response.content).decode('utf-8')}"
                            }
                        })
                    except Exception as e:
                        logging.exception(f"Error fetching image: {e}")
                        continue
            
            if image_content:
                # Add to the last user message or create a new one
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] = image_content + ([{"type": "text", "text": question}] if question else [])
                else:
                    messages.append({
                        "role": "user",
                        "content": image_content + ([{"type": "text", "text": question}] if question else [])
                    })

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    # Add system prompt at the beginning of messages - FIXED VERSION
    system_prompt = llm_config.get("system_prompt") or ""
    
    if system_prompt:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        # IMPORTANT: Insert system prompt at the beginning, not append it at the end
        messages.insert(0, dict(role="system", content=system_prompt))
    else:
        logging.info("No system prompt found in config")

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    # Use current model or fallback to first available
    model_to_use = current_model or "qwen3"
    
    openai_kwargs = dict(model=model_to_use, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception as e:
        logging.exception(f"Error while generating response: {e}")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
