import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import io
import logging
from typing import Any, Literal, Optional

from PIL import Image

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

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

SUPPORTED_IMAGE_TYPES = ("image/jpeg", "image/png")


def _ensure_jpeg_or_png(content_type: str, data: bytes, max_size: int = 1024) -> tuple[str, bytes]:
    """Convert image to JPEG if it's not already JPEG or PNG, and cap resolution."""
    img = Image.open(io.BytesIO(data))
    resized = max(img.size) > max_size
    if resized:
        img.thumbnail((max_size, max_size))
    if not resized and content_type in SUPPORTED_IMAGE_TYPES:
        return content_type, data
    fmt = "PNG" if content_type == "image/png" else "JPEG"
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return f"image/{fmt.lower()}", buf.getvalue()


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private
    is_thread = new_msg.channel.type in (discord.ChannelType.public_thread, discord.ChannelType.private_thread)

    if new_msg.author.bot:
        return
    if new_msg.content.startswith("."):
        return
    is_mentioned = is_dm or is_thread or discord_bot.user in new_msg.mentions

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

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

    # Gate check: if not explicitly mentioned, use a smaller model to decide whether to interject
    if not is_mentioned:
        interjection_model_name = config.get("interjection_model")
        if not interjection_model_name:
            return

        # Fetch recent messages for the gate to evaluate
        gate_msgs = []
        async for msg in new_msg.channel.history(limit=10):
            if msg.type in (discord.MessageType.default, discord.MessageType.reply):
                gate_msgs.append(msg)

        gate_context = "\n".join(
            f"{'[Assistant]' if m.author == discord_bot.user else f'[{m.author.display_name}]'}: {m.content}"
            for m in reversed(gate_msgs)
        )

        ij_provider, ij_model = interjection_model_name.removesuffix(":vision").split("/", 1)
        ij_provider_config = config["providers"][ij_provider]
        ij_client = AsyncOpenAI(
            base_url=ij_provider_config["base_url"],
            api_key=ij_provider_config.get("api_key", "sk-no-key-required"),
        )

        gate_system = (
            "You decide whether an AI assistant should interject in a Discord conversation. "
            "Be conservative. Only say YES if:\n"
            "- Someone says hi to the assistant (the assistant is called Claude)"
            "- Someone is asking a question that hasn't been answered\n"
            "- Someone is directly asking for help or information the assistant could provide\n"
            "- The assistant was recently part of the conversation and a follow-up is natural\n\n"
            "Do NOT interject if users are just chatting, joking, or having a normal conversation. "
            "When in doubt, say NO.\n"
            "Respond with only YES or NO."
        )

        try:
            gate_resp = await ij_client.chat.completions.create(
                model=ij_model,
                messages=[
                    dict(role="system", content=gate_system),
                    dict(role="user", content=gate_context),
                ],
                max_tokens=3,
            )
            should_interject = gate_resp.choices[0].message.content.strip().upper().startswith("YES")
            if not should_interject:
                return
            logging.info(f"Interjection approved for message {new_msg.id}")
        except Exception:
            logging.exception("Interjection gate check failed")
            return  # Conservative: don't interject on error

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    nodes_needing_descriptions = []

    async def process_msg(msg: discord.Message) -> tuple[dict | None, set]:
        """Process a single message into a chat message dict and warnings."""
        warnings = set()
        node = msg_nodes.setdefault(msg.id, MsgNode())

        async with node.lock:
            if node.text == None:
                cleaned_content = msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                node.role = "assistant" if msg.author == discord_bot.user else "user"

                node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in msg.embeds]
                    + [component.content for component in msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{mime};base64,{b64encode(data).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                    for mime, data in [_ensure_jpeg_or_png(att.content_type, resp.content)]
                ]

                if node.role == "user" and (node.text or node.images):
                    node.text = f"<@{msg.author.id}>: {node.text}"

                node.has_bad_attachments = len(msg.attachments) > len(good_attachments)

            if node.images[:max_images] and node.image_descriptions:
                img_notes = "\n".join(f"[Image: {desc}]" for desc in node.image_descriptions[:max_images])
                text_part = node.text[:max_text]
                content = f"{text_part}\n{img_notes}" if text_part else img_notes
            elif node.images[:max_images]:
                content = [dict(type="text", text=node.text[:max_text])] + node.images[:max_images]
                nodes_needing_descriptions.append(node)
            else:
                content = node.text[:max_text]

            if len(node.text) > max_text:
                warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(node.images) > max_images:
                warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if node.has_bad_attachments:
                warnings.add("⚠️ Unsupported attachments")

            if content != "":
                return dict(content=content, role=node.role), warnings
            return None, warnings

    earliest_msg_time = new_msg.created_at

    if is_thread:
        # In threads, fetch full history chronologically (oldest first)
        thread_msgs = []
        try:
            # Fetch the thread starter message if this is a public thread
            if new_msg.channel.type == discord.ChannelType.public_thread and new_msg.channel.parent.type == discord.ChannelType.text:
                starter = new_msg.channel.starter_message or await new_msg.channel.parent.fetch_message(new_msg.channel.id)
                if starter:
                    thread_msgs.append(starter)
        except (discord.NotFound, discord.HTTPException):
            logging.exception("Error fetching thread starter message")

        async for msg in new_msg.channel.history(limit=max_messages, oldest_first=True):
            if msg.type in (discord.MessageType.default, discord.MessageType.reply):
                thread_msgs.append(msg)

        # Trim to max_messages (keep the most recent ones)
        if len(thread_msgs) > max_messages:
            thread_msgs = thread_msgs[-max_messages:]
            user_warnings.add(f"⚠️ Only using last {max_messages} message{'' if max_messages == 1 else 's'}")

        for msg in thread_msgs:
            result, warnings = await process_msg(msg)
            user_warnings |= warnings
            if result:
                messages.append(result)

        if thread_msgs:
            earliest_msg_time = thread_msgs[0].created_at
    elif is_dm:
        # Original reply-chain behavior for DMs
        curr_msg = new_msg

        while curr_msg != None and len(messages) < max_messages:
            curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

            async with curr_node.lock:
                if curr_node.text == None:
                    cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                    good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                    curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                    curr_node.text = "\n".join(
                        ([cleaned_content] if cleaned_content else [])
                        + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                        + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                        + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                    )

                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{mime};base64,{b64encode(data).decode('utf-8')}"))
                        for att, resp in zip(good_attachments, attachment_responses)
                        if att.content_type.startswith("image")
                        for mime, data in [_ensure_jpeg_or_png(att.content_type, resp.content)]
                    ]

                    if curr_node.role == "user" and (curr_node.text or curr_node.images):
                        curr_node.text = f"<@{curr_msg.author.id}>: {curr_node.text}"

                    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                    try:
                        if (
                            curr_msg.reference == None
                            and discord_bot.user.mention not in curr_msg.content
                            and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                            and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                            and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                        ):
                            curr_node.parent_msg = prev_msg_in_channel
                        else:
                            is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                            parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                            if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                                if parent_is_thread_start:
                                    curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                                else:
                                    curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                    except (discord.NotFound, discord.HTTPException):
                        logging.exception("Error fetching next message in the chain")
                        curr_node.fetch_parent_failed = True

                if curr_node.images[:max_images] and curr_node.image_descriptions:
                    img_notes = "\n".join(f"[Image: {desc}]" for desc in curr_node.image_descriptions[:max_images])
                    text_part = curr_node.text[:max_text]
                    content = f"{text_part}\n{img_notes}" if text_part else img_notes
                elif curr_node.images[:max_images]:
                    content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images]
                    nodes_needing_descriptions.append(curr_node)
                else:
                    content = curr_node.text[:max_text]

                if content != "":
                    messages.append(dict(content=content, role=curr_node.role))
                    earliest_msg_time = curr_msg.created_at

                if len(curr_node.text) > max_text:
                    user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
                if len(curr_node.images) > max_images:
                    user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
                if curr_node.has_bad_attachments:
                    user_warnings.add("⚠️ Unsupported attachments")
                if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                    user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

                curr_msg = curr_node.parent_msg

    else:
        # Fetch recent channel history with gap-based and token-based cutoffs
        context_gap_minutes = config.get("context_gap_minutes", 10)
        max_context_tokens = config.get("max_context_tokens", 10000)

        recent_msgs = []
        estimated_tokens = 0
        prev_msg_time = new_msg.created_at

        async for msg in new_msg.channel.history(limit=max_messages, before=new_msg):
            if msg.type not in (discord.MessageType.default, discord.MessageType.reply):
                continue

            # Check for silence gap
            time_gap = (prev_msg_time - msg.created_at).total_seconds() / 60
            if time_gap > context_gap_minutes:
                break

            # Estimate tokens (rough: ~4 chars per token)
            msg_tokens = len(msg.content) / 4
            if estimated_tokens + msg_tokens > max_context_tokens:
                break

            estimated_tokens += msg_tokens
            recent_msgs.append(msg)
            prev_msg_time = msg.created_at

        # Process in chronological order (oldest first)
        for msg in reversed(recent_msgs):
            result, warnings = await process_msg(msg)
            user_warnings |= warnings
            if result:
                messages.append(result)

        # Always include the current message
        result, warnings = await process_msg(new_msg)
        user_warnings |= warnings
        if result:
            messages.append(result)

        if recent_msgs:
            earliest_msg_time = recent_msgs[-1].created_at

        if len(recent_msgs) >= max_messages:
            user_warnings.add(f"⚠️ Only using last {max_messages} message{'s' if max_messages != 1 else ''}")

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    ordered_messages = messages[::-1] if is_dm else messages
    openai_kwargs = dict(model=model, messages=ordered_messages, stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    earliest_timestamp = int(earliest_msg_time.timestamp())
    context_info = f"Earliest message: <t:{earliest_timestamp}:t>"

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        fields = [dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]
        fields.append(dict(name=context_info, value="", inline=False))
        embed = discord.Embed.from_dict(dict(fields=fields))

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
                for i, content in enumerate(response_contents):
                    if i == 0:
                        content = f"-# {context_info}\n{content}"
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception as e:
        logging.exception("Error while generating response")
        if not response_msgs:
            error_type = type(e).__name__
            error_brief = str(e).split("\n")[0][:200] if str(e) else "Unknown error"
            error_text = f"⚠️ **{error_type}**: {error_brief}"
            try:
                if use_plain_responses:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=error_text)))
                else:
                    error_embed = discord.Embed(description=error_text, color=EMBED_COLOR_INCOMPLETE)
                    await reply_helper(embed=error_embed, silent=True)
            except Exception:
                logging.exception("Error while sending error message")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Generate and cache image descriptions for nodes that sent real images
    for node in nodes_needing_descriptions:
        if node.image_descriptions:
            continue
        descriptions = []
        for img in node.images[:max_images]:
            try:
                desc_resp = await openai_client.chat.completions.create(
                    model=model,
                    messages=[dict(role="user", content=[
                        dict(type="text", text="Describe this image in one brief sentence."),
                        img,
                    ])],
                    max_tokens=100,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                )
                descriptions.append(desc_resp.choices[0].message.content.strip())
            except Exception:
                logging.exception("Error generating image description")
                descriptions.append("(image description unavailable)")
        node.image_descriptions = descriptions
        logging.info(f"Cached {len(descriptions)} image description(s) for message node")

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
