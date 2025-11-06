import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional

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

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


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
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

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

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    use_channel_context = config.get("use_channel_context", False)
    prefix_with_user_id = config.get("prefix_with_user_id", False)

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
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    # ------------------------------------------------------------------
    # Build the list of messages to feed the LLM
    # ------------------------------------------------------------------
    messages: list[dict] = []
    user_warnings: set[str] = set()
    message_history: list[discord.Message] = []
    if use_channel_context:
        # ---- Full‑channel mode -------------------------------------------------
        # Discord returns newest → oldest, so we start with the current message
        message_history.append(new_msg)
        async for msg in new_msg.channel.history(limit=max_messages - 1, before=new_msg):
            message_history.append(msg)
        # Ensure we never exceed max_messages
        message_history = message_history[:max_messages]
    else:
        # ---- Reply‑chain mode (original logic) ---------------------------------
        curr = new_msg
        while curr and len(message_history) < max_messages:
            message_history.append(curr)
            # Resolve the next parent (same logic as original)
            try:
                if (
                    curr.reference is None
                    and discord_bot.user.mention not in curr.content
                    and (prev := ([m async for m in curr.channel.history(before=curr, limit=1)] or [None])[0])
                    and prev.type in (discord.MessageType.default, discord.MessageType.reply)
                    and prev.author == (discord_bot.user if curr.channel.type == discord.ChannelType.private else curr.author)
                ):
                    curr = prev
                else:
                    is_thread = curr.channel.type == discord.ChannelType.public_thread
                    thread_start = is_thread and curr.reference is None and curr.channel.parent.type == discord.ChannelType.text
                    parent_id = curr.channel.id if thread_start else getattr(curr.reference, "message_id", None)
                    if parent_id:
                        if thread_start:
                            curr = curr.channel.starter_message or await curr.channel.parent.fetch_message(parent_id)
                        else:
                            curr = curr.reference.cached_message or await curr.channel.fetch_message(parent_id)
                    else:
                        curr = None
            except (discord.NotFound, discord.HTTPException):
                logging.exception("Error walking reply chain")
                curr = None
    # ------------------------------------------------------------------
    # Process each message (shared for both modes)
    # ------------------------------------------------------------------
    for msg in message_history:
        if len(messages) >= max_messages:
            break
        node = msg_nodes.setdefault(msg.id, MsgNode())
        async with node.lock:
            if node.text is None:
                cleaned = msg.content.removeprefix(discord_bot.user.mention).lstrip()
                good_attachments = [
                    att for att in msg.attachments
                    if att.content_type and any(att.content_type.startswith(t) for t in ("text", "image"))
                ]
                att_resps = await asyncio.gather(*[httpx_client.get(a.url) for a in good_attachments])
                node.text = "\n".join(
                    ([cleaned] if cleaned else [])
                    + ["\n".join(filter(None, (e.title, e.description, e.footer.text))) for e in msg.embeds]
                    + [c.content for c in msg.components if c.type == discord.ComponentType.text_display]
                    + [r.text for a, r in zip(good_attachments, att_resps) if a.content_type.startswith("text")]
                )
                node.images = [
                    dict(
                        type="image_url",
                        image_url=dict(url=f"data:{a.content_type};base64,{b64encode(r.content).decode()}")
                    )
                    for a, r in zip(good_attachments, att_resps) if a.content_type.startswith("image")
                ]
                node.role = "assistant" if msg.author == discord_bot.user else "user"
                node.user_id = msg.author.id if node.role == "user" else None
                node.has_bad_attachments = len(msg.attachments) > len(good_attachments)
                # Parent linking only needed for reply‑chain mode
                if not use_channel_context:
                    # (same parent‑fetch logic as above, omitted for brevity)
                    pass
            formatted_text = node.text[:max_text]  # base text (already trimmed)
            # Apply prefix only when:
            #   • toggle is enabled
            #   • provider does NOT support native usernames (accept_usernames == False)
            #   • the message role is "user"
            #   • we have a valid Discord user ID
            if prefix_with_user_id and not accept_usernames and node.role == "user" and node.user_id is not None:
                formatted_text = f"{node.user_id}: {formatted_text}"
                # keep node.text consistent for any later use
                node.text = formatted_text
            # ---- Build LLM message payload ----
            if node.images[:max_images]:
                content = ([dict(type="text", text=formatted_text)] if formatted_text else []) + node.images[:max_images]
            else:
                content = formatted_text
            if content:
                payload = dict(content=content, role=node.role)
                # Preserve native name field for providers that support it
                if accept_usernames and node.user_id:
                    payload["name"] = str(node.user_id)
                messages.append(payload)
            # ---- Warnings ----
            if len(node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if not use_channel_context and (node.fetch_parent_failed or (node.parent_msg and len(messages) == max_messages)):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'s' if len(messages) != 1 else ''}")

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        # Commented out, because it messes up with the end-of-prompt flags, like /nothink for GML. Users can specify this instruction on their own, and it can be added to config-example.
        # if accept_usernames:
        #     system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

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

    except Exception:
        logging.exception("Error while generating response")

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
