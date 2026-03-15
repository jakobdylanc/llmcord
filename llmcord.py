import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
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
MAX_MESSAGES_SANITY = 500

SUPPORTED_IMAGE_TYPES = ("image/jpeg", "image/png")

SESSION_GAP_SECONDS = 2 * 60 * 60  # 2 hours


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def message_token_estimate(msg_dict: dict) -> int:
    """Estimate tokens for an OpenAI message dict."""
    content = msg_dict["content"]
    if isinstance(content, str):
        return estimate_tokens(content)
    # list of content parts (text + images)
    return sum(estimate_tokens(p["text"]) for p in content if p.get("type") == "text")


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

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
    if fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"image/{fmt.lower()}", buf.getvalue()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes: dict[int, "MsgNode"] = {}
last_task_time = 0
active_channels: set[int] = set()
channel_locks: dict[int, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Memory system
# ---------------------------------------------------------------------------

class MemoryStore:
    """Encapsulates memory storage and sweep state. Swap this class to change backends."""

    def __init__(self, memory_file: Path, sweep_state_file: Path):
        self._memory_file = memory_file
        self._sweep_state_file = sweep_state_file
        self._sweep_lock = asyncio.Lock()

    def load(self) -> str:
        try:
            return self._memory_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def save(self, content: str) -> None:
        self._memory_file.write_text(content, encoding="utf-8")

    async def get_last_sweep_time(self, channel_id: int) -> datetime | None:
        async with self._sweep_lock:
            state = self._load_sweep_state()
            ts = state.get(str(channel_id))
            return datetime.fromisoformat(ts) if ts else None

    async def set_last_sweep_time(self, channel_id: int, dt: datetime) -> None:
        async with self._sweep_lock:
            state = self._load_sweep_state()
            state[str(channel_id)] = dt.isoformat()
            self._save_sweep_state(state)

    def _load_sweep_state(self) -> dict[str, str]:
        try:
            return json.loads(self._sweep_state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_sweep_state(self, state: dict[str, str]) -> None:
        self._sweep_state_file.write_text(json.dumps(state), encoding="utf-8")


memory_store = MemoryStore(
    memory_file=Path(__file__).parent / "memory.md",
    sweep_state_file=Path(__file__).parent / "sweep_state.json",
)


SWEEP_PROMPT = """\
You are Claude. You are accessing a discord channel via a discord bot. You are performing a memory review for yourself. Below are two inputs:

1. The current memory file
2. The messages from the previous session

Your job is to output an updated version of the complete memory file. Follow these rules:



PROMOTION:
- If a Scratch item came up multiple times or proved important, promote it to Active Context.
- If an Active Context item has been consistently relevant across sessions or represents something enduring, promote it to Identity.

DEMOTION / PRUNING:
- If a Scratch item seems trivial, one-off, or irrelevant in hindsight, remove it.
- If an Active Context item hasn't been relevant and seems stale, demote it to Scratch or remove it.
- Identity items should only be removed if they are clearly wrong or outdated.

NEW ENTRIES:
- Add new Scratch items for notable things from the session: preferences expressed, facts shared, projects mentioned, emotional context, anything that might matter later.
- Add entries to your section for self-correction notes, behavioral feedback, or operational observations.

GENERAL RULES:
- [IMPORTANT] Write casually but succintly in first person. These are your memories for you.
- Keep the file under 100 lines.
- Be aggressive about pruning. A tight file is more valuable than a comprehensive one.
- Preserve the exact file structure and formatting.
- Every entry should be a single concise line.
- When in doubt about importance, keep it in Scratch for one more cycle rather than promoting.

Output ONLY the updated memory file, followed by a blank line, then a single line in this format:
SUMMARY: [brief natural language description of what changed]

<current_memory_file>
{memory}
</current_memory_file>

<previous_session_messages>
{messages}
</previous_session_messages>"""


async def collect_previous_session(channel: discord.abc.Messageable, before_msg: discord.Message, bot_user: discord.User) -> list[dict[str, str]]:
    """Fetch the most recent previous session from Discord history.

    Walks backwards from before_msg. The first 2hr+ gap marks the boundary between
    the current session and the previous one. Once past that gap, collects messages
    until another 2hr+ gap or the last sweep time is reached.
    Returns messages in chronological order.
    """
    last_sweep = await memory_store.get_last_sweep_time(channel.id)
    session_msgs: list[dict[str, str]] = []
    found_session_start = False
    prev_time = before_msg.created_at

    async for msg in channel.history(limit=MAX_MESSAGES_SANITY, before=before_msg):
        if msg.type not in (discord.MessageType.default, discord.MessageType.reply):
            continue

        # Stop if we've gone past the last sweep
        if last_sweep and msg.created_at <= last_sweep:
            break

        gap = (prev_time - msg.created_at).total_seconds()

        if not found_session_start:
            # Still in the current session — skip until we cross a 2hr+ gap
            if gap >= SESSION_GAP_SECONDS:
                found_session_start = True
            else:
                prev_time = msg.created_at
                continue

        if found_session_start:
            # We're now in the previous session — collect until another gap
            if session_msgs and gap >= SESSION_GAP_SECONDS:
                break

            author = "Assistant" if msg.author == bot_user else msg.author.display_name
            session_msgs.append(dict(author=author, content=msg.content))

        prev_time = msg.created_at

    session_msgs.reverse()
    return session_msgs


async def run_memory_sweep(channel: discord.abc.Messageable, session_msgs: list[dict[str, str]], openai_client: AsyncOpenAI, model: str, **api_kwargs) -> None:
    """Run the memory sweep on a list of session messages."""
    if not session_msgs:
        await channel.send("🧠 Memory reviewed — no new messages to process.")
        return

    memory_content = memory_store.load()
    messages_text = "\n".join(f"[{m['author']}]: {m['content']}" for m in session_msgs)

    prompt = SWEEP_PROMPT.replace("{memory}", memory_content).replace("{messages}", messages_text)

    try:
        resp = await openai_client.chat.completions.create(
            model=model,
            messages=[dict(role="user", content=prompt)],
            max_tokens=2000,
            extra_headers=api_kwargs.get("extra_headers"),
            extra_query=api_kwargs.get("extra_query"),
            extra_body=api_kwargs.get("extra_body"),
        )
        output = resp.choices[0].message.content.strip()

        lines = output.split("\n")
        summary_line = ""
        memory_lines = []
        for i, line in enumerate(lines):
            if line.startswith("SUMMARY:"):
                summary_line = line[len("SUMMARY:"):].strip()
                memory_lines = lines[:i]
                break
        else:
            memory_lines = lines
            summary_line = "no summary provided"

        while memory_lines and memory_lines[-1].strip() == "":
            memory_lines.pop()

        new_memory = "\n".join(memory_lines) + "\n"
        memory_store.save(new_memory)

        await memory_store.set_last_sweep_time(channel.id, datetime.now(timezone.utc))

        if summary_line:
            await channel.send(f"🧠 Memory updated — {summary_line}")
        else:
            await channel.send("🧠 Memory reviewed — no updates.")

        logging.info(f"Memory sweep complete: {summary_line}")

    except Exception:
        logging.exception("Memory sweep failed")
        await channel.send("🧠 Memory sweep failed — keeping previous memories.")


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


# ---------------------------------------------------------------------------
# Helper: populate a MsgNode from a Discord message
# ---------------------------------------------------------------------------

async def populate_node(node: MsgNode, msg: discord.Message) -> None:
    """Fill in a MsgNode's text, images, role, and attachment info (idempotent)."""
    if node.text is not None:
        return

    cleaned_content = msg.content.removeprefix(discord_bot.user.mention).lstrip()

    good_attachments = [
        att for att in msg.attachments
        if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))
    ]
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

    if node.role == "assistant" and node.text:
        while node.text.startswith("-# "):
            node.text = node.text.split("\n", 1)[-1] if "\n" in node.text else ""

    if node.role == "user" and (node.text or node.images):
        node.text = f"<@{msg.author.id}>: {node.text}"

    if msg.reactions:
        reaction_strs = [f"{r.emoji} x{r.count}" for r in msg.reactions]
        node.text += f"\n[Reactions: {', '.join(reaction_strs)}]"

    node.has_bad_attachments = len(msg.attachments) > len(good_attachments)


def build_message_content(
    node: MsgNode,
    max_text: int,
    max_images: int,
    nodes_needing_descriptions: list[MsgNode],
) -> tuple[dict | None, set[str]]:
    """Build an OpenAI message dict from a populated MsgNode. Returns (message_dict, warnings)."""
    warnings: set[str] = set()

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


# ---------------------------------------------------------------------------
# Helper: resolve parent message for DM reply-chain walking
# ---------------------------------------------------------------------------

async def resolve_parent_msg(node: MsgNode, msg: discord.Message) -> None:
    """Populate node.parent_msg by checking reply references and channel history."""
    try:
        if (
            msg.reference is None
            and discord_bot.user.mention not in msg.content
            and (prev_msg_in_channel := ([m async for m in msg.channel.history(before=msg, limit=1)] or [None])[0])
            and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
            and prev_msg_in_channel.author == (discord_bot.user if msg.channel.type == discord.ChannelType.private else msg.author)
        ):
            node.parent_msg = prev_msg_in_channel
        else:
            is_public_thread = msg.channel.type == discord.ChannelType.public_thread
            parent_is_thread_start = is_public_thread and msg.reference is None and msg.channel.parent.type == discord.ChannelType.text

            if parent_msg_id := msg.channel.id if parent_is_thread_start else getattr(msg.reference, "message_id", None):
                if parent_is_thread_start:
                    node.parent_msg = msg.channel.starter_message or await msg.channel.parent.fetch_message(parent_msg_id)
                else:
                    node.parent_msg = msg.reference.cached_message or await msg.channel.fetch_message(parent_msg_id)

    except (discord.NotFound, discord.HTTPException):
        logging.exception("Error fetching next message in the chain")
        node.fetch_parent_failed = True


# ---------------------------------------------------------------------------
# Permission checking
# ---------------------------------------------------------------------------

def check_permissions(new_msg: discord.Message, cfg: dict) -> bool:
    """Return True if the user/channel is allowed to interact with the bot."""
    is_dm = new_msg.channel.type == discord.ChannelType.private

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    allow_dms = cfg.get("allow_dms", True)
    permissions = cfg["permissions"]

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

    return not is_bad_user and not is_bad_channel


# ---------------------------------------------------------------------------
# Interjection gate
# ---------------------------------------------------------------------------

async def check_interjection(new_msg: discord.Message, cfg: dict) -> bool:
    """Decide whether the bot should interject in a conversation it was not mentioned in."""
    if new_msg.channel.id in active_channels:
        return False

    interjection_model_name = cfg.get("interjection_model")
    if not interjection_model_name:
        return False

    gate_msgs = []
    async for msg in new_msg.channel.history(limit=10):
        if msg.type in (discord.MessageType.default, discord.MessageType.reply):
            gate_msgs.append(msg)

    gate_context = "\n".join(
        f"{'[Assistant]' if m.author == discord_bot.user else f'[{m.author.display_name}]'}: {m.content}"
        for m in reversed(gate_msgs)
    )

    ij_provider, ij_model = interjection_model_name.removesuffix(":vision").split("/", 1)
    ij_provider_config = cfg["providers"][ij_provider]
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
        if not gate_resp.choices or not gate_resp.choices[0].message.content:
            logging.warning("Interjection gate returned empty response")
            return False
        should_interject = gate_resp.choices[0].message.content.strip().upper().startswith("YES")
        if should_interject:
            logging.info(f"Interjection approved for message {new_msg.id}")
        return should_interject
    except Exception:
        logging.exception("Interjection gate check failed")
        return False


# ---------------------------------------------------------------------------
# Message chain builders
# ---------------------------------------------------------------------------

async def build_chain_thread(
    new_msg: discord.Message,
    max_text: int,
    max_images: int,
    max_context_tokens: int,
    nodes_needing_descriptions: list[MsgNode],
) -> tuple[list[dict], set[str], datetime]:
    """Build the message chain from a thread's history, limited by token budget."""
    warnings: set[str] = set()
    thread_msgs: list[discord.Message] = []

    try:
        if new_msg.channel.type == discord.ChannelType.public_thread and new_msg.channel.parent.type == discord.ChannelType.text:
            starter = new_msg.channel.starter_message or await new_msg.channel.parent.fetch_message(new_msg.channel.id)
            if starter:
                thread_msgs.append(starter)
    except (discord.NotFound, discord.HTTPException):
        logging.exception("Error fetching thread starter message")

    async for msg in new_msg.channel.history(limit=MAX_MESSAGES_SANITY, oldest_first=True):
        if msg.type in (discord.MessageType.default, discord.MessageType.reply):
            thread_msgs.append(msg)

    # Build all message dicts
    all_built: list[tuple[dict, discord.Message]] = []
    for msg in thread_msgs:
        node = msg_nodes.setdefault(msg.id, MsgNode())
        async with node.lock:
            await populate_node(node, msg)
        result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
        warnings |= w
        if result:
            all_built.append((result, msg))

    # Trim oldest messages to fit token budget
    total_tokens = sum(message_token_estimate(m) for m, _ in all_built)
    original_count = len(all_built)
    while all_built and total_tokens > max_context_tokens:
        removed, _ = all_built.pop(0)
        total_tokens -= message_token_estimate(removed)

    if len(all_built) < original_count:
        warnings.add(f"⚠️ Trimmed to last {len(all_built)} messages (~{total_tokens:,} tokens)")

    messages = [m for m, _ in all_built]
    earliest = all_built[0][1].created_at if all_built else new_msg.created_at
    return messages, warnings, earliest


async def build_chain_dm(
    new_msg: discord.Message,
    max_text: int,
    max_images: int,
    max_context_tokens: int,
    nodes_needing_descriptions: list[MsgNode],
) -> tuple[list[dict], set[str], datetime]:
    """Build the message chain by walking the reply chain (DMs), limited by token budget."""
    messages: list[dict] = []
    warnings: set[str] = set()
    earliest = new_msg.created_at
    curr_msg = new_msg
    estimated_tokens = 0

    while curr_msg is not None and len(messages) < MAX_MESSAGES_SANITY:
        node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with node.lock:
            await populate_node(node, curr_msg)
            await resolve_parent_msg(node, curr_msg)

        result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
        warnings |= w
        if result:
            msg_tokens = message_token_estimate(result)
            if messages and estimated_tokens + msg_tokens > max_context_tokens:
                warnings.add(f"⚠️ Stopped at {len(messages)} messages (~{estimated_tokens:,} tokens)")
                break
            estimated_tokens += msg_tokens
            messages.append(result)
            earliest = curr_msg.created_at

        if node.fetch_parent_failed:
            warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

        curr_msg = node.parent_msg

    return messages, warnings, earliest


async def build_chain_channel(
    new_msg: discord.Message,
    cfg: dict,
    max_text: int,
    max_images: int,
    max_context_tokens: int,
    nodes_needing_descriptions: list[MsgNode],
) -> tuple[list[dict], set[str], datetime]:
    """Build the message chain from recent channel history with gap/token cutoffs."""
    messages: list[dict] = []
    warnings: set[str] = set()

    context_gap_minutes = cfg.get("context_gap_minutes", 10)

    recent_msgs: list[discord.Message] = []
    estimated_tokens = 0
    prev_msg_time = new_msg.created_at

    async for msg in new_msg.channel.history(limit=MAX_MESSAGES_SANITY, before=new_msg):
        if msg.type not in (discord.MessageType.default, discord.MessageType.reply):
            continue

        time_gap = (prev_msg_time - msg.created_at).total_seconds() / 60
        if time_gap > context_gap_minutes:
            break

        msg_tokens = estimate_tokens(msg.content)
        if estimated_tokens + msg_tokens > max_context_tokens:
            break

        estimated_tokens += msg_tokens
        recent_msgs.append(msg)
        prev_msg_time = msg.created_at

    for msg in reversed(recent_msgs):
        node = msg_nodes.setdefault(msg.id, MsgNode())
        async with node.lock:
            await populate_node(node, msg)
        result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
        warnings |= w
        if result:
            messages.append(result)

    # Always include the current message
    node = msg_nodes.setdefault(new_msg.id, MsgNode())
    async with node.lock:
        await populate_node(node, new_msg)
    result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
    warnings |= w
    if result:
        messages.append(result)

    earliest = recent_msgs[-1].created_at if recent_msgs else new_msg.created_at

    return messages, warnings, earliest


# ---------------------------------------------------------------------------
# Memory sweep trigger
# ---------------------------------------------------------------------------

async def check_and_run_memory_sweep(
    new_msg: discord.Message,
    openai_client: AsyncOpenAI,
    model: str,
    **api_kwargs,
) -> None:
    """Detect a session gap and run a memory sweep on the previous session if needed."""
    try:
        prev_msgs = [m async for m in new_msg.channel.history(limit=1, before=new_msg)]
        if prev_msgs:
            gap = (new_msg.created_at - prev_msgs[0].created_at).total_seconds()
            if gap >= SESSION_GAP_SECONDS:
                session_msgs = await collect_previous_session(new_msg.channel, new_msg, discord_bot.user)
                if session_msgs:
                    await run_memory_sweep(new_msg.channel, session_msgs, openai_client, model, **api_kwargs)
    except Exception:
        logging.exception("Error checking session gap")


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt(cfg: dict) -> str | None:
    """Build the system prompt with date/time and memory injected."""
    system_prompt = cfg.get("system_prompt")
    if not system_prompt:
        return None

    now = datetime.now().astimezone()
    system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

    memory_content = memory_store.load()
    if memory_content.strip():
        system_prompt += (
            "\n\nYou have a persistent memory system. Below are your current memories. "
            "Use them naturally — don't reference the system itself, don't announce what you remember unless it's relevant. "
            "Just let the context inform how you respond.\n\n"
            "If a memory seems outdated or contradicted by the current conversation, note it internally. "
            "It will be reviewed at the end of the session.\n\n"
            f"<memories>\n{memory_content}</memories>"
        )

    return system_prompt


# ---------------------------------------------------------------------------
# Response streaming
# ---------------------------------------------------------------------------

async def stream_response(
    new_msg: discord.Message,
    openai_client: AsyncOpenAI,
    openai_kwargs: dict,
    user_warnings: set[str],
    context_info: str,
    use_plain_responses: bool,
) -> tuple[list[discord.Message], str]:
    """Stream the LLM response, sending/editing Discord messages. Returns (response_msgs, full_text)."""
    global last_task_time

    curr_content = finish_reason = None
    response_msgs: list[discord.Message] = []
    response_contents: list[str] = []

    if use_plain_responses:
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

    active_channels.add(new_msg.channel.id)
    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason is not None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason is not None or msg_split_incoming
                    is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                            # Clear fields after the first message so subsequent splits don't repeat them
                            embed.clear_fields()
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

    active_channels.discard(new_msg.channel.id)
    return response_msgs, "".join(response_contents)


# ---------------------------------------------------------------------------
# Image description caching
# ---------------------------------------------------------------------------

async def cache_image_descriptions(
    nodes: list[MsgNode],
    openai_client: AsyncOpenAI,
    model: str,
    max_images: int,
    **api_kwargs,
) -> None:
    """Generate and cache text descriptions for image attachments."""
    for node in nodes:
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
                    extra_headers=api_kwargs.get("extra_headers"),
                    extra_query=api_kwargs.get("extra_query"),
                    extra_body=api_kwargs.get("extra_body"),
                )
                descriptions.append(desc_resp.choices[0].message.content.strip())
            except Exception:
                logging.exception("Error generating image description")
                descriptions.append("(image description unavailable)")
        node.image_descriptions = descriptions
        logging.info(f"Cached {len(descriptions)} image description(s) for message node")


# ---------------------------------------------------------------------------
# Node cache cleanup
# ---------------------------------------------------------------------------

async def cleanup_old_nodes() -> None:
    """Evict the oldest MsgNodes if the cache exceeds MAX_MESSAGE_NODES."""
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


# ---------------------------------------------------------------------------
# Discord commands and events
# ---------------------------------------------------------------------------

@discord_bot.tree.command(name="info", description="Estimate the number of tokens in the current chat context")
async def info_command(interaction: discord.Interaction) -> None:
    cfg = await asyncio.to_thread(get_config)
    context_gap_minutes = cfg.get("context_gap_minutes", 10)
    max_context_tokens = cfg.get("max_context_tokens", 10000)

    channel = interaction.channel
    estimated_tokens = 0
    msg_count = 0
    prev_msg_time = datetime.now(timezone.utc)
    earliest_time = prev_msg_time

    async for msg in channel.history(limit=MAX_MESSAGES_SANITY):
        if msg.type not in (discord.MessageType.default, discord.MessageType.reply):
            continue

        time_gap = (prev_msg_time - msg.created_at).total_seconds() / 60
        if msg_count > 0 and time_gap > context_gap_minutes:
            break

        msg_tokens = estimate_tokens(msg.content)
        if msg_count > 0 and estimated_tokens + msg_tokens > max_context_tokens:
            break

        estimated_tokens += msg_tokens
        msg_count += 1
        earliest_time = msg.created_at
        prev_msg_time = msg.created_at

    earliest_ts = int(earliest_time.timestamp())
    await interaction.response.send_message(
        f"**Context estimate:**\n"
        f"Messages: {msg_count}\n"
        f"Estimated tokens: ~{int(estimated_tokens):,}\n"
        f"Earliest message: <t:{earliest_ts}:R>\n"
        f"Max context tokens: {max_context_tokens:,}",
        ephemeral=True,
    )


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if interaction.user.id in config["permissions"]["users"]["admin_ids"]:
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
    is_dm = new_msg.channel.type == discord.ChannelType.private
    is_thread = new_msg.channel.type in (discord.ChannelType.public_thread, discord.ChannelType.private_thread)

    if new_msg.author.bot or new_msg.content.startswith("."):
        return

    is_mentioned = is_dm or is_thread or discord_bot.user in new_msg.mentions

    cfg = await asyncio.to_thread(get_config)

    if not check_permissions(new_msg, cfg):
        return

    if not is_mentioned and not await check_interjection(new_msg, cfg):
        return

    # --- Acquire per-channel lock to serialize responses ---
    lock = channel_locks.setdefault(new_msg.channel.id, asyncio.Lock())
    async with lock:
        # --- Provider / model setup ---
        provider_slash_model = curr_model
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
        provider_config = cfg["providers"][provider]

        openai_client = AsyncOpenAI(
            base_url=provider_config["base_url"],
            api_key=provider_config.get("api_key", "sk-no-key-required"),
        )

        model_parameters = cfg["models"].get(provider_slash_model, None)
        extra_headers = provider_config.get("extra_headers")
        extra_query = provider_config.get("extra_query")
        extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

        accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
        max_text = cfg.get("max_text", 100000)
        max_images = cfg.get("max_images", 5) if accept_images else 0
        max_context_tokens = cfg.get("max_context_tokens", 10000)

        # --- Build message chain ---
        nodes_needing_descriptions: list[MsgNode] = []

        if is_thread:
            messages, user_warnings, earliest_msg_time = await build_chain_thread(
                new_msg, max_text, max_images, max_context_tokens, nodes_needing_descriptions,
            )
        elif is_dm:
            messages, user_warnings, earliest_msg_time = await build_chain_dm(
                new_msg, max_text, max_images, max_context_tokens, nodes_needing_descriptions,
            )
        else:
            messages, user_warnings, earliest_msg_time = await build_chain_channel(
                new_msg, cfg, max_text, max_images, max_context_tokens, nodes_needing_descriptions,
            )

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

        # --- Memory sweep ---
        await check_and_run_memory_sweep(
            new_msg, openai_client, model,
            extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body,
        )

        # --- System prompt & message ordering ---
        ordered_messages = messages[::-1] if is_dm else messages
        if system_prompt := build_system_prompt(cfg):
            ordered_messages.insert(0, dict(role="system", content=system_prompt))

        # --- Stream response ---
        openai_kwargs = dict(model=model, messages=ordered_messages, stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

        earliest_timestamp = int(earliest_msg_time.timestamp())
        context_info = f"Earliest message: <t:{earliest_timestamp}:t>"
        use_plain_responses = cfg.get("use_plain_responses", False)

        response_msgs, full_response = await stream_response(
            new_msg, openai_client, openai_kwargs, user_warnings, context_info, use_plain_responses,
        )

        # --- Finalize response nodes ---
        for response_msg in response_msgs:
            msg_nodes[response_msg.id].text = full_response
            msg_nodes[response_msg.id].lock.release()

        # --- Cache image descriptions ---
        await cache_image_descriptions(
            nodes_needing_descriptions, openai_client, model, max_images,
            extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body,
        )

    # --- Cleanup old nodes ---
    await cleanup_old_nodes()


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
