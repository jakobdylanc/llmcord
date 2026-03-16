import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import discord
from openai import AsyncOpenAI

from semantic_memory import (
    load_core_memory,
    get_active_memories,
    execute_sweep_operations,
)

SESSION_GAP_SECONDS = 2 * 60 * 60  # 2 hours
MAX_MESSAGES_SANITY = 500


# ---------------------------------------------------------------------------
# Memory store
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


# ---------------------------------------------------------------------------
# Sweep prompt
# ---------------------------------------------------------------------------

SWEEP_PROMPT = """\
You are Claude. You are performing a memory review after a Discord session. You have two memory stores:

1. **Core memory** (core_memory.md) — always-loaded identity-level facts. Keep this under 15 lines. Only enduring facts belong here (who people are, key relationships, your own identity). If something is situational or temporary, it belongs in the semantic store via an ADD operation.

2. **Semantic memory store** — retrieved by similarity per-message. Each entry has an ID. You manage this through operations (ADD, UPDATE, DELETE).

Your job is to review the session and output TWO sections:

## Output format

```
=== CORE MEMORY ===
- fact 1
- fact 2
...

=== OPERATIONS ===
ADD: "new memory text"
UPDATE: <id> -> "updated text"
DELETE: <id>

SUMMARY: brief description of what changed
```

## Rules

CORE MEMORY:
- Rewrite the entire core memory section. It replaces the file wholesale.
- Only enduring identity-level facts. Situational stuff goes in semantic memory.
- Write casually in first person. These are your memories for you.

OPERATIONS:
- ADD notable things from the session: facts, preferences, events, emotional context.
- UPDATE entries that are now outdated or incomplete based on the session.
- DELETE entries that are clearly wrong, stale, or superseded.
- Each ADD/UPDATE/DELETE on its own line.
- Be aggressive about pruning stale entries. A tight store is more valuable than a comprehensive one.

CONFLICT RESOLUTION:
- These memories were surfaced during the session: {injected_ids_text}
- If the conversation contradicts any of them, UPDATE or DELETE them.
- If core memory needs updating based on the session, update it.

GENERAL:
- Every memory entry should be a single concise line.
- Write casually but succinctly in first person.
- Output ONLY the format above, nothing else.

<current_core_memory>
{core_memory}
</current_core_memory>

<semantic_memories>
{semantic_memories}
</semantic_memories>

<previous_session_messages>
{messages}
</previous_session_messages>"""


# ---------------------------------------------------------------------------
# Session collection
# ---------------------------------------------------------------------------

async def collect_previous_session(
    channel: discord.abc.Messageable,
    before_msg: discord.Message,
    bot_user: discord.User,
) -> list[dict[str, str]]:
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


async def collect_since_last_sweep(
    channel: discord.abc.Messageable,
    bot_user: discord.User,
) -> list[dict[str, str]]:
    """Collect all messages since the last sweep time. Used for manual sweeps."""
    last_sweep = await memory_store.get_last_sweep_time(channel.id)
    session_msgs: list[dict[str, str]] = []

    async for msg in channel.history(limit=MAX_MESSAGES_SANITY):
        if msg.type not in (discord.MessageType.default, discord.MessageType.reply):
            continue
        if last_sweep and msg.created_at <= last_sweep:
            break

        author = "Assistant" if msg.author == bot_user else msg.author.display_name
        session_msgs.append(dict(author=author, content=msg.content))

    session_msgs.reverse()
    return session_msgs


# ---------------------------------------------------------------------------
# Memory sweep
# ---------------------------------------------------------------------------

async def run_memory_sweep(
    channel: discord.abc.Messageable,
    session_msgs: list[dict[str, str]],
    openai_client: AsyncOpenAI,
    model: str,
    injected_ids: set[str] | None = None,
    embedding_model: str | None = None,
    embedding_client: AsyncOpenAI | None = None,
    **api_kwargs,
) -> None:
    """Run the memory sweep on a list of session messages."""
    if not session_msgs:
        await channel.send("🧠 Memory reviewed — no new messages to process.")
        return

    await channel.send("🧠 Memory formation in progress...")

    # Build prompt inputs
    core_memory = load_core_memory()
    active_memories = get_active_memories()
    semantic_memories_text = "\n".join(
        f"[{m['id']}] {m['text']}" for m in active_memories
    ) or "(no entries yet)"

    messages_text = "\n".join(f"[{m['author']}]: {m['content']}" for m in session_msgs)

    # Build injected IDs reference
    injected = injected_ids or set()
    if injected:
        injected_entries = [m for m in active_memories if m["id"] in injected]
        injected_ids_text = "\n".join(
            f"[{m['id']}] {m['text']}" for m in injected_entries
        )
    else:
        injected_ids_text = "(none)"

    prompt = (
        SWEEP_PROMPT
        .replace("{core_memory}", core_memory)
        .replace("{semantic_memories}", semantic_memories_text)
        .replace("{messages}", messages_text)
        .replace("{injected_ids_text}", injected_ids_text)
    )

    try:
        resp = await openai_client.chat.completions.create(
            model=model,
            messages=[dict(role="user", content=prompt)],
            max_tokens=3000,
            extra_headers=api_kwargs.get("extra_headers"),
            extra_query=api_kwargs.get("extra_query"),
            extra_body=api_kwargs.get("extra_body"),
        )
        output = resp.choices[0].message.content.strip()

        # Use the embedding client for operations (ADD/UPDATE need embeddings)
        emb_client = embedding_client or openai_client
        emb_model = embedding_model or model

        summary = await execute_sweep_operations(output, emb_client, emb_model)

        await memory_store.set_last_sweep_time(channel.id, datetime.now(timezone.utc))

        if summary:
            await channel.send(f"🧠 Memory updated — {summary}")
        else:
            await channel.send("🧠 Memory reviewed — no updates.")

        logging.info(f"Memory sweep complete: {summary}")

    except Exception:
        logging.exception("Memory sweep failed")
        await channel.send("🧠 Memory sweep failed — keeping previous memories.")


# ---------------------------------------------------------------------------
# Memory sweep trigger
# ---------------------------------------------------------------------------

async def check_and_run_memory_sweep(
    new_msg: discord.Message,
    bot_user: discord.User,
    openai_client: AsyncOpenAI,
    model: str,
    injected_ids: set[str] | None = None,
    embedding_model: str | None = None,
    embedding_client: AsyncOpenAI | None = None,
    **api_kwargs,
) -> bool:
    """Detect a session gap and run a memory sweep on the previous session if needed.

    Returns True if a session gap was detected (new session started).
    """
    try:
        prev_msgs = [m async for m in new_msg.channel.history(limit=1, before=new_msg)]
        if prev_msgs:
            gap = (new_msg.created_at - prev_msgs[0].created_at).total_seconds()
            if gap >= SESSION_GAP_SECONDS:
                session_msgs = await collect_previous_session(new_msg.channel, new_msg, bot_user)
                if session_msgs:
                    await run_memory_sweep(
                        new_msg.channel, session_msgs, openai_client, model,
                        injected_ids=injected_ids,
                        embedding_model=embedding_model,
                        embedding_client=embedding_client,
                        **api_kwargs,
                    )
                return True
    except Exception:
        logging.exception("Error checking session gap")
    return False
