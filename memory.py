import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import discord
from openai import AsyncOpenAI

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


# ---------------------------------------------------------------------------
# Memory sweep
# ---------------------------------------------------------------------------

async def run_memory_sweep(
    channel: discord.abc.Messageable,
    session_msgs: list[dict[str, str]],
    openai_client: AsyncOpenAI,
    model: str,
    **api_kwargs,
) -> None:
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


# ---------------------------------------------------------------------------
# Memory sweep trigger
# ---------------------------------------------------------------------------

async def check_and_run_memory_sweep(
    new_msg: discord.Message,
    bot_user: discord.User,
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
                session_msgs = await collect_previous_session(new_msg.channel, new_msg, bot_user)
                if session_msgs:
                    await run_memory_sweep(new_msg.channel, session_msgs, openai_client, model, **api_kwargs)
    except Exception:
        logging.exception("Error checking session gap")
