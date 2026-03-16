"""Semantic memory system with embedding-based retrieval.

Two-file memory:
- core_memory.md: always-injected identity facts, rewritten by sweep
- memory.json: everything else, retrieved by cosine similarity
"""

import json
import logging
import math
import re
import uuid
from datetime import date
from pathlib import Path

from openai import AsyncOpenAI


MEMORY_DIR = Path(__file__).parent
CORE_MEMORY_FILE = MEMORY_DIR / "core_memory.md"
MEMORY_JSON_FILE = MEMORY_DIR / "memory.json"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def get_embedding(text: str, client: AsyncOpenAI, model: str) -> list[float]:
    """Get an embedding vector for text using an OpenAI-compatible API."""
    resp = await client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Memory JSON operations
# ---------------------------------------------------------------------------

def load_memories() -> list[dict]:
    """Load all memory entries from memory.json."""
    try:
        data = json.loads(MEMORY_JSON_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_memories(memories: list[dict]) -> None:
    """Write all memory entries to memory.json."""
    MEMORY_JSON_FILE.write_text(
        json.dumps(memories, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_active_memories() -> list[dict]:
    """Return all non-deleted memories."""
    return [m for m in load_memories() if not m.get("deleted", False)]


# ---------------------------------------------------------------------------
# Core memory (always injected)
# ---------------------------------------------------------------------------

def load_core_memory() -> str:
    try:
        return CORE_MEMORY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def save_core_memory(content: str) -> None:
    CORE_MEMORY_FILE.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Operation functions (called by sweep executor)
# ---------------------------------------------------------------------------

async def add_memory(
    text: str, client: AsyncOpenAI, embedding_model: str,
) -> str:
    """Add a new memory entry. Returns the new ID."""
    memory_id = uuid.uuid4().hex[:8]
    vector = await get_embedding(text, client, embedding_model)

    memories = load_memories()
    memories.append({
        "id": memory_id,
        "text": text,
        "vector": vector,
        "created": date.today().isoformat(),
        "deleted": False,
    })
    save_memories(memories)

    logging.info(f"Added memory {memory_id}: {text[:60]}")
    return memory_id


async def update_memory(
    memory_id: str, new_text: str, client: AsyncOpenAI, embedding_model: str,
) -> bool:
    """Update an existing memory's text and re-embed. Returns True if found."""
    memories = load_memories()
    for entry in memories:
        if entry["id"] == memory_id:
            entry["text"] = new_text
            entry["vector"] = await get_embedding(new_text, client, embedding_model)
            save_memories(memories)
            logging.info(f"Updated memory {memory_id}: {new_text[:60]}")
            return True
    logging.warning(f"Memory {memory_id} not found for update")
    return False


def delete_memory(memory_id: str) -> bool:
    """Soft-delete a memory. Returns True if found."""
    memories = load_memories()
    for entry in memories:
        if entry["id"] == memory_id:
            entry["deleted"] = True
            save_memories(memories)
            logging.info(f"Deleted memory {memory_id}")
            return True
    logging.warning(f"Memory {memory_id} not found for deletion")
    return False


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve_memories(
    query: str,
    client: AsyncOpenAI,
    embedding_model: str,
    top_k: int = 3,
    threshold: float = 0.4,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Retrieve the top-k most similar non-deleted memories for a query.

    Returns list of dicts with 'id', 'text', 'score'.
    """
    active = get_active_memories()
    if not active:
        return []

    query_vector = await get_embedding(query, client, embedding_model)
    exclude = exclude_ids or set()

    scored = []
    for entry in active:
        if entry["id"] in exclude:
            continue
        score = cosine_similarity(query_vector, entry["vector"])
        if score >= threshold:
            scored.append({"id": entry["id"], "text": entry["text"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Sweep output parser
# ---------------------------------------------------------------------------

async def execute_sweep_operations(
    sweep_output: str, client: AsyncOpenAI, embedding_model: str,
) -> str:
    """Parse sweep LLM output and execute core memory + operations.

    Expected format:
        === CORE MEMORY ===
        - fact 1
        - fact 2

        === OPERATIONS ===
        ADD: "some new memory"
        UPDATE: abc12345 -> "updated text"
        DELETE: def67890

        SUMMARY: what changed

    Returns the summary line.
    """
    # Split sections
    core_match = re.search(
        r"=== CORE MEMORY ===\s*\n(.*?)(?==== OPERATIONS ===|\Z)",
        sweep_output,
        re.DOTALL,
    )
    ops_match = re.search(
        r"=== OPERATIONS ===\s*\n(.*?)(?=\nSUMMARY:|\Z)",
        sweep_output,
        re.DOTALL,
    )
    summary_match = re.search(r"SUMMARY:\s*(.+)", sweep_output)

    # Update core memory
    if core_match:
        core_content = core_match.group(1).strip()
        if core_content:
            save_core_memory(core_content + "\n")
            logging.info("Core memory updated")

    # Execute operations
    if ops_match:
        for line in ops_match.group(1).strip().splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                if line.startswith("ADD:"):
                    text = line[4:].strip().strip('"')
                    await add_memory(text, client, embedding_model)

                elif line.startswith("UPDATE:"):
                    rest = line[7:].strip()
                    match = re.match(r"(\w+)\s*->\s*\"(.+)\"", rest)
                    if match:
                        await update_memory(match.group(1), match.group(2), client, embedding_model)
                    else:
                        logging.warning(f"Could not parse UPDATE line: {line}")

                elif line.startswith("DELETE:"):
                    memory_id = line[7:].strip()
                    delete_memory(memory_id)

                else:
                    logging.warning(f"Unknown operation line: {line}")

            except Exception:
                logging.exception(f"Failed to execute operation: {line}")

    summary = summary_match.group(1).strip() if summary_match else "no summary provided"
    return summary
