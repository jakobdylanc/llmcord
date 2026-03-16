"""One-time migration: memory.md -> core_memory.md + memory.json

Usage:
    python migrate_memory.py

Requires embedding_model and provider config in config.yaml, e.g.:
    embedding_model: openrouter/openai/text-embedding-3-small
"""

import asyncio
import sys
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from semantic_memory import (
    CORE_MEMORY_FILE,
    MEMORY_JSON_FILE,
    add_memory,
    save_core_memory,
    load_memories,
)

MEMORY_MD = Path(__file__).parent / "memory.md"


def parse_memory_md(text: str) -> tuple[list[str], list[str]]:
    """Parse memory.md into (identity_lines, other_lines).

    Identity lines come from sections named 'Identity' under any heading.
    Everything else goes into the retrieval store.
    """
    identity_lines: list[str] = []
    other_lines: list[str] = []

    in_identity = False

    for line in text.splitlines():
        stripped = line.strip()

        # Track section headers
        if stripped.startswith("# "):
            section = stripped.lstrip("# ").strip().lower()
            in_identity = section == "identity"
            continue

        # Skip top-level group headers (## General, ## Claude)
        if stripped.startswith("## "):
            continue

        # Skip blank lines
        if not stripped:
            continue

        # Route content
        if stripped.startswith("- "):
            if in_identity:
                identity_lines.append(stripped)
            else:
                other_lines.append(stripped[2:])  # strip the "- " prefix
        elif stripped:
            # Non-bullet content in a section
            if in_identity:
                identity_lines.append(stripped)
            else:
                other_lines.append(stripped)

    return identity_lines, other_lines


async def migrate(embedding_model: str, client: AsyncOpenAI) -> None:
    if not MEMORY_MD.exists():
        print(f"No {MEMORY_MD} found — nothing to migrate.")
        return

    if MEMORY_JSON_FILE.exists() and load_memories():
        print(f"{MEMORY_JSON_FILE} already has entries. Aborting to avoid duplicates.")
        print("Delete memory.json first if you want to re-run migration.")
        return

    text = MEMORY_MD.read_text(encoding="utf-8")
    identity_lines, other_lines = parse_memory_md(text)

    # Write core memory
    core_content = "\n".join(identity_lines) + "\n"
    save_core_memory(core_content)
    print(f"Wrote {len(identity_lines)} identity lines to {CORE_MEMORY_FILE}")

    # Embed and write each memory entry
    print(f"Embedding {len(other_lines)} memory entries...")
    for i, entry_text in enumerate(other_lines):
        memory_id = await add_memory(entry_text, client, embedding_model)
        print(f"  [{i+1}/{len(other_lines)}] {memory_id}: {entry_text[:70]}")

    print(f"\nMigration complete.")
    print(f"  Core memory: {CORE_MEMORY_FILE}")
    print(f"  Semantic memories: {MEMORY_JSON_FILE} ({len(other_lines)} entries)")
    print(f"  Original backup: {MEMORY_MD} (unchanged)")


def main():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    embedding_model_full = cfg.get("embedding_model")
    if not embedding_model_full:
        print("Error: 'embedding_model' not set in config.yaml")
        print("Example: embedding_model: openrouter/openai/text-embedding-3-small")
        sys.exit(1)

    # Split provider/model the same way llmcord does
    provider, model = embedding_model_full.split("/", 1)
    provider_config = cfg["providers"][provider]

    client = AsyncOpenAI(
        base_url=provider_config["base_url"],
        api_key=provider_config.get("api_key", "sk-no-key-required"),
    )

    asyncio.run(migrate(model, client))


if __name__ == "__main__":
    main()
