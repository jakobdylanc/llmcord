"""
Alternate entrypoint that simply delegates to the existing legacy script.

This keeps backward compatibility while allowing `python -m bot.main`
to run the bot.
"""

import asyncio
from typing import Any

from llmcord import main as legacy_main


async def run_bot(config: dict[str, Any] | None = None) -> None:  # config kept for future use
    await legacy_main()


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


