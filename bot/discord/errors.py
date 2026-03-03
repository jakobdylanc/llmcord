from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

import discord

from bot.llm.errors import parse_error_message


async def notify_admin_error(
    discord_bot: discord.Client,
    config: dict[str, Any],
    error: Exception,
    context: str = "",
) -> None:
    """
    Send a concise error notification to all configured admins.
    """
    try:
        admin_ids = config.get("permissions", {}).get("users", {}).get("admin_ids", [])
        if not admin_ids:
            return

        msg = (
            "🤖 **Bot Error Notification**\n"
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"📝 Context: {context}\n\nError: {parse_error_message(error)}"
        )
        for admin_id in admin_ids:
            try:
                user = discord_bot.get_user(admin_id) or await discord_bot.fetch_user(
                    admin_id
                )
                await user.send(msg)
            except Exception as e:  # noqa: BLE001
                logging.warning("Could not notify admin %s: %s", admin_id, e)
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to notify admins: %s", e)


async def handle_app_command_error(
    interaction: discord.Interaction,
    error: Exception,
    discord_bot: discord.Client,
    config: dict[str, Any],
) -> None:
    """
    Standard handler for slash command errors.
    """
    logging.exception("App command error: %s", error)
    await notify_admin_error(
        discord_bot,
        config,
        error,
        f"App command error: {getattr(interaction.command, 'name', 'unknown')}",
    )
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(
                "指令執行時發生錯誤，已通知管理員。", ephemeral=True
            )
        else:
            await interaction.followup.send(
                "指令執行時發生錯誤，已通知管理員。", ephemeral=True
            )
    except Exception:  # noqa: BLE001
        pass

