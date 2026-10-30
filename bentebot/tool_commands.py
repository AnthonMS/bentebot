"""Explicit utility access when a model does not choose a tool itself."""

from __future__ import annotations

import json
from datetime import datetime

import discord
from discord import app_commands

from .domain import Source
from .text import discord_chunks
from .tools import TOOL_GUARD
from .utility_tools import TOOL_REQUEST, ToolRequest


def format_result(name, arguments, result):
    if "error" in result:
        return result["error"] + ("\n" + result["hint"] if result.get("hint") else "")
    evidence = result["untrusted_evidence"]
    if name == "calculate":
        return f"{arguments['expression']} = {evidence['value']}"
    if name == "convert_units":
        return (
            f"{arguments['value']:g} {arguments['from_unit']} = {evidence['value']:.15g} {evidence['unit']}"
        )
    if name == "current_time":
        moment = datetime.fromisoformat(evidence["iso_time"])
        return f"{moment:%Y-%m-%d %H:%M:%S %z} ({evidence['timezone']})"
    output = evidence.get("output", "").strip()
    if not output:
        return "Python completed." if evidence["ok"] else "Python execution failed."
    return "```text\n" + output.replace("```", "'''") + "\n```"


def install_tools(bot):
    group = app_commands.Group(name="tools", description="Calculations and owner-only Python analysis")

    @group.command(name="status", description="Show available tools without making an AI request")
    async def status(interaction: discord.Interaction):
        request = TOOL_REQUEST.set(ToolRequest(interaction.user.id, allow_python=True))
        try:
            names = [s["function"]["name"] for s in bot.chat.utilities.schemas()]
            await interaction.response.send_message(
                "Utilities: "
                + (", ".join(names) or "disabled")
                + "\nPython is owner-only. Each execution gets a fresh kernel. "
                "The shared Jupyter server determines filesystem and network access.",
                ephemeral=True,
            )
        finally:
            TOOL_REQUEST.reset(request)

    async def execute(interaction, name, arguments):
        if name == "execute_python" and interaction.user.id not in bot.settings.owner_ids:
            raise app_commands.CheckFailure("Python execution is restricted to configured bot owners.")
        await interaction.response.defer(ephemeral=True)
        source = (
            Source("channel", interaction.channel_id, interaction.guild_id)
            if interaction.guild_id
            else Source("dm", interaction.channel_id, owner_id=interaction.user.id)
        )
        recipients = await bot.audience_for(interaction.channel, owner_id=interaction.user.id)

        async def allowed():
            return await bot.audience_for(
                interaction.channel, owner_id=interaction.user.id
            ) == recipients and await bot.policy.allows(source, recipients)

        context = ToolRequest(interaction.user.id, allow_python=True)
        request_token, guard_token = TOOL_REQUEST.set(context), TOOL_GUARD.set(allowed)
        try:
            result = json.loads(await bot.chat.utilities.execute(name, json.dumps(arguments)))
            if not await allowed():
                raise app_commands.CheckFailure("The destination audience changed during execution.")
            text = format_result(name, arguments, result)
            # Explicit command output is ephemeral, including generated plot files.
            for chunk in discord_chunks(text):
                if not await allowed():
                    break
                await interaction.followup.send(
                    chunk, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
                )
            for artifact in context.artifacts:
                if not await allowed():
                    break
                import io

                await interaction.followup.send(
                    file=discord.File(io.BytesIO(artifact.data), filename=artifact.name), ephemeral=True
                )
        finally:
            TOOL_GUARD.reset(guard_token)
            TOOL_REQUEST.reset(request_token)

    @group.command(name="calculate", description="Calculate arithmetic without an AI request")
    async def calculate(interaction: discord.Interaction, expression: str):
        await execute(interaction, "calculate", {"expression": expression})

    @group.command(name="time", description="Get the current time in an IANA timezone")
    async def time(interaction: discord.Interaction, timezone: str = "Europe/Copenhagen"):
        await execute(interaction, "current_time", {"timezone": timezone})

    @group.command(name="convert", description="Convert units without an AI request")
    async def convert(interaction: discord.Interaction, value: float, from_unit: str, to_unit: str):
        await execute(
            interaction, "convert_units", {"value": value, "from_unit": from_unit, "to_unit": to_unit}
        )

    @group.command(name="python", description="Run owner-requested Python in a fresh Jupyter kernel")
    async def python(interaction: discord.Interaction, code: app_commands.Range[str, 1, 6000]):
        await execute(interaction, "execute_python", {"code": code})

    bot.tree.add_command(group)
