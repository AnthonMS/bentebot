"""Discord editing of bounded server/channel persona rules."""

from __future__ import annotations

from typing import Literal

import discord
from discord import app_commands

from .text import discord_chunks


def install_persona(bot):
    group = app_commands.Group(name="persona", description="View and edit Bentebot's personality")

    def authorize(interaction):
        if not interaction.guild_id or not bot.is_admin(interaction.user):
            raise app_commands.CheckFailure("Persona editing requires Manage Server or bot ownership.")

    def target(interaction, scope, channel):
        authorize(interaction)
        resolved = bot.resolve_model_channel(interaction, channel)
        if scope == "server":
            if resolved is not None:
                raise ValueError("Use channel scope with a channel, or omit channel for the server default.")
            return None
        return bot.model_target(interaction, resolved)

    async def report(interaction, data, scope):
        text = (
            f"Persona: {scope}; server revision {data['server_revision']}; selected scope revision {data['revision']}\n"
            + "\n".join(
                f"{r['id']}. {r['text']}" + (" [channel]" if r["id"] in data["channel_rule_ids"] else "")
                for r in data["rules"]
            )
        )
        text += "\nStyle rules apply to new requests. Access and tool permissions are separate."
        text += "\nServer edits update the same rule in every channel; channel edits stay local. Reset channel to inherit the server, or reset server and all channels to built-ins."
        for index, chunk in enumerate(discord_chunks(text)):
            send = (
                interaction.followup.send
                if index or interaction.response.is_done()
                else interaction.response.send_message
            )
            await send(chunk, ephemeral=True)

    async def channel_choices(interaction, current):
        authorize(interaction)
        return [
            app_commands.Choice(name=c.name[:100], value=str(c.id))
            for c in bot.model_channels(interaction)
            if current.casefold() in c.name.casefold()
        ][:25]

    @group.command(name="show", description="Show the effective persona and stable rule IDs")
    @app_commands.autocomplete(channel=channel_choices)
    async def show(interaction: discord.Interaction, channel: str | None = None):
        authorize(interaction)
        selected = bot.resolve_model_channel(interaction, channel)
        channel_id = bot.model_target(interaction, selected)
        await report(interaction, *bot.memory.personas.effective(interaction.guild_id, channel_id))

    @group.command(name="add", description="Add a personality rule")
    @app_commands.autocomplete(channel=channel_choices)
    async def add(
        interaction: discord.Interaction,
        text: app_commands.Range[str, 1, 700],
        scope: Literal["server", "channel"] = "server",
        channel: str | None = None,
    ):
        channel_id = target(interaction, scope, channel)
        await report(
            interaction,
            *bot.memory.personas.change(interaction.guild_id, channel_id, "add", text=text),
        )

    @group.command(name="remove", description="Remove a persona rule by its displayed ID")
    @app_commands.autocomplete(channel=channel_choices)
    async def remove(
        interaction: discord.Interaction,
        rule_id: int,
        scope: Literal["server", "channel"] = "server",
        channel: str | None = None,
    ):
        channel_id = target(interaction, scope, channel)
        await report(
            interaction,
            *bot.memory.personas.change(interaction.guild_id, channel_id, "remove", rule_id=rule_id),
        )

    class RuleEditor(discord.ui.Modal, title="Edit Bentebot persona"):
        text = discord.ui.TextInput(label="Rule", style=discord.TextStyle.paragraph, max_length=700)

        def __init__(self, guild_id, channel_id, rule, revision, user_id):
            super().__init__(timeout=600)
            self.guild_id, self.channel_id = guild_id, channel_id
            self.rule_id, self.revision, self.user_id = rule["id"], revision, user_id
            self.text.default = rule["text"]

        async def on_submit(self, interaction):
            authorize(interaction)
            if interaction.user.id != self.user_id or interaction.guild_id != self.guild_id:
                raise app_commands.CheckFailure("This persona editor belongs to another request.")
            if self.channel_id is not None:
                bot.resolve_model_channel(interaction, str(self.channel_id))
            await report(
                interaction,
                *bot.memory.personas.change(
                    self.guild_id,
                    self.channel_id,
                    "edit",
                    text=str(self.text),
                    rule_id=self.rule_id,
                    revision=self.revision,
                ),
            )

        async def on_error(self, interaction, error):
            message = (
                str(error)
                if isinstance(error, (ValueError, app_commands.CheckFailure))
                else "Persona edit failed."
            )
            await interaction.response.send_message(message, ephemeral=True)

    @group.command(name="edit", description="Edit one persona rule in a text form")
    @app_commands.autocomplete(channel=channel_choices)
    async def edit(
        interaction: discord.Interaction,
        rule_id: int,
        scope: Literal["server", "channel"] = "server",
        channel: str | None = None,
    ):
        channel_id = target(interaction, scope, channel)
        data, _ = bot.memory.personas.effective(interaction.guild_id, channel_id)
        rule = next((r for r in data["rules"] if r["id"] == rule_id), None)
        if rule is None:
            raise ValueError("No persona rule has that ID in the selected scope.")
        await interaction.response.send_modal(
            RuleEditor(interaction.guild_id, channel_id, rule, data["edit_token"], interaction.user.id)
        )

    @group.command(
        name="reset", description="Reset channel to server, or reset server and all channels to built-ins"
    )
    @app_commands.autocomplete(channel=channel_choices)
    async def reset(
        interaction: discord.Interaction,
        scope: Literal["server", "channel"] = "server",
        channel: str | None = None,
    ):
        channel_id = target(interaction, scope, channel)
        await report(interaction, *bot.memory.personas.change(interaction.guild_id, channel_id, "reset"))

    @group.command(name="undo", description="Undo the latest persona edit in this scope")
    @app_commands.autocomplete(channel=channel_choices)
    async def undo(
        interaction: discord.Interaction,
        scope: Literal["server", "channel"] = "server",
        channel: str | None = None,
    ):
        channel_id = target(interaction, scope, channel)
        await report(interaction, *bot.memory.personas.change(interaction.guild_id, channel_id, "undo"))

    bot.tree.add_command(group)
