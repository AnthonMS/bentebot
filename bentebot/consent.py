"""Discord confirms another member's explicit consent before sampling their account audio."""

from __future__ import annotations

import discord


class VoiceConsent(discord.ui.View):
    def __init__(self, target_id: int, start):
        super().__init__(timeout=120)
        self.target_id, self.start_capture = target_id, start
        self.consumed = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.target_id:
            await interaction.response.send_message(
                "Only the selected speaker can consent or decline.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Allow voice sample", style=discord.ButtonStyle.success)
    async def allow(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.consumed:
            await interaction.response.send_message(
                "This consent request has already been answered.", ephemeral=True
            )
            return
        self.consumed = True
        await interaction.response.defer(ephemeral=True)
        # Consume the view before awaiting network/membership checks; a second click cannot start another capture.
        self.stop()
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)
        try:
            await self.start_capture(interaction)
        except (ValueError, RuntimeError) as exc:
            await interaction.followup.send(str(exc), ephemeral=True)

    @discord.ui.button(label="Decline", style=discord.ButtonStyle.secondary)
    async def decline(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.consumed:
            await interaction.response.send_message(
                "This consent request has already been answered.", ephemeral=True
            )
            return
        self.consumed = True
        self.stop()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(
            content="Voice sampling declined. No audio was collected.", view=self
        )
