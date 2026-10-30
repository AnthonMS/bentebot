from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.domain import Source


async def test_voice_channel_text_has_a_broader_audience_than_its_call(store):
    bot = BenteBot(Settings(health_port=0), store)
    bot._connection.user = SimpleNamespace(id=999)
    member, outsider, other_bot = (SimpleNamespace(id=id_) for id_ in (100, 200, 300))
    guild = SimpleNamespace(id=1, members=[member, outsider, other_bot, bot.user])
    channel = MagicMock(spec=discord.VoiceChannel)
    channel.id, channel.guild, channel.members = 10, guild, [member]
    bot.policy.member_can_access = AsyncMock(return_value=True)
    try:
        audible = await bot.audience_for(channel, voice=True)
        visible = await bot.audience_for(channel)
        assert audible.user_ids == frozenset({100}) and audible.destination_id is None
        assert visible.user_ids == frozenset({100, 200, 300}) and visible.destination_id == 10
        private = Source("dm", 50, owner_id=100)
        assert await bot.policy.allows(private, audible)
        assert not await bot.policy.allows(private, visible)
        channel.members = [member, other_bot]
        assert not await bot.policy.allows(private, await bot.audience_for(channel, voice=True))
    finally:
        await bot.close()
