from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord

from bentebot.discord_app import DiscordAccess
from bentebot.domain import Audience, Source


def permissions(view=True, history=True, manage=False):
    return SimpleNamespace(view_channel=view, read_message_history=history, manage_threads=manage)


def setup_access(*, chunked=True, thread=False):
    channel = MagicMock(spec=discord.Thread if thread else discord.TextChannel)
    channel.id = 10
    channel.permissions_for.return_value = permissions()
    if thread:
        channel.is_private.return_value = True
        channel.fetch_members = AsyncMock(return_value=[SimpleNamespace(id=id_) for id_ in (100, 200, 999)])
    members = {100: SimpleNamespace(id=100), 200: SimpleNamespace(id=200), 999: SimpleNamespace(id=999)}
    guild = SimpleNamespace(
        id=1,
        chunked=chunked,
        get_member=members.get,
        members=list(members.values()),
        get_channel_or_thread=lambda id: channel if id == 10 else None,
    )
    bot = SimpleNamespace(
        get_guild=lambda id: guild if id == 1 else None,
        user=SimpleNamespace(id=999),
        settings=SimpleNamespace(guild_ids=frozenset({1})),
    )
    return DiscordAccess(bot), channel, guild


async def test_member_overrides_are_applied_for_every_voice_recipient():
    access, channel, _ = setup_access()
    channel.permissions_for.side_effect = lambda member: permissions(view=member.id != 200)
    source = Source("channel", 10, 1)
    assert await access.allows(source, Audience(frozenset({100}), 1))
    assert not await access.allows(source, Audience(frozenset({100, 200}), 1))


async def test_read_history_and_bot_access_required():
    access, channel, _ = setup_access()
    channel.permissions_for.side_effect = lambda member: permissions(history=member.id != 999)
    assert not await access.allows(Source("channel", 10, 1), Audience(frozenset({100}), 1))


async def test_private_thread_membership_is_fetched_and_denied_on_unknown():
    access, channel, _ = setup_access(thread=True)
    source, audience = Source("channel", 10, 1), Audience(frozenset({100}), 1)
    assert await access.allows(source, audience)
    assert channel.fetch_members.await_count == 1
    response = SimpleNamespace(status=404, reason="Not Found")
    channel.fetch_members.side_effect = discord.NotFound(
        response, {"message": "unknown member", "code": 10007}
    )
    assert not await access.allows(source, audience)


async def test_complete_text_audience_is_required_for_cross_channel_access():
    access, _, _ = setup_access(chunked=False)
    source = Source("channel", 10, 1)
    assert not await access.allows(source, Audience(frozenset({100}), 1, 20))
    assert not await access.allows(source, Audience(frozenset({100}), 1, 10))


async def test_missing_source_member_and_other_server_fail_closed():
    access, _, _ = setup_access()
    assert not await access.allows(Source("channel", 11, 1), Audience(frozenset({100}), 1))
    assert not await access.allows(Source("channel", 10, 1), Audience(frozenset({300}), 1))
    assert not await access.allows(Source("channel", 10, 2), Audience(frozenset({100}), 2))


async def test_live_viewer_without_history_is_still_a_recipient():
    access, channel, _ = setup_access()
    channel.permissions_for.side_effect = lambda member: permissions(history=member.id != 200)
    source = Source("channel", 10, 1)
    assert await access.member_can_access(200, source, history=False)
    assert not await access.allows(source, Audience(frozenset({100, 200}), 1, 10))


async def test_dm_context_can_be_posted_to_verified_owner_only_bot_channel():
    access, channel, _ = setup_access()
    channel.permissions_for.side_effect = lambda member: permissions(view=member.id in {100, 999})
    assert await access.allows(Source("dm", 20, owner_id=100), Audience(frozenset({100}), 1, 10))


async def test_claimed_owner_only_audience_cannot_hide_another_actual_reader():
    access, _, _ = setup_access()
    assert not await access.allows(Source("dm", 20, owner_id=100), Audience(frozenset({100}), 1, 10))


async def test_dm_context_rejects_different_owner_and_incomplete_destination_cache():
    access, channel, guild = setup_access()
    channel.permissions_for.side_effect = lambda member: permissions(view=member.id in {100, 999})
    assert not await access.allows(Source("dm", 20, owner_id=200), Audience(frozenset({100}), 1, 10))
    guild.chunked = False
    assert not await access.allows(Source("dm", 20, owner_id=100), Audience(frozenset({100}), 1, 10))


async def test_owner_only_dm_destination_requires_bot_and_owner_access():
    access, channel, _ = setup_access()
    for denied in (100, 999):
        channel.permissions_for.side_effect = lambda member, denied=denied: permissions(
            view=member.id in {100, 999} and member.id != denied
        )
        assert not await access.allows(Source("dm", 20, owner_id=100), Audience(frozenset({100}), 1, 10))
