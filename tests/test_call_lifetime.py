import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy


def state(channel):
    return SimpleNamespace(channel=channel, self_stream=False)


def idle_call(bot, *, fail_close=False):
    channel = SimpleNamespace(id=10, members=[])
    client = SimpleNamespace(disconnect=AsyncMock(), channel=channel)
    guild = SimpleNamespace(id=1, get_channel=lambda _: channel, voice_client=client)
    bot.get_guild = lambda _: guild
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1),
        close=AsyncMock(side_effect=RuntimeError("cleanup failure") if fail_close else None),
        invalidate=Mock(),
    )
    receiver = SimpleNamespace(close=AsyncMock())
    bot.voice_sessions[1] = (conversation, receiver, None)
    return channel, guild, conversation, receiver, client


async def test_alone_timeout_counts_bots_as_alone_and_cleans_up_even_if_one_close_fails(store):
    assert Settings().voice_alone_timeout == 60
    bot = BenteBot(
        Settings(health_port=0, voice_alone_timeout=0.04, native_viewer_enabled=True, native_viewer_id=123),
        store,
    )
    channel, _, conversation, receiver, client = idle_call(bot, fail_close=True)
    channel.members = [SimpleNamespace(id=999, bot=True), SimpleNamespace(id=123, bot=False)]
    try:
        bot.update_alone_timer(1)
        timer = bot._alone_tasks[1]
        await asyncio.sleep(0.01)
        client.disconnect.assert_not_awaited()
        await timer
        assert not bot.voice_sessions and not bot._alone_tasks
        conversation.close.assert_awaited_once()
        receiver.close.assert_awaited_once()
        client.disconnect.assert_awaited_once_with(force=True)
    finally:
        await bot.close()


async def test_rejoining_cancels_old_timer_and_next_departure_starts_full_new_interval(store):
    bot = BenteBot(Settings(health_port=0, voice_alone_timeout=0.04), store)
    channel, guild, _, _, client = idle_call(bot)
    human = SimpleNamespace(id=100, bot=False, guild=guild)
    try:
        bot.update_alone_timer(1)
        old = bot._alone_tasks[1]
        channel.members.append(human)
        await bot.on_voice_state_update(human, state(None), state(channel))
        await asyncio.gather(old, return_exceptions=True)
        assert old.cancelled() and not bot._alone_tasks
        await asyncio.sleep(0.05)
        client.disconnect.assert_not_awaited()
        channel.members.clear()
        await bot.on_voice_state_update(human, state(channel), state(None))
        timer = bot._alone_tasks[1]
        assert timer is not old
        await timer
        client.disconnect.assert_awaited_once()
    finally:
        await bot.close()


@pytest.mark.parametrize("change", ["human_arrived", "call_replaced", "shutdown"])
async def test_expired_timer_never_disconnects_new_occupants_or_a_replacement_call(store, change):
    bot = BenteBot(Settings(health_port=0, voice_alone_timeout=0.03), store)
    channel, _, conversation, _, client = idle_call(bot)
    try:
        bot.update_alone_timer(1)
        timer = bot._alone_tasks[1]
        if change == "human_arrived":
            channel.members.append(SimpleNamespace(id=100, bot=False))  # Cache changed before event handler.
        elif change == "call_replaced":
            replacement = SimpleNamespace(source=Source("channel", 10, 1), close=AsyncMock())
            bot.voice_sessions[1] = (replacement, SimpleNamespace(close=AsyncMock()), None)
        else:
            await bot.close()
            assert timer.cancelled()
            return
        await timer
        client.disconnect.assert_not_awaited()
        conversation.close.assert_not_awaited()
    finally:
        await bot.close()


@pytest.mark.parametrize("change", ["requester_left", "other_joined", "call_replaced", "left_before_goodbye"])
async def test_leave_finishes_if_membership_changes_during_goodbye(store, wav_bytes, change):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    channel = SimpleNamespace(id=10, members=[])
    member = SimpleNamespace(id=100, bot=False, voice=SimpleNamespace(channel=channel))
    other = SimpleNamespace(id=200, bot=False, voice=SimpleNamespace(channel=channel))
    client = SimpleNamespace(disconnect=AsyncMock())
    guild = SimpleNamespace(
        id=1,
        voice_client=client,
        get_channel=lambda _: channel,
        get_member=lambda ident: member if ident == 100 else other if ident == 200 else None,
    )
    member.guild = other.guild = guild
    channel.members[:] = [member]
    bot.get_guild = lambda _: guild
    entered = asyncio.Event()
    finish_audio = asyncio.Event()

    async def play(pcm, text, progress):
        entered.set()
        await finish_audio.wait()

    conversation = VoiceConversation(
        bot.settings,
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 8000),
        bot.chat,
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        Source("channel", 10, 1),
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        play,
        Mock(),
        AsyncMock(),
        control_feedback=AsyncMock(return_value="See you later."),
    )
    receiver = SimpleNamespace(close=AsyncMock())
    bot.voice_sessions[1] = (conversation, receiver, None)
    try:
        assert await bot.handle_spoken_control(1, 10, 100, "Leave the call")
        jobs = list(bot.voice_jobs)
        async with asyncio.timeout(2):
            if change != "left_before_goodbye":
                await entered.wait()
            if change in {"requester_left", "left_before_goodbye"}:
                member.voice.channel = None
                channel.members.clear()
                await bot.on_voice_state_update(member, state(channel), state(None))
            elif change == "other_joined":
                channel.members.append(other)
                await bot.on_voice_state_update(other, state(None), state(channel))
                finish_audio.set()
            else:
                replacement = SimpleNamespace(source=Source("channel", 11, 1), close=AsyncMock())
                bot.voice_sessions[1] = (replacement, SimpleNamespace(close=AsyncMock()), None)
                bot._voice_membership_versions[1] = 1
                conversation.invalidate()
            await asyncio.gather(*jobs)
        if change == "call_replaced":
            client.disconnect.assert_not_awaited()
            assert bot.voice_sessions[1][0] is replacement
            await conversation.close()
            return
        client.disconnect.assert_awaited_once_with(force=True)
        receiver.close.assert_awaited_once()
        assert not bot.voice_sessions and not bot._alone_tasks
        assert conversation.metrics.last_control_result == "completed"
    finally:
        await bot.close()
