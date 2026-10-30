from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience, Event, Source, Transcript


@pytest.mark.parametrize("before_id,after_id,expected", [(20, 30, False), (10, 30, True), (30, 10, True)])
async def test_voice_moves_update_presence_without_cancelling_calls(store, before_id, after_id, expected):
    bot = BenteBot(Settings(health_port=0), store)
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1), member_left=Mock(), listener_joined=Mock()
    )
    bot.voice_sessions[1] = (conversation, None, None)
    bot.invalidate = Mock()
    bot.update_alone_timer = Mock()
    try:
        await bot.on_voice_state_update(
            SimpleNamespace(id=100, guild=SimpleNamespace(id=1)),
            SimpleNamespace(channel=SimpleNamespace(id=before_id)),
            SimpleNamespace(channel=SimpleNamespace(id=after_id)),
        )
        bot.invalidate.assert_not_called()
        assert bot.update_alone_timer.called is expected
        assert conversation.member_left.called is (before_id == 10)
        assert conversation.listener_joined.called is (after_id == 10)
    finally:
        bot.voice_sessions.clear()
        await bot.close()


def message(*, content="original", bot=False, guild_id=1, author_id=100):
    return SimpleNamespace(
        content=content,
        author=SimpleNamespace(id=author_id, bot=bot),
        guild=SimpleNamespace(id=guild_id) if guild_id is not None else None,
        channel=SimpleNamespace(id=10),
    )


@pytest.mark.parametrize(
    "before,after,disabled",
    [
        (message(), message(), False),  # Discord embed-only/unfurl update.
        (message(bot=True), message(bot=True, content="streaming update"), False),
        (message(guild_id=2), message(guild_id=2, content="edited"), False),
        (message(), message(content="edited"), True),
        (
            message(guild_id=None, author_id=200),
            message(guild_id=None, author_id=200, content="edited"),
            False,
        ),
    ],
)
async def test_irrelevant_edits_do_not_cancel_replies_or_forget_history(store, before, after, disabled):
    bot = BenteBot(Settings(guild_ids=frozenset({1}), dm_user_ids=frozenset({100}), health_port=0), store)
    store.save(Event(Source("channel", 10, 1), 100, "Me", "keep this context"))
    if disabled:
        store.set("channel:10", "collect", "false")
    bot.quiesce = AsyncMock()
    bot._remember = AsyncMock()
    try:
        await bot.on_message_edit(before, after)
        assert store.count() == 1
        bot.quiesce.assert_not_awaited()
        bot._remember.assert_not_awaited()
    finally:
        await bot.close()


@pytest.mark.parametrize("guild_id", [1, None])
async def test_actual_enabled_human_text_edit_invalidates_then_remembers(store, guild_id):
    bot = BenteBot(Settings(guild_ids=frozenset({1}), dm_user_ids=frozenset({100}), health_port=0), store)
    source = Source("channel", 10, 1) if guild_id else Source("dm", 10, owner_id=100)
    store.save(Event(source, 100, "Me", "old text"))
    bot.quiesce = AsyncMock()
    bot._remember = AsyncMock(return_value=True)
    after = message(guild_id=guild_id, content="edited")
    try:
        await bot.on_message_edit(message(guild_id=guild_id), after)
        bot.quiesce.assert_awaited_once_with(guild_id)
        assert store.count() == 0
        bot._remember.assert_awaited_once_with(after)
    finally:
        await bot.close()


async def test_resumed_gateway_restores_readiness_without_on_ready(store):
    bot = BenteBot(Settings(health_port=0), store)
    try:
        await bot._async_setup_hook()
        bot._ready.set()
        bot.accepting = True
        await bot.on_disconnect()
        assert (await bot._health_ready(None)).status == 503
        await bot.on_resumed()
        assert (await bot._health_ready(None)).status == 200
        bot._closing = True
        await bot.on_resumed()
        bot.accepting = False
        await bot.on_resumed()
        assert not bot.accepting
    finally:
        bot._closing = False
        await bot.close()


async def test_bot_move_closes_old_source_before_new_channel_audio_can_be_saved(store):
    bot = BenteBot(Settings(guild_ids=frozenset({1}), health_port=0), store)
    public, private = SimpleNamespace(id=10), SimpleNamespace(id=20)
    guild = SimpleNamespace(id=1, voice_client=None)
    bot._connection.user = SimpleNamespace(id=999, guild=guild)
    bot.get_guild = lambda id_: guild

    async def noop(*args):
        pass

    async def audience():
        return Audience(frozenset({100}), 1)

    conversation = VoiceConversation(
        bot.settings,
        store,
        bot.memory,
        bot.chat,
        bot.tts,
        Source("channel", 10, 1),
        999,
        audience,
        noop,
        lambda: None,
        noop,
    )

    async def receiver_close():
        # A final result already queued by the newly connected private channel must not
        # be persisted under the old public channel during cancellation/teardown.
        await conversation.transcript(100, "Me", Transcript("PRIVATE CHANNEL SPEECH", final=True))

    receiver = SimpleNamespace(close=AsyncMock(side_effect=receiver_close))
    client = SimpleNamespace(disconnect=AsyncMock())
    guild.voice_client = client
    bot.voice_sessions[1] = (conversation, receiver, None)

    async def disconnect(**kwargs):
        await bot.on_voice_state_update(
            bot.user,
            SimpleNamespace(channel=private),
            SimpleNamespace(channel=None),
        )
        guild.voice_client = None

    client.disconnect.side_effect = disconnect
    try:
        await bot.on_voice_state_update(
            bot.user,
            SimpleNamespace(channel=public),
            SimpleNamespace(channel=private),
        )
        assert conversation.closed and 1 not in bot.voice_sessions
        receiver.close.assert_awaited_once()
        client.disconnect.assert_awaited_once_with(force=True)
        assert store.count() == 0
    finally:
        await bot.close()


async def test_initial_bot_join_does_not_disconnect_new_voice_client(store):
    bot = BenteBot(Settings(health_port=0), store)
    bot._connection.user = SimpleNamespace(id=999)
    member = SimpleNamespace(id=999, guild=SimpleNamespace(id=1))
    bot.leave_voice = AsyncMock()
    try:
        await bot.on_voice_state_update(
            member,
            SimpleNamespace(channel=None),
            SimpleNamespace(channel=SimpleNamespace(id=10)),
        )
        bot.leave_voice.assert_not_awaited()
    finally:
        await bot.close()
