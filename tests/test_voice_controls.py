import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Metrics, Source, Transcript
from bentebot.voice_control import spoken_control


@pytest.mark.parametrize(
    "text,action",
    [
        ("Bentebot, leave", "leave"),
        ("Can you please leave the voice call?", "leave"),
        ("I would like you to leave the call now", "leave"),
        ("Bente bot, kan du forlade opkaldet?", "leave"),
        ("Stop talking please", "stop"),
        ("Please. leave the call bend the button.", "leave"),
        ("Bend the button, please leave the call", "leave"),
        ("Leave Bentebot", "leave"),
        ("Stop speaking bend the button", "stop"),
        ("Pause listening", "pause"),
        ("Stop watching my screen", "screen_stop"),
        ("Watch my stream", "screen_start"),
        ("Please can you leave the call now, thanks?", "leave"),
        ("Bentebot, could you please stop talking now?", "stop"),
        ("Venligst kan du forlade opkaldet nu, tak?", "leave"),
        ("Bentebot, Bentebot. stop talking, Bentebot.", "stop"),
        ("Bentebot stop stop stop talking please", "stop"),
        ("Please Bente could you please stop talking", "stop"),
        ("Please Bentebot leave the call", "leave"),
        ("Hello Bente please disconnect now", "leave"),
        ("Bente Bente Bente please leave", "leave"),
        ("Hold on a second Bente", "stop"),
        ("Give me a second please", "stop"),
        ("That's enough Bentebot", "stop"),
        ("Please stop your response", "stop"),
        ("Leave the vc please", "leave"),
        ("Vent et øjeblik Bente", "stop"),
        ("Can you please stop talking?", "stop"),
        ("Stop talking now, Bentebot.", "stop"),
        ("Perfect. Can you leave the chat? please?", "leave"),
        ("Bentebot, you pause for a second?", "stop"),
        ("Great, thank you. Could you please leave the call?", "leave"),
        ("I'm sorry, can you pause for a second?", "stop"),
        ("Sorry, could you please stop talking?", "stop"),
        ("Excuse me, would you leave the call please?", "leave"),
        ("I am sorry, can you pause listening?", "pause"),
        ("Oh, sorry, sorry, sorry. Can you please pause again?", "stop"),
        ("Thank you. can you just please leave the call now, Bentebot?", "leave"),
        ("Can you just pause for a second please?", "stop"),
        ("Can you pause for et sekund?", "stop"),
        ("Thank you. You can leave the call now.", "leave"),
    ],
)
def test_clear_spoken_controls(text, action):
    assert spoken_control(text) == action


@pytest.mark.parametrize(
    "text",
    [
        "Don't leave the call",
        "I told you to leave the call earlier",
        "If you leave the call, will the viewer leave too?",
        "Show me the code to leave the call",
        "John, leave the call",
        "I am about to share my screen",
        "We should leave the dungeon",
        "Please don't leave the call now, thanks",
        "Please can you explain how to leave the call?",
        "Could you please not stop watching my screen?",
        "Don't leave the call bend the button",
        "John leave the call bend the button",
        "Show me how to bend the button",
        "If I ask you to leave the call bend the button",
        "If you can, please leave the call",
        '"Stop talking"',
        "'leave the call'",
        "Stop speaking like a robot",
        "Stop talking about Docker",
        "Stop speaking in Danish",
        "Bentebot is a funny name",
        "Bente Bente",
        "Perfect, John, can you leave the chat please?",
        "Perfect, you said leave the call earlier",
        "Bentebot, you said stop talking",
        "Bentebot, you pause the video when I ask",
        "Bentebot, you stop speaking like a robot please",
        "Great, could you leave a comment on my code?",
        "Perfect, don't leave the chat",
        "Sorry, can you pause the video?",
        "I'm sorry, don't stop talking",
        "Sorry, can you stop talking about Docker?",
        "Can you just not leave the call?",
        "Oh sorry, can you pause the video again?",
        "Just stop talking about Docker again",
        "Thank you. You can tell John to leave the call now.",
    ],
)
def test_discussion_and_negation_are_not_commands(text):
    assert spoken_control(text) is None


@pytest.mark.parametrize(
    "utterance",
    [
        "Bentebot, leave the call",
        "Perfect. Can you leave the chat? please?",
        "Thank you. can you just please leave the call now, Bentebot?",
        "Thank you. You can leave the call now.",
    ],
)
async def test_spoken_leave_executes_without_model_and_can_close_its_stt_caller(store, utterance):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    channel = SimpleNamespace(id=10)
    member = SimpleNamespace(id=100, bot=False, voice=SimpleNamespace(channel=channel))
    client = SimpleNamespace(disconnect=AsyncMock())
    guild = SimpleNamespace(get_member=lambda user: member if user == 100 else None, voice_client=client)
    bot.get_guild = lambda _: guild
    source = Source("channel", 10, 1)
    chat = SimpleNamespace(stream=Mock(side_effect=AssertionError("Commands must bypass generation")))
    conversation = VoiceConversation(
        bot.settings,
        store,
        bot.memory,
        chat,
        bot.tts,
        source,
        999,
        AsyncMock(),
        AsyncMock(),
        Mock(),
        AsyncMock(),
        control=lambda user, text: bot.handle_spoken_control(1, 10, user, text),
    )
    caller = None

    async def close_receiver():
        assert asyncio.current_task() is not caller
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)

    receiver = SimpleNamespace(close=AsyncMock(side_effect=close_receiver))
    bot.voice_sessions[1] = (conversation, receiver, None)

    async def stt_caller():
        await conversation.transcript(100, "Me", Transcript(utterance, final=True))
        await conversation._answer()
        await asyncio.Event().wait()

    try:
        caller = asyncio.create_task(stt_caller())
        await asyncio.sleep(0)
        jobs = list(bot.voice_jobs)
        assert jobs
        await asyncio.wait_for(asyncio.gather(*jobs), 2)
        assert 1 not in bot.voice_sessions
        assert conversation.closed and not conversation.pending
        client.disconnect.assert_awaited_once_with(force=True)
        chat.stream.assert_not_called()
        assert conversation.metrics.last_control_action == "leave"
        assert conversation.metrics.last_control_result == "completed"
        assert conversation.metrics.last_control_method == "direct"
        assert conversation.metrics.control_requests == 1
    finally:
        if caller:
            caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
        bot.voice_sessions.clear()
        await bot.close()


async def test_repeated_natural_stops_execute_without_model_and_increment_status(store):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    channel = SimpleNamespace(id=10)
    member = SimpleNamespace(id=100, bot=False, voice=SimpleNamespace(channel=channel))
    bot.get_guild = lambda _: SimpleNamespace(get_member=lambda _: member)
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1), metrics=Metrics(), pending=True, interrupt=AsyncMock()
    )
    conversation.hold_output = AsyncMock()
    bot.voice_sessions[1] = (conversation, None, None)
    bot.chat.resolve_spoken_control = AsyncMock(side_effect=AssertionError("No model needed"))
    try:
        for number, text in enumerate(
            [
                "Can you please stop talking?",
                "Bentebot, you pause for a second?",
                "I'm sorry, can you pause for a second?",
            ],
            start=1,
        ):
            assert await bot.handle_spoken_control(1, 10, 100, text)
            await asyncio.gather(*list(bot.voice_jobs))
            assert conversation.hold_output.await_count == number
            assert conversation.metrics.control_requests == number
            assert conversation.metrics.last_control_action == "stop"
            assert conversation.metrics.last_control_result == "completed"
            assert conversation.metrics.last_control_method == "direct"
        assert not await bot.handle_spoken_control(1, 10, 100, "Perfect, John, leave the chat")
        assert conversation.metrics.control_requests == 3
        bot.chat.resolve_spoken_control.assert_not_awaited()
    finally:
        bot.voice_sessions.clear()
        await bot.close()


@pytest.mark.parametrize(
    "text", ["Bentebot, bug, please stop. talking.", "Before we continue, could you pause for a second?"]
)
async def test_noisy_addressed_stop_does_not_restart_interrupted_story(store, text):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1}), turn_silence_ms=200), store)
    channel = SimpleNamespace(id=10)
    member = SimpleNamespace(id=100, bot=False, voice=SimpleNamespace(channel=channel))
    bot.get_guild = lambda _: SimpleNamespace(get_member=lambda _: member)
    resolved = asyncio.Event()

    async def resolve(text, model):
        resolved.set()
        return "stop"

    bot.chat.resolve_spoken_control = AsyncMock(side_effect=resolve)
    chat = SimpleNamespace(stream=Mock(side_effect=AssertionError("Story must not restart")))
    conversation = VoiceConversation(
        bot.settings,
        store,
        bot.memory,
        chat,
        bot.tts,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        Mock(),
        AsyncMock(),
        control=lambda user, text: bot.handle_spoken_control(1, 10, user, text),
    )
    bot.voice_sessions[1] = (conversation, None, None)
    story = conversation.task = asyncio.create_task(asyncio.Event().wait())
    try:
        await conversation.transcript(100, "Me", Transcript(text))
        assert story.cancelled()
        await conversation.transcript(100, "Me", Transcript(text, final=True))
        conversation.start()
        await asyncio.wait_for(resolved.wait(), 2)
        await asyncio.gather(*list(bot.voice_jobs))
        await asyncio.sleep(0.3)
        assert not conversation.pending
        assert not conversation.pending_controls
        assert conversation.metrics.control_requests == 1
        assert conversation.metrics.last_control_action == "stop"
        assert conversation.metrics.last_control_method == "intent"
        bot.chat.resolve_spoken_control.assert_awaited_once()
        chat.stream.assert_not_called()
    finally:
        await asyncio.gather(*list(bot.voice_jobs))
        bot.voice_sessions.clear()
        await conversation.close()
        await bot.close()


async def test_spoken_control_requires_current_channel_membership(store):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    member = SimpleNamespace(bot=False, voice=SimpleNamespace(channel=SimpleNamespace(id=99)))
    bot.get_guild = lambda _: SimpleNamespace(get_member=lambda _: member)
    bot.voice_sessions[1] = (SimpleNamespace(source=Source("channel", 10, 1)), None, None)
    bot.leave_voice = AsyncMock()
    try:
        assert await bot.handle_spoken_control(1, 10, 100, "leave the call")
        assert not bot.voice_jobs
        bot.leave_voice.assert_not_awaited()
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_committed_transcript_waits_for_end_of_turn_before_control_dispatch(store):
    handled = asyncio.Event()

    async def control(user, text):
        assert user == 100 and text == "leave the call"
        handled.set()
        return True

    conversation = VoiceConversation(
        Settings(),
        store,
        None,
        None,
        None,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        Mock(),
        AsyncMock(),
        control=control,
    )
    conversation.last_speech = float("inf")  # Keep the quiet gate closed deterministically.
    conversation.start()
    try:
        await conversation.transcript(100, "Me", Transcript("leave the call", final=True))
        await asyncio.sleep(0)
        assert not handled.is_set()
        conversation.last_speech = 0
        conversation.changed.set()
        await asyncio.wait_for(handled.wait(), 1)
        assert not conversation.pending
    finally:
        await conversation.close()


async def test_control_is_not_overwritten_or_blocked_by_another_callers_speech(store):
    handled = asyncio.Event()

    async def control(user, text):
        assert user == 100 and text == "leave the call"
        handled.set()
        return True

    conversation = VoiceConversation(
        Settings(turn_silence_ms=200),
        store,
        None,
        None,
        None,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        Mock(),
        AsyncMock(),
        control=control,
    )
    conversation.speaker_last_voice[100] = 0
    conversation.last_speech = float("inf")  # Other callers keep the channel busy.
    await conversation.transcript(100, "Me", Transcript("leave the call", final=True))
    await conversation.transcript(200, "Other", Transcript("my unrelated question", final=True))
    await conversation.transcript(300, "Third", Transcript("still speaking"))
    conversation.start()
    try:
        await asyncio.wait_for(handled.wait(), 1)
        assert conversation.partial[300] == "still speaking"
        assert conversation.latest_speaker_id == 200
    finally:
        await conversation.close()


async def test_speakers_new_final_supersedes_their_own_pending_command(store):
    control = AsyncMock(return_value=True)
    conversation = VoiceConversation(
        Settings(turn_silence_ms=200),
        store,
        None,
        None,
        None,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        Mock(),
        AsyncMock(),
        control=control,
    )
    conversation.partial[200] = "still speaking"  # Keep ordinary chat pending.
    await conversation.transcript(100, "Me", Transcript("leave the call", final=True))
    await conversation.transcript(100, "Me", Transcript("actually never mind", final=True))
    conversation.start()
    try:
        await asyncio.sleep(0.15)
        control.assert_not_awaited()
        assert not conversation.pending_controls
    finally:
        await conversation.close()


async def test_other_callers_cannot_cancel_inflight_command_classification(store):
    entered, finish = asyncio.Event(), asyncio.Event()

    async def control(user, text):
        entered.set()
        await finish.wait()
        return True

    conversation = VoiceConversation(
        Settings(),
        store,
        None,
        None,
        None,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        Mock(),
        AsyncMock(),
        control=control,
    )
    conversation.task = asyncio.create_task(conversation._dispatch_control(100, "leave the call Bender"))
    try:
        await entered.wait()
        conversation.speech_activity(200)
        await conversation.transcript(200, "Other", Transcript("one two three four five"))
        assert not conversation.task.done()
        finish.set()
        await conversation.task
    finally:
        await conversation.close()


@pytest.mark.parametrize("command", ["join", "leave"])
async def test_join_and_leave_remove_deferred_confirmation_without_sending_text(store, command):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    user = SimpleNamespace(id=100, voice=SimpleNamespace(channel=SimpleNamespace(id=10)))
    interaction = SimpleNamespace(
        user=user,
        guild_id=1,
        channel=SimpleNamespace(id=20),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
        delete_original_response=AsyncMock(),
    )
    bot.is_admin = lambda _: False
    bot.join_voice = AsyncMock()
    bot.leave_voice = AsyncMock()
    if command == "leave":
        bot.voice_sessions[1] = (SimpleNamespace(source=Source("channel", 10, 1)), None, None)
    try:
        await bot.tree.get_command("voice").get_command(command).callback(interaction)
        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        interaction.delete_original_response.assert_awaited_once()
        interaction.followup.send.assert_not_awaited()
    finally:
        bot.voice_sessions.clear()
        await bot.close()


@pytest.mark.parametrize(
    "text,action",
    [
        ("Stop speaking", "stop"),
        ("Can you pause for a sec bente", "stop"),
        ("please leave the call bentebot", "leave"),
        ("Bentebot, can you leave the call", "leave"),
        ("Bente, stop talking please", "stop"),
        ("Kan du holde pause et \u00f8jeblik Bente", "stop"),
    ],
)
def test_natural_call_control_phrasing(text, action):
    assert spoken_control(text) == action
