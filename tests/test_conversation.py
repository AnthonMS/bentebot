import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot.config import Settings
from bentebot.conversation import TextConversation, VoiceConversation
from bentebot.domain import Audience, Event, Source, Transcript
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import ProviderError

SOURCE = Source("channel", 10, 1)


@pytest.mark.parametrize("enabled", [True, False])
async def test_music_can_be_controlled_while_spoken_answer_is_held(store, enabled):
    control = AsyncMock(return_value=True)
    session = make_session(store, control=control)
    session.settings = replace(session.settings, music_enabled=enabled)
    session.output_held = True
    try:
        await session.transcript(100, "Me", Transcript("play some Agnes Obel", final=True))
        assert bool(session.pending_controls) == enabled
        if enabled:
            await session._dispatch_control(100, session.pending_controls.pop(100))
            control.assert_awaited_once_with(100, "play some Agnes Obel")
            assert session.output_held
    finally:
        await session.close()


async def yes(*args):
    return True


async def noop(*args):
    pass


async def audience():
    return Audience(frozenset({100}), 1)


def make_session(store, chat=None, **kwargs):
    settings = Settings(turn_silence_ms=200)
    memory = Memory(store, AccessPolicy(yes), 10000)
    return VoiceConversation(
        settings, store, memory, chat, None, SOURCE, 999, audience, noop, lambda: None, noop, **kwargs
    )


async def test_interruption_counts_revised_words_once_and_requires_speech(store):
    session = make_session(store)
    session.task = asyncio.create_task(asyncio.Event().wait())
    await session.transcript(100, "Me", Transcript("one two three four five"))
    assert not session.task.done()  # No recent actual speech.
    session.speech_activity()
    for text in ("one two", "one two three", "one two three four"):
        await session.transcript(100, "Me", Transcript(text))
        assert not session.task.done()
    await session.transcript(100, "Me", Transcript("one two three four five"))
    assert session.task.cancelled()
    assert session.metrics.interruptions == 1
    await session.close()


async def test_explicit_stop_and_multi_user_identity(store):
    session = make_session(store)
    session.task = asyncio.create_task(asyncio.Event().wait())
    session.speech_activity()
    await session.transcript(200, "Other", Transcript("stop", final=True))
    await session.transcript(100, "Me", Transcript("hello", final=True))
    assert session.task.cancelled()
    events = store.recent(1, frozenset())
    assert {event.author_id for event in events} == {100, 200}
    await session.close()


@pytest.mark.parametrize(
    "text",
    [
        "Stop speaking",
        "Pause listening",
        "Leave the call",
        "Bentebot Bentebot stop",
        "Can you please stop talking?",
        "Bentebot, you pause for a second?",
        "Perfect. Can you leave the chat? please?",
        "I'm sorry, can you pause for a second?",
    ],
)
async def test_delayed_explicit_control_interrupts_output_before_final_dispatch(store, text):
    session = make_session(store)
    session.task = asyncio.create_task(asyncio.Event().wait())
    # Recognition can arrive more than three seconds after the final VAD frame.
    session.last_speech = 0
    await session.transcript(100, "Me", Transcript(text))
    assert session.task.cancelled()
    assert session.metrics.interruptions == 1
    assert not session.pending  # A provisional hypothesis must not execute the action.
    await session.close()


@pytest.mark.parametrize(
    "request_text", ["Please leave the call bend a bar", "Bentebot, bug, please stop talking"]
)
async def test_unresolved_control_asks_briefly_without_model_capability_claim(store, wav_bytes, request_text):
    chat = SimpleNamespace(stream=Mock(side_effect=AssertionError("No conversational capability claims")))
    session = make_session(store, chat=chat, control=AsyncMock(return_value=False))
    session.control_feedback = AsyncMock(return_value="Could you rephrase that?")
    session.tts = SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes))
    session.play = AsyncMock()
    session.latest_request = request_text
    session.latest_speaker_id = 100
    store.save(Event(SOURCE, 100, "Me", session.latest_request))
    await session._answer()
    assert session.play.await_count >= 1
    assert session.tts.synthesize.call_args_list[0].args[0] == "Could you rephrase that?"
    session.control_feedback.assert_awaited_once_with("clarify", request_text)
    chat.stream.assert_not_called()
    assert session.metrics.provider_errors == 0
    await session.close()


async def test_waits_for_quiet_and_all_partials_before_answer(store):
    answered = asyncio.Event()
    session = make_session(store)

    async def answer():
        answered.set()

    session._answer = answer
    session.start()
    try:
        session.speech_activity()
        await session.transcript(100, "Me", Transcript("hello", final=True))
        await session.transcript(200, "Other", Transcript("still speaking"))
        await asyncio.sleep(0.25)
        assert not answered.is_set()
        await session.transcript(200, "Other", Transcript("still speaking done", final=True))
        await asyncio.wait_for(answered.wait(), 1)
    finally:
        await session.close()


async def test_interrupted_draft_is_saved_but_not_marked_spoken(store, wav_bytes):
    playing = asyncio.Event()

    async def stream(messages, model):
        yield "Here is what I intended to say. More detail."

    async def synthesize(text, voice):
        return wav_bytes

    async def play(pcm, sentence, progress):
        progress("Here is")
        playing.set()
        await asyncio.Event().wait()

    session = make_session(store, chat=SimpleNamespace(stream=stream))
    session.tts = SimpleNamespace(synthesize=synthesize)
    session.play = play
    store.save(Event(SOURCE, 100, "Me", "hello"))
    session.task = asyncio.create_task(session._answer())
    await asyncio.wait_for(playing.wait(), 2)
    await session.interrupt()
    event = store.recent(1, frozenset())[-1]
    assert event.kind == "assistant" and event.interrupted
    assert event.spoken == "Here is" and "More detail" in event.generated
    assert event.provenance == (SOURCE,)
    await session.close()


async def test_text_reply_rechecks_audience_after_generation(store):
    changed = False

    async def current_audience():
        return Audience(frozenset({100, 200}) if changed else frozenset({100}), 1)

    async def stream(messages, model):
        nonlocal changed
        yield "private answer"
        changed = True

    sent = []

    async def send(text):
        sent.append(text)

    store.save(Event(SOURCE, 100, "Me", "private question"))
    conversation = TextConversation(
        store, Memory(store, AccessPolicy(yes), 10000), SimpleNamespace(stream=stream)
    )
    with pytest.raises(ProviderError):
        await conversation.answer(SOURCE, 999, current_audience, send, "test")
    assert sent == []
    event = store.recent(1, frozenset())[-1]
    assert event.delivery == "none" and event.generated == "private answer"
