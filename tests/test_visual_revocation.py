from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.config import Settings
from bentebot.conversation import TextConversation, VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import ProviderError

CHANNEL = Source("channel", 10, 1)
FRAME = Source("screen", 10, 1, owner_id=100, session_id="sample", audience_ids=(100,))


async def permitted(*args):
    return True


async def audience():
    return Audience(frozenset({100}), 1)


async def test_stopping_capture_during_voice_generation_prevents_audio_and_text(store):
    active = True

    async def guard(sources):
        return active

    async def extra(recipients):
        return [], (FRAME,), "vision"

    async def stream(messages, model):
        nonlocal active
        active = False
        yield "Private visual detail."

    synth, play, written, notices = AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()
    store.save(Event(CHANNEL, 100, "Me", "What do you see?"))
    conversation = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(permitted), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synth),
        CHANNEL,
        999,
        audience,
        play,
        lambda: None,
        notices,
        text_output=written,
        extra_context=extra,
        extra_guard=guard,
    )
    conversation.latest_request = "What do you see?"
    await conversation._answer()
    synth.assert_not_awaited()
    play.assert_not_awaited()
    written.assert_not_awaited()
    assert notices.await_count == 1
    event = store.recent(1, frozenset())[-1]
    assert event.spoken == "" and FRAME in event.provenance


async def test_stopping_capture_during_text_generation_prevents_delivery(store):
    active = True

    async def guard(sources):
        return active

    async def extra(recipients):
        return [], (FRAME,), "vision"

    async def stream(messages, model):
        nonlocal active
        active = False
        yield "Private visual detail."

    sent = AsyncMock()
    store.save(Event(CHANNEL, 100, "Me", "What do you see?"))
    conversation = TextConversation(
        store, Memory(store, AccessPolicy(permitted), 10000), SimpleNamespace(stream=stream)
    )
    with pytest.raises(ProviderError, match="Screen sharing ended"):
        await conversation.answer(
            CHANNEL, 999, audience, sent, "chat", extra_context=extra, extra_guard=guard
        )
    sent.assert_not_awaited()
    assert store.recent(1, frozenset())[-1].delivery == "none"
