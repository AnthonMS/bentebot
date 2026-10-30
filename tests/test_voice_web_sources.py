import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import CHAT_OBSERVATION


@pytest.mark.parametrize("delivered", [True, False])
async def test_voice_web_sources_are_written_without_mirroring_answer_or_reading_urls(
    store, wav_bytes, delivered
):
    channel = Source("channel", 10, 1)
    store.save(Event(channel, 100, "Me", "What is the latest project news?", kind="voice"))
    spoken = []

    async def stream(_messages, _model):
        CHAT_OBSERVATION.get().web_sources.append("https://example.com/release")
        yield "The release landed today. https://example.com/release"

    async def synthesize(text, _voice):
        spoken.append(text)
        return wav_bytes

    async def play(_pcm, text, progress):
        progress(text)

    write = AsyncMock(return_value=delivered)
    conversation = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synthesize),
        channel,
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        play,
        lambda: None,
        AsyncMock(),
        text_output=write,
        text_channel_id=20,
    )
    conversation.latest_request = "What is the latest project news?"
    await conversation._answer()
    assert spoken == ["The release landed today."]
    write.assert_awaited_once()
    assert "https://example.com/release" in write.await_args.args[0]
    assert "The release landed today." not in write.await_args.args[0]
    saved = next(event for event in store.recent(1, frozenset({100})) if event.kind == "assistant")
    assert bool(saved.written) is delivered
    assert saved.spoken == "The release landed today."
    assert conversation.metrics.last_chat["web_source_delivery"] == (
        "delivered" if delivered else "denied_or_changed"
    )


@pytest.mark.parametrize("write_error", [False, True])
async def test_source_delivery_runs_during_playback_and_survives_later_barge_in(
    store, wav_bytes, write_error
):
    channel = Source("channel", 10, 1)
    store.save(Event(channel, 100, "Me", "Latest release?", kind="voice"))
    playback_started, write_attempted = asyncio.Event(), asyncio.Event()

    async def stream(messages, model):
        CHAT_OBSERVATION.get().web_sources.append("https://example.com/release")
        yield "Here is the verified release. More explanation follows."

    async def play(pcm, text, progress):
        progress(text)
        playback_started.set()
        await asyncio.Event().wait()

    async def write(text, sources, audience):
        write_attempted.set()
        if write_error:
            raise RuntimeError("synthetic Discord failure")
        return True

    notices = AsyncMock()
    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        channel,
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        play,
        lambda: None,
        notices,
        text_output=write,
        text_channel_id=20,
    )
    session.latest_request = "Latest release?"
    task = asyncio.create_task(session._answer())
    try:
        async with asyncio.timeout(3):
            await playback_started.wait()
            await write_attempted.wait()
        assert not task.done()  # Source attempted before playback completes.
        assert session.metrics.last_chat["web_source_delivery"] == ("failed" if write_error else "delivered")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert session.metrics.provider_errors == 0
    notices.assert_not_awaited()
    saved = next(event for event in store.recent(1, frozenset({100})) if event.kind == "assistant")
    assert saved.interrupted
    assert bool(saved.written) is not write_error


async def test_interrupted_turn_cancels_inflight_source_delivery(store, wav_bytes):
    channel = Source("channel", 10, 1)
    store.save(Event(channel, 100, "Me", "Latest release?", kind="voice"))
    write_started, write_cancelled = asyncio.Event(), asyncio.Event()

    async def stream(messages, model):
        CHAT_OBSERVATION.get().web_sources.append("https://example.com/release")
        yield "The release is ready."

    async def write(text, sources, audience):
        write_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            write_cancelled.set()

    async def play(pcm, text, progress):
        await asyncio.Event().wait()

    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        channel,
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        play,
        lambda: None,
        AsyncMock(),
        text_output=write,
    )
    session.latest_request = "Latest release?"
    task = asyncio.create_task(session._answer())
    async with asyncio.timeout(3):
        await write_started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert write_cancelled.is_set()
    assert session.metrics.last_chat["web_source_delivery"] == "cancelled"


async def test_prefetched_source_card_waits_for_acknowledgement_barrier(store, wav_bytes):
    channel = Source("channel", 10, 1)
    store.save(Event(channel, 100, "Me", "Latest release?", kind="voice"))
    prepared, barrier = asyncio.Event(), asyncio.Event()

    async def stream(messages, model):
        CHAT_OBSERVATION.get().web_sources.append("https://example.com/release")
        yield "The release is ready."

    async def synthesize(text, voice):
        prepared.set()
        return wav_bytes

    write = AsyncMock(return_value=True)
    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synthesize),
        channel,
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        AsyncMock(),
        lambda: None,
        AsyncMock(),
        text_output=write,
    )
    session.latest_request = "Latest release?"
    task = asyncio.create_task(session._answer(playback_barrier=barrier))
    try:
        async with asyncio.timeout(3):
            await prepared.wait()
        write.assert_not_awaited()
        barrier.set()
        async with asyncio.timeout(3):
            await task
        write.assert_awaited_once()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
