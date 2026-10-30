import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.domain import Metrics, Transcript
from bentebot.voice import Speaker, VoiceReceiver


async def test_failed_turn_callback_does_not_disconnect_stt_or_drop_next_command(caplog):
    class STT:
        async def transcribe(self, audio):
            yield Transcript("Stop speaking", final=True)
            yield Transcript("Please leave the call", final=True)

    callback = AsyncMock(side_effect=[RuntimeError("PRIVATE_TRANSCRIPT_NOT_FOR_LOGS"), None])
    conversation = SimpleNamespace(
        metrics=Metrics(),
        source=SimpleNamespace(channel_id=10),
        partial={100: "unfinished"},
        transcript=callback,
    )
    receiver = VoiceReceiver(SimpleNamespace(), conversation, STT(), AsyncMock())
    await receiver._transcribe(Speaker(100, "Me"))
    assert callback.await_count == 2
    assert conversation.metrics.provider_errors == 1
    assert not conversation.partial
    receiver.notify.assert_not_awaited()
    assert "voice_transcript_handler_failed" in caplog.text
    assert "PRIVATE_TRANSCRIPT_NOT_FOR_LOGS" not in caplog.text
    assert "stt_stream" not in caplog.text


async def test_cancelled_receiver_closes_suspended_stt_generator():
    entered, closed = asyncio.Event(), asyncio.Event()

    class STT:
        async def transcribe(self, audio):
            try:
                entered.set()
                yield Transcript("hello")
            finally:
                closed.set()

    # Keep the generator suspended at its yield while the handler is running.
    async def handle(*args):
        await asyncio.sleep(60)

    conversation = SimpleNamespace(transcript=handle, metrics=Metrics())
    receiver = VoiceReceiver(SimpleNamespace(), conversation, STT(), AsyncMock())
    task = asyncio.create_task(receiver._transcribe(Speaker(100, "Me")))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_failed_stream_waits_for_fresh_speech_before_reconnecting():
    connected = asyncio.Event()

    class STT:
        async def transcribe(self, audio):
            connected.set()
            async for _ in audio:
                pass
            if False:
                yield

    conversation = SimpleNamespace(
        metrics=Metrics(),
        settings=SimpleNamespace(max_speakers=6),
        partial={},
        speech_activity=lambda user_id: None,
    )
    client = SimpleNamespace(listen=lambda sink: None, stop_listening=lambda: None)
    receiver = VoiceReceiver(client, conversation, STT(), AsyncMock())
    voiced = False
    speaker = Speaker(100, "Me", vad=SimpleNamespace(is_speech=lambda frame, rate: voiced))
    speaker.last_voice = time.monotonic() - 10
    speaker.task = asyncio.create_task(asyncio.sleep(0))
    await speaker.task
    receiver.speakers[100] = speaker
    receiver.start()
    try:
        receiver.incoming.put((100, "Me", b"\0" * 3840))
        await asyncio.sleep(0.15)
        assert not connected.is_set()
        assert speaker.last_voice == 0
        voiced = True
        receiver.incoming.put((100, "Me", b"\0\1" * 1920))
        await asyncio.wait_for(connected.wait(), 1)
    finally:
        await receiver.close()


async def test_provider_failure_survives_failed_discord_notification(caplog):
    class STT:
        async def transcribe(self, audio):
            raise RuntimeError("PRIVATE_PROVIDER_BODY")
            yield

    conversation = SimpleNamespace(
        metrics=Metrics(), source=SimpleNamespace(channel_id=10), partial={100: "pending"}
    )
    receiver = VoiceReceiver(
        SimpleNamespace(), conversation, STT(), AsyncMock(side_effect=RuntimeError("PRIVATE_NOTICE_BODY"))
    )
    speaker = Speaker(100, "Me")
    await receiver._transcribe(speaker)
    assert speaker.retry_after > time.monotonic()
    assert not conversation.partial
    assert conversation.metrics.provider_errors == 1
    assert "stt_notice outcome=failed" in caplog.text
    assert "PRIVATE" not in caplog.text
