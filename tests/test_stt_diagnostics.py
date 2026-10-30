import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from websockets.asyncio.server import serve

from bentebot.config import Settings
from bentebot.domain import Metrics, Transcript
from bentebot.providers import StreamingSTT, STTStreamError, TranscriptParser
from bentebot.voice import Speaker, VoiceReceiver


@pytest.mark.parametrize(
    "description,reason",
    [
        ("ASR failed: RuntimeError: CUDA failed with error out of memory", "out_of_memory"),
        ("CUDA_ERROR_OUT_OF_MEMORY", "out_of_memory"),
        ("CUBLAS_STATUS_ALLOC_FAILED", "out_of_memory"),
        ("CUDNN_STATUS_ALLOC_FAILED", "out_of_memory"),
        ("Transcription backlog did not clear within 30 seconds.", "overloaded"),
        ("Audio queue capacity exceeded", "overloaded"),
        ("No audio or KeepAlive message was received for 30 seconds.", "timeout"),
        ("Internal transcription error.", "provider_error"),
        (None, "provider_error"),
    ],
)
def test_provider_error_retains_only_fixed_category(description, reason):
    payload = {
        "type": "Error",
        "description": f"{description} SECRET_TRANSCRIPT SECRET_KEY" if description else None,
    }
    with pytest.raises(STTStreamError) as failure:
        TranscriptParser().parse(payload)
    assert failure.value.reason == reason
    assert "SECRET" not in str(failure.value)
    assert "SECRET" not in repr(vars(failure.value))


async def test_last_failure_remains_visible_after_success_without_raw_details(caplog):
    conversation = SimpleNamespace(
        metrics=Metrics(),
        source=SimpleNamespace(channel_id=10),
        partial={1: "PRIVATE"},
        transcript=AsyncMock(),
    )

    class Failure:
        async def transcribe(self, audio):
            raise STTStreamError("out_of_memory", close_code=1011)
            yield

    receiver = VoiceReceiver(SimpleNamespace(), conversation, Failure(), AsyncMock())
    await receiver._transcribe(Speaker(1, "Me"))
    assert conversation.metrics.stt_failures == 1
    assert conversation.metrics.stt_last_failure_reason == "out_of_memory"
    assert conversation.metrics.stt_last_close_code == 1011
    assert conversation.metrics.stt_last_failure_utc
    assert "out_of_memory" in caplog.text and "PRIVATE" not in caplog.text
    assert "out of memory" in receiver.last_error

    async def good_updates():
        yield Transcript("Private new words", final=True)

    await receiver._receive_updates(Speaker(1, "Me"), good_updates())
    assert receiver.last_error == ""
    assert conversation.metrics.stt_last_failure_reason == "out_of_memory"
    assert "Private new words" not in caplog.text


@pytest.mark.parametrize("case", ["close", "invalid_json", "error_event"])
async def test_real_socket_failure_preserves_category_without_promoting_partial(case):
    async def handler(socket):
        await socket.recv()
        await socket.send(
            json.dumps({"type": "Results", "channel": {"alternatives": [{"transcript": "unfinished"}]}})
        )
        if case == "invalid_json":
            await socket.send("SECRET invalid json")
        elif case == "error_event":
            await socket.send(json.dumps({"type": "Error", "description": "CUDA out of memory SECRET"}))
        await socket.close(code=1011, reason="SECRET server failure")

    async def audio():
        yield b"\0" * 3200
        await asyncio.Event().wait()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        provider = StreamingSTT(Settings(stt_url=f"ws://127.0.0.1:{port}/v1/listen"))
        updates = []
        with pytest.raises(STTStreamError) as failure:
            async with asyncio.timeout(3):
                async for update in provider.transcribe(audio()):
                    updates.append(update)
        expected = {
            "close": "unexpected_close",
            "invalid_json": "invalid_response",
            "error_event": "out_of_memory",
        }
        assert failure.value.reason == expected[case]
        assert "SECRET" not in str(failure.value)
        if case == "close":
            assert failure.value.close_code == 1011
        assert updates and all(not update.final for update in updates)
