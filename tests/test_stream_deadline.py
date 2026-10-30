import asyncio
import json

import httpx
import pytest

from bentebot.config import Settings
from bentebot.domain import ResponseRecord
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat, ProviderError
from bentebot.speech import speak_response


def delta(text):
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n"


async def test_slow_stream_consumer_is_not_cancelled_by_provider_deadline():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=delta("First ") + delta("second.") + "data: [DONE]\n\n"),
        )
    )
    provider = OpenAIChat(Settings(generation_timeout=0.03), client)
    pieces = []
    try:
        async with asyncio.timeout(1):
            async for piece in provider.stream([], "test"):
                pieces.append(piece)
                await asyncio.sleep(0.05)
        assert "".join(pieces) == "First second."
    finally:
        await client.aclose()


@pytest.mark.parametrize("filler", ["reasoning", "whitespace", "keepalive"])
async def test_voice_first_answer_deadline_ignores_non_answer_events_and_closes_stream(filler, caplog):
    class PendingStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.01)
                if filler == "reasoning":
                    yield b'data: {"choices":[{"delta":{"reasoning_content":"PRIVATE"}}]}\n\n'
                elif filler == "whitespace":
                    yield delta(" ").encode()
                else:
                    yield b": heartbeat\n\n"

        async def aclose(self):
            self.closed = True

    stream = PendingStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                stream=stream,
                headers={"x-litellm-model-api-base": "http://192.168.50.10:9292/v1"},
            )
        )
    ) as client:
        record = ResponseRecord()
        token = CHAT_OBSERVATION.set(record)
        try:
            provider = OpenAIChat(Settings(generation_timeout=1, voice_first_content_timeout=0.04), client)
            with pytest.raises(ProviderError, match="no answer text"):
                _ = [piece async for piece in provider.stream([], "local")]
        finally:
            CHAT_OBSERVATION.reset(token)
        assert stream.closed
        assert record.chat_metadata["generation_failure"] == "voice_first_content_timeout"
        assert "upstream_host=192.168.50.10" in caplog.text
        assert "PRIVATE" not in caplog.text


async def test_voice_first_answer_deadline_stops_after_real_text_without_cancelling_playback():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=delta("First.") + delta(" Second.") + "data: [DONE]\n\n")
        )
    ) as client:
        token = CHAT_OBSERVATION.set(ResponseRecord())
        try:
            provider = OpenAIChat(Settings(voice_first_content_timeout=0.03), client)
            parts = []
            async for piece in provider.stream([], "local"):
                parts.append(piece)
                await asyncio.sleep(0.04)
            assert "".join(parts) == "First. Second."
        finally:
            CHAT_OBSERVATION.reset(token)


async def test_sentence_queue_backpressure_does_not_leave_pipeline_hung(wav_bytes):
    text = "A complete sentence. " * 16
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=delta(text) + "data: [DONE]\n\n"),
        )
    )
    provider = OpenAIChat(Settings(generation_timeout=0.03), client)
    spoken = []

    async def synthesize(sentence):
        return wav_bytes

    async def play(pcm, sentence, progress):
        await asyncio.sleep(0.02)
        spoken.append(sentence)
        progress(sentence)

    async def allowed():
        return True

    try:
        async with asyncio.timeout(2):
            await speak_response(provider.stream([], "test"), synthesize, play, allowed, ResponseRecord())
        assert spoken == ["A complete sentence."] * 16
    finally:
        await client.aclose()


async def test_reasoning_only_stream_hits_provider_deadline_and_closes_response():
    class ReasoningStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.01)
                yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'

        async def aclose(self):
            self.closed = True

    stream = ReasoningStream()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    )
    provider = OpenAIChat(Settings(generation_timeout=0.04), client)
    try:
        async with asyncio.timeout(1):
            with pytest.raises(ProviderError, match="generation time limit"):
                _ = [piece async for piece in provider.stream([], "test")]
        assert stream.closed
    finally:
        await client.aclose()
