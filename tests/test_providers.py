import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from aiohttp import web
from websockets.asyncio.server import serve

from bentebot.config import Settings
from bentebot.providers import OpenAIChat, OpenAITTS, ProviderError, StreamingSTT, TranscriptParser


def result(text, *, final=False, endpoint=False, start=0, duration=1):
    return {
        "type": "Results",
        "is_final": final,
        "speech_final": endpoint,
        "start": start,
        "duration": duration,
        "channel": {"alternatives": [{"transcript": text}]},
    }


def test_revised_partials_replace_and_final_segments_accumulate():
    parser = TranscriptParser()
    assert parser.parse(result("hello")).text == "hello"
    assert parser.parse(result("hello world")).text == "hello world"
    assert parser.parse(result("hello world", final=True)).text == "hello world"
    assert parser.parse(result("again", start=1)).text == "hello world again"
    update = parser.parse(result("again", final=True, endpoint=True, start=1))
    assert update.final and update.text == "hello world again"
    assert parser.parse(result("again", final=True, endpoint=True, start=1)) is None
    assert parser.text == ""


def test_empty_endpoint_flushes_prior_finals():
    parser = TranscriptParser()
    parser.parse(result("one two", final=True))
    assert parser.parse(result("", final=True, endpoint=True, start=1)).text == "one two"


def test_utterance_end_and_provider_errors():
    parser = TranscriptParser()
    parser.parse(result("hello", final=True))
    assert parser.parse({"type": "UtteranceEnd", "last_word_end": 1}).final
    assert parser.parse({"type": "Metadata"}) is None
    with pytest.raises(ProviderError):
        parser.parse({"type": "Error", "description": "secret token"})


@pytest.mark.parametrize("context", ["Bentebot, Bente.", "", "Bente, skærm."])
async def test_real_websocket_returns_interim_before_input_finishes(context):
    got_interim = asyncio.Event()
    input_open = True
    received = []

    async def handler(socket):
        assert "interim_results=true" in socket.request.path
        params = parse_qs(urlsplit(socket.request.path).query)
        assert params.get("context") == ([context] if context else None)
        async for message in socket:
            received.append(message)
            if isinstance(message, bytes):
                await socket.send(json.dumps(result("still talking")))
            else:
                assert json.loads(message)["type"] == "CloseStream"
                await socket.send(json.dumps(result("still talking done", final=True, endpoint=True)))
                await socket.close()

    async def audio():
        nonlocal input_open
        yield b"\0" * 3200
        await asyncio.wait_for(got_interim.wait(), 2)
        input_open = False

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = StreamingSTT(Settings(stt_url=f"ws://127.0.0.1:{port}/v1/listen", stt_context=context))
        updates = []
        async with asyncio.timeout(5):
            async for update in stt.transcribe(audio()):
                updates.append(update)
                if not update.final:
                    assert input_open
                    got_interim.set()
        assert updates[-1].final
        assert updates[-1].text == "still talking done"
        assert received[0] == b"\0" * 3200


@pytest.mark.parametrize("context", ["x" * 1001, "invalid\x00context"])
def test_stt_context_rejects_invalid_provider_hint(context):
    with pytest.raises(ValueError, match="stt_context"):
        Settings(stt_context=context).validate(login=False)


async def test_real_http_sse_chat_and_wav_contract(wav_bytes):
    calls = []

    async def chat(request):
        calls.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for text in ("Hello", " world."):
            await response.write(
                ("data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n").encode()
            )
            await asyncio.sleep(0)
        await response.write(b"data: [DONE]\n\n")
        return response

    async def tts(request):
        payload = await request.json()
        assert payload["response_format"] == "wav"
        assert payload["input"] == "Hello world."
        return web.Response(body=wav_bytes, content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/audio/speech", tts)

    async def models(request):
        return web.json_response({"data": [{"id": "test"}]})

    app.router.add_get("/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    settings = Settings(
        chat_base_url=f"http://127.0.0.1:{port}/v1", tts_base_url=f"http://127.0.0.1:{port}/v1"
    )
    provider, speech = OpenAIChat(settings), OpenAITTS(settings)
    try:
        assert await provider.models() == ["test"]
        assert "".join([piece async for piece in provider.stream([], "test")]) == "Hello world."
        assert await speech.synthesize("Hello world.") == wav_bytes
        assert len(calls) == 1
        assert calls[0] == {
            "model": "test",
            "messages": calls[0]["messages"],
            "stream": True,
            "max_tokens": 2048,
        }
        assert len(calls[0]["messages"]) == 1 and calls[0]["messages"][0]["role"] == "system"
        assert 'selected LiteLLM model alias is "test"' in calls[0]["messages"][0]["content"]
    finally:
        await provider.close()
        await speech.close()
        await runner.cleanup()


async def test_http_errors_are_sanitized_and_no_retries_to_other_providers():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="SECRET_KEY sensitive transcript")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIChat(Settings(), client)
    with pytest.raises(ProviderError) as error:
        _ = [piece async for piece in provider.stream([], "test")]
    assert "SECRET" not in str(error.value) and calls == 1
    await client.aclose()


async def test_malformed_tts_and_size_limit():
    async def handler(request):
        return httpx.Response(200, content=b"not-a-wav")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError):
        await OpenAITTS(Settings(tts_base_url="http://test/v1"), client).synthesize("hello")
    await client.aclose()


async def test_chat_caps_generation():
    payload = {"choices": [{"delta": {"content": "x" * 100}}]}
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"),
        )
    )
    chat = OpenAIChat(Settings(max_response_chars=20), client)
    assert "".join([part async for part in chat.stream([], "test")]) == "x" * 20
    await client.aclose()


async def test_stt_sender_failure_does_not_hang_waiting_for_server():
    async def handler(socket):
        await socket.wait_closed()

    async def broken_audio():
        raise RuntimeError("audio ingress failed")
        yield b""  # Make this an async generator.

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = StreamingSTT(Settings(stt_url=f"ws://127.0.0.1:{port}/v1/listen"))
        async with asyncio.timeout(2):
            with pytest.raises(ProviderError):
                _ = [update async for update in stt.transcribe(broken_audio())]


async def test_unexpected_stt_disconnect_does_not_commit_an_interim():
    async def handler(socket):
        await socket.recv()
        await socket.send(json.dumps(result("uncertain words")))
        await socket.close()

    async def audio():
        yield b"\0" * 3200
        await asyncio.Event().wait()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        updates = []
        stt = StreamingSTT(Settings(stt_url=f"ws://127.0.0.1:{port}/v1/listen"))
        async with asyncio.timeout(2):
            with pytest.raises(ProviderError):
                async for update in stt.transcribe(audio()):
                    updates.append(update)
        assert all(not update.final for update in updates)
