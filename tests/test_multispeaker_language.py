import asyncio
import json
import struct
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve

from bentebot.config import Settings
from bentebot.domain import Metrics, Transcript
from bentebot.providers import StreamingSTT
from bentebot.voice import Speaker, VoiceReceiver


@pytest.mark.parametrize("maximum,expected", [(1, {100}), (2, {100, 200})])
async def test_noise_does_not_reserve_stt_slots_and_speakers_never_mix(maximum, expected):
    heard, updated = {}, asyncio.Event()

    async def transcript(user_id, name, update):
        heard[user_id] = (name, update.text)
        if expected.issubset(heard):
            updated.set()

    class STT:
        async def transcribe(self, audio):
            async for block in audio:
                if any(block):
                    value = int.from_bytes(block[:2], "little", signed=True)
                    yield Transcript(str(value), final=True)
                    await asyncio.Event().wait()

    conversation = SimpleNamespace(
        metrics=Metrics(),
        settings=SimpleNamespace(max_speakers=maximum),
        partial={},
        speech_activity=lambda user: None,
        transcript=transcript,
    )
    client = SimpleNamespace(listen=lambda sink: None, stop_listening=lambda: None)
    receiver = VoiceReceiver(client, conversation, STT(), None)
    receiver.speakers[300] = Speaker(300, "Noise", vad=SimpleNamespace(is_speech=lambda *args: False))
    receiver.speakers[100] = Speaker(100, "First", vad=SimpleNamespace(is_speech=lambda *args: True))
    receiver.speakers[200] = Speaker(200, "Second", vad=SimpleNamespace(is_speech=lambda *args: True))
    # A noise-only account arrives first and must not occupy the only inference slot.
    for user_id, name, amplitude in [(300, "Noise", 0), (100, "First", 100), (200, "Second", 200)]:
        receiver.incoming.put((user_id, name, struct.pack("<h", amplitude) * 1920))
    receiver.start()
    try:
        await asyncio.wait_for(updated.wait(), 2)
        assert set(heard) == expected
        assert heard[100] == ("First", "100")
        if maximum == 2:
            assert heard[200] == ("Second", "200")
        assert receiver.speakers[300].task is None
    finally:
        await receiver.close()


async def test_danish_override_reaches_actual_websocket_query():
    async def handler(socket):
        assert "language=da" in socket.request.path
        await socket.recv()
        await socket.send(
            '{"type":"Results","is_final":true,"speech_final":true,"start":0,"duration":1,'
            '"channel":{"alternatives":[{"transcript":"Hej med dig"}]}}'
        )
        await socket.recv()  # CloseStream
        await socket.close()

    async def audio():
        yield b"\0" * 3200

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = StreamingSTT(Settings(stt_url=f"ws://127.0.0.1:{port}/v1/listen", stt_language="en"))
        async with asyncio.timeout(3):
            values = [update async for update in stt.transcribe(audio(), language="da")]
        assert values[-1] == Transcript("Hej med dig", final=True)


async def test_english_and_danish_updates_share_one_automatic_stream():
    connections = 0

    async def handler(socket):
        nonlocal connections
        connections += 1
        assert "language=auto" in socket.request.path
        await socket.recv()
        for start, text in [(0, "Hello there"), (1, "Hej med dig")]:
            for final in (False, True):
                await socket.send(
                    json.dumps(
                        {
                            "type": "Results",
                            "is_final": final,
                            "speech_final": final,
                            "start": start,
                            "duration": 1,
                            "channel": {"alternatives": [{"transcript": text}]},
                        }
                    )
                )
        while isinstance(await socket.recv(), bytes):
            pass
        await socket.close()

    async def audio():
        yield b"\0" * 3200

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = StreamingSTT(Settings(stt_url=f"ws://127.0.0.1:{port}/v1/listen"))
        async with asyncio.timeout(3):
            updates = [update async for update in stt.transcribe(audio())]
    assert connections == 1
    assert [update.text for update in updates if update.final] == ["Hello there", "Hej med dig"]
