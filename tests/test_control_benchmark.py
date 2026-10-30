import asyncio
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot import control_benchmark as benchmark
from bentebot.config import Settings
from bentebot.domain import Transcript


async def test_control_benchmark_compares_hints_and_intent_without_discord_actions(monkeypatch, wav_bytes):
    current = ""

    async def synthesize(phrase):
        nonlocal current
        current = phrase
        return wav_bytes

    tts = SimpleNamespace(synthesize=synthesize, close=AsyncMock())
    chat = SimpleNamespace(resolve_spoken_control=AsyncMock(return_value="leave"), close=AsyncMock())
    monkeypatch.setattr(benchmark, "OpenAITTS", lambda settings: tts)
    monkeypatch.setattr(benchmark, "OpenAIChat", lambda settings: chat)
    hints = []

    class STT:
        def __init__(self, settings):
            hints.append(settings.stt_context)

        async def transcribe(self, audio):
            async with aclosing(audio):
                await anext(audio)
            yield Transcript(
                "Please leave the call bend a bar" if "leave" in current else current, final=True
            )

    monkeypatch.setattr(benchmark, "StreamingSTT", STT)
    rows = await benchmark.qualify(Settings(), compare_hints=True, include_intent=True)
    assert len(rows) == 8 and all(row["passed"] for row in rows)
    assert hints == ["", "Bentebot, Bente."] * 4
    assert chat.resolve_spoken_control.await_count == 2
    tts.close.assert_awaited_once()
    chat.close.assert_awaited_once()


@pytest.mark.parametrize("cancel", [False, True])
async def test_control_benchmark_stops_on_provider_failure_and_closes_clients(monkeypatch, wav_bytes, cancel):
    tts = SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes), close=AsyncMock())
    chat = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(benchmark, "OpenAITTS", lambda settings: tts)
    monkeypatch.setattr(benchmark, "OpenAIChat", lambda settings: chat)

    class STT:
        async def transcribe(self, audio):
            if cancel:
                raise asyncio.CancelledError()
            raise RuntimeError("PRIVATE_PROVIDER_BODY")
            yield

    monkeypatch.setattr(benchmark, "StreamingSTT", lambda settings: STT())
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await benchmark.qualify(Settings(), include_intent=True)
    else:
        rows = await benchmark.qualify(Settings(), include_intent=True)
        assert len(rows) == 1 and rows[0]["error"] == "RuntimeError"
        assert "PRIVATE_PROVIDER_BODY" not in str(rows)
        tts.synthesize.assert_awaited_once()
    tts.close.assert_awaited_once()
    chat.close.assert_awaited_once()
