import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.benchmark import measure
from bentebot.config import Settings
from bentebot.domain import Transcript


async def test_synthetic_benchmark_keeps_content_out_of_results_and_closes_tts(monkeypatch, wav_bytes):
    tts = SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes), close=AsyncMock())
    monkeypatch.setattr("bentebot.benchmark.OpenAITTS", lambda settings: tts)

    class STT:
        async def transcribe(self, audio, *, language):
            assert language == "auto"
            async for block in audio:
                assert len(block) == 3200
                yield Transcript("PRIVATE_TRANSCRIPT")
            yield Transcript("PRIVATE_TRANSCRIPT", final=True)

    monkeypatch.setattr("bentebot.benchmark.StreamingSTT", lambda settings: STT())
    results = await measure(Settings(), "synthetic phrase", repeats=2, padding_seconds=0)
    assert len(results) == 2
    assert results[0]["audio_ms"] > 0 and results[0]["tts_ms"] >= 0
    assert "PRIVATE_TRANSCRIPT" not in str(results)
    assert results[0]["final_after_speech_ms"] is not None
    tts.close.assert_awaited_once()


@pytest.mark.parametrize("repeats", [0, 11])
async def test_benchmark_rejects_unbounded_runs_before_provider_use(repeats):
    with pytest.raises(ValueError, match="runs"):
        await measure(Settings(), "test", repeats=repeats)


async def test_benchmark_closes_tts_when_provider_fails(monkeypatch):
    tts = SimpleNamespace(synthesize=AsyncMock(side_effect=RuntimeError("down")), close=AsyncMock())
    monkeypatch.setattr("bentebot.benchmark.OpenAITTS", lambda settings: tts)
    with pytest.raises(RuntimeError):
        await measure(Settings(), "test")
    tts.close.assert_awaited_once()


@pytest.mark.parametrize("fail", [False, True])
async def test_overlapping_tts_is_measured_and_cancelled_on_stt_failure(monkeypatch, wav_bytes, fail):
    started, ended = asyncio.Event(), asyncio.Event()
    calls = 0

    async def synthesize(text):
        nonlocal calls
        calls += 1
        if calls > 1:
            started.set()
            try:
                if fail:
                    await asyncio.Event().wait()
            finally:
                ended.set()
        return wav_bytes

    tts = SimpleNamespace(synthesize=synthesize, close=AsyncMock())
    monkeypatch.setattr("bentebot.benchmark.OpenAITTS", lambda settings: tts)

    class STT:
        async def transcribe(self, audio, *, language):
            async for _ in audio:
                await started.wait()
                if fail:
                    raise RuntimeError("synthetic failure")
            yield Transcript("synthetic", final=True)

    monkeypatch.setattr("bentebot.benchmark.StreamingSTT", lambda settings: STT())
    if fail:
        with pytest.raises(RuntimeError):
            await measure(Settings(), "test", repeats=1, padding_seconds=0, overlap_tts=True)
    else:
        results = await measure(Settings(), "test", repeats=1, padding_seconds=0, overlap_tts=True)
        assert results[0]["overlap_tts_ms"] >= 0
    assert ended.is_set()
    tts.close.assert_awaited_once()
