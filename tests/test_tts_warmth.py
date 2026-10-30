import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.config import Settings
from bentebot.tts_warmth import VoiceTTSWarmth


async def test_tts_warm_is_shared_discarded_and_released(wav_bytes):
    tts = SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes), recently_used=lambda *_: False)
    manager = VoiceTTSWarmth(Settings(tts_base_url="http://192.168.50.10:18768/v1", tts_model="kokoro"), tts)
    try:
        state = manager.select(1, "af_heart")
        assert manager.select(2, "af_heart") is state
        await manager.prepare(1, "af_heart")
        tts.synthesize.assert_awaited_once_with("Ready.", "af_heart")
        assert "TTS prewarming: ready" in manager.describe(1)
        await manager.release(1)
        assert not state.task.done()
        await manager.release(2)
        assert state.task.cancelled()
    finally:
        await manager.close()


@pytest.mark.parametrize("url,enabled", [("https://api.example.com/v1", True), ("http://tts/v1", False)])
async def test_cloud_or_disabled_tts_does_not_warm(url, enabled):
    tts = SimpleNamespace(synthesize=AsyncMock(), recently_used=lambda *_: False)
    manager = VoiceTTSWarmth(Settings(tts_base_url=url, tts_model="kokoro", tts_warm_enabled=enabled), tts)
    try:
        await manager.prepare(1, "af_heart")
        tts.synthesize.assert_not_awaited()
    finally:
        await manager.close()


async def test_first_sentence_waits_for_warmup_and_failure_releases_gate():
    started, finish = asyncio.Event(), asyncio.Event()

    async def synthesize(*_):
        started.set()
        await finish.wait()
        raise RuntimeError("unavailable")

    tts = SimpleNamespace(synthesize=synthesize, recently_used=lambda *_: False)
    manager = VoiceTTSWarmth(Settings(tts_base_url="http://tts/v1", tts_model="kokoro"), tts)
    task = asyncio.create_task(manager.prepare(1, "af_heart"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert not task.done()
        finish.set()
        await asyncio.wait_for(task, 1)
        assert manager.models["af_heart"].status == "failed"
    finally:
        await manager.close()
