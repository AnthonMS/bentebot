import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx

from bentebot.audio_awareness import AudioAwareness
from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy


def settings():
    return Settings(audio_awareness_enabled=True, audio_token="t" * 32, turn_silence_ms=10)


async def test_audio_branch_is_nonblocking_discards_departed_results_and_stays_bounded():
    entered, release = asyncio.Event(), asyncio.Event()

    async def transport(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"events": [{"label": "sneeze", "score": 0.99}], "tone": "unknown"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        audio = AudioAwareness(settings(), asyncio.Event(), client=client)
        for _ in range(150):
            audio.observe(100, b"\x00\x20" * 320, True)
        await entered.wait()
        assert not audio.cues
        for _ in range(900):
            audio.observe(100, b"\x00\x20" * 320, True)
        assert audio.queue.qsize() <= 2 and len(audio.buffers[100]) < 96000
        audio.drop(100)
        release.set()
        await asyncio.sleep(0.02)
        assert not audio.cues and audio.take_reaction() is None
        await audio.close()


async def test_sound_ttl_cooldown_and_tone_require_multiple_windows():
    now = [100.0]
    completed = asyncio.Event()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "events": [{"label": "laugh", "score": 0.97, "injected": "not context"}],
                    "tone": "happy",
                    "tone_score": 0.95,
                },
            )
        )
    ) as client:
        audio = AudioAwareness(settings(), completed, client=client, clock=lambda: now[0])
        audio.revisions[100] = 0
        try:
            for _ in range(2):
                audio.queue.put_nowait((100, 0, b"x" * 96000, True, now[0]))
                if audio.runner is None:
                    audio.runner = asyncio.create_task(audio._run())
                await asyncio.wait_for(completed.wait(), 1)
                completed.clear()
                now[0] += 3
            assert audio.cues[100]["tone"] == "happy"
            prompt = str(audio.context(Audience(frozenset({100}), 1), {100: "Owner"}))
            assert "happy" in prompt and "injected" not in prompt
            assert audio.take_reaction()[:2] == (100, "laugh")
            audio.reaction = (100, "laugh", now[0], 0)
            assert audio.take_reaction() is None
            assert audio.context(Audience(frozenset({200}), 1), {}) == []
            now[0] += 20
            assert audio.context(Audience(frozenset({100}), 1), {}) == []
        finally:
            await audio.close()


async def test_nonverbal_reaction_can_choose_silence_without_fake_transcript(store, wav_bytes):
    source = Source("channel", 10, 1)
    store.save(Event(source, 100, "Owner", "We were discussing the code."))

    async def stream(*args):
        yield "[SILENT]"

    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 8000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        source,
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        AsyncMock(),
        Mock(),
        AsyncMock(),
    )
    try:
        await session._answer(audio_reaction=(100, "laugh", 1.0))
        assert session.metrics.last_outcome == "sound_ignored"
        session.play.assert_not_awaited()
        assert len(store.recent(1, frozenset({100}))) == 1
    finally:
        await session.close()


async def test_sound_only_windows_do_not_extend_old_tone_indefinitely():
    now = [100.0]
    changed = asyncio.Event()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"events": [], "tone": "unknown"}))
    ) as client:
        audio = AudioAwareness(settings(), changed, client=client, clock=lambda: now[0])
        audio.revisions[100] = 0
        audio.cues[100] = {
            "at": 100.0,
            "tone_at": 100.0,
            "tone": "angry",
            "candidate": "angry",
            "tone_score": 0.95,
            "events": [],
        }
        audio.runner = asyncio.create_task(audio._run())
        try:
            for observed in (103.0, 106.0, 109.0, 113.0):
                now[0] = observed
                audio.queue.put_nowait((100, 0, b"x" * 96000, False, observed))
                await asyncio.wait_for(changed.wait(), 1)
                changed.clear()
            assert audio.cues[100]["tone"] == "unknown"
            assert audio.context(Audience(frozenset({100}), 1), {}) == []
        finally:
            await audio.close()


async def test_audio_diagnostics_preserve_weak_candidate_without_treating_it_as_mood_or_sound():
    now = [100.0]
    changed = asyncio.Event()
    responses = [
        {
            "events": [],
            "tone": "unknown",
            "tone_score": 0.62,
            "tone_margin": 0.31,
            "tone_candidate": "angry",
            "tone_status": "low_confidence",
            "speech_score": 0.91,
            "top_sound": {"label": "sneeze", "score": 0.23},
            "device": "cuda",
        },
        {
            "events": [{"label": "sneeze", "score": 0.9}],
            "tone": "unknown",
            "tone_status": "not_requested",
            "top_sound": {"label": "sneeze", "score": 0.9},
        },
        {
            "events": [],
            "tone": "unknown",
            "tone_status": "not_requested",
            "top_sound": {"label": "sigh", "score": 0.02},
        },
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=responses.pop(0)))
    ) as client:
        audio = AudioAwareness(settings(), changed, client=client, clock=lambda: now[0])
        audio.revisions[100] = 0
        audio.runner = asyncio.create_task(audio._run())
        try:
            for tone in (True, False, False):
                audio.queue.put_nowait((100, 0, b"x" * 96000, tone, now[0]))
                await asyncio.wait_for(changed.wait(), 1)
                changed.clear()
                if tone:
                    assert audio.context(Audience(frozenset({100}), 1), {}) == []
                now[0] += 3
            state = audio.speaker_status(100)
            assert state["your_possible_tone"] == "unknown"
            assert state["your_sound_events"] == "none"
            assert state["your_tone_candidate"] == "angry" and state["your_tone_confidence"] == 0.62
            assert state["your_tone_gate"] == "low_confidence"
            assert state["your_strongest_sound"] == "sigh"
            assert state["your_last_confirmed_sound"] == "sneeze"
            assert audio.speaker_status(200) == {}
            assert audio.status()["audio_tone_requests"] == 1
            assert audio.status()["audio_tone_analyses"] == 1
            assert audio.status()["audio_sound_detections"] == 1
            now[0] += 65
            assert audio.speaker_status(100) == {}
            audio.drop(100)
            assert not audio.diagnostics
        finally:
            await audio.close()
