import asyncio
import io
import wave

import pytest

from bentebot.enrollment import EnrollmentError, EnrollmentTimeoutError, VoiceSpeechBank
from bentebot.voice_profiles import KittenVoiceProfiles

VOICE = b"\x11\x01" * 320
LATER = b"\x33\x03" * 320
OTHER = b"\x77\x07" * 320
SILENCE = bytes(640)


def read_pcm(sample):
    with wave.open(io.BytesIO(sample), "rb") as value:
        assert value.getnchannels() == 1 and value.getsampwidth() == 2
        assert value.getframerate() == 16000
        return value.readframes(value.getnframes())


def feed(manager, count, *, user=100, channel=10, pcm=VOICE, voiced=True):
    for _ in range(count):
        manager.feed(user, channel, pcm, voiced)


async def test_collects_forty_five_actual_seconds_across_minutes_of_silence():
    now = 0.0
    manager = VoiceSpeechBank(clock=lambda: now)
    capture = manager.start(100, 10, "Bente")
    feed(manager, 15 * 50)
    assert capture.voiced_seconds == 15
    assert not capture.complete and capture.stats()["target_seconds"] == 45
    now = 135.0
    feed(manager, 120 * 50, pcm=SILENCE, voiced=False)
    assert capture.voiced_seconds == 15 and not capture.complete
    feed(manager, 30 * 50, pcm=LATER)
    assert capture.complete and capture.voiced_seconds == 45
    sample = await capture.finish()
    assert read_pcm(sample) == VOICE * 750
    KittenVoiceProfiles._validate_sample(sample)
    assert capture.buffered_bytes == 0 and manager.active_count == 0
    assert capture.stats()["status"] == "collected"
    with pytest.raises(EnrollmentError, match="already been collected"):
        await capture.finish()


async def test_waits_without_any_audio_and_excludes_other_speakers_entirely():
    now = 0.0
    manager = VoiceSpeechBank(clock=lambda: now)
    capture = manager.start(100, 10, "Target", duration_seconds=30)
    now = 120.0
    feed(manager, 30 * 50, user=200, pcm=OTHER)
    assert capture.voiced_seconds == 0 and capture.buffered_bytes == 0
    for _ in range(30 * 50):
        manager.feed(100, 10, VOICE, True)
        manager.feed(200, 10, OTHER, True)
    assert capture.complete
    assert read_pcm(await capture.finish()) == VOICE * 750


async def test_short_clicks_never_add_usable_speech_or_survive_in_reference():
    manager = VoiceSpeechBank(pre_roll_seconds=0)
    capture = manager.start(100, 10, "Target", duration_seconds=30)
    for _ in range(100):
        feed(manager, 4, pcm=OTHER)
        feed(manager, 20, pcm=SILENCE, voiced=False)
    assert capture.voiced_seconds == 0 and not capture.complete
    feed(manager, 30 * 50)
    assert capture.voiced_seconds == 30
    assert OTHER not in read_pcm(await capture.finish())


async def test_small_pauses_preserve_original_audio_without_counting_as_speech():
    manager = VoiceSpeechBank(pre_roll_seconds=0)
    capture = manager.start(100, 10, "Target", duration_seconds=30)
    feed(manager, 125)
    feed(manager, 10, pcm=SILENCE, voiced=False)
    feed(manager, 125, pcm=LATER)
    # The five voiced seconds make a 5.2s original reference, not a spliced 5s file.
    assert capture.reference_seconds == 5.2
    feed(manager, 100, pcm=SILENCE, voiced=False)
    # Repeated short utterances satisfy the bank but cannot replace that reference.
    for _ in range(25):
        feed(manager, 50, pcm=OTHER)
        feed(manager, 20, pcm=SILENCE, voiced=False)
    assert capture.complete and capture.voiced_seconds == 30
    assert read_pcm(await capture.finish()) == VOICE * 125 + SILENCE * 10 + LATER * 125


async def test_missing_receive_gap_does_not_join_unrelated_utterances():
    now = 0.0
    manager = VoiceSpeechBank(clock=lambda: now, pre_roll_seconds=0)
    capture = manager.start(100, 10, "Target", duration_seconds=30)
    feed(manager, 4 * 50)
    now += 120
    feed(manager, 4 * 50, pcm=LATER)
    assert capture.reference_seconds == 0
    now += 1
    feed(manager, 22 * 50, pcm=OTHER)
    assert capture.complete and capture.voiced_seconds == 30
    assert read_pcm(await capture.finish()) == OTHER * 750


async def test_short_utterances_complete_at_target_with_labelled_assembled_reference():
    manager = VoiceSpeechBank(pre_roll_seconds=0)
    capture = manager.start(100, 10, "Target", duration_seconds=30)
    for _ in range(30):
        feed(manager, 50)
        feed(manager, 20, pcm=SILENCE, voiced=False)
    assert capture.voiced_seconds == 30 and capture.complete
    assert capture.reference_spliced and capture.reference_segment_count > 1
    sample = await capture.finish()
    audio = read_pcm(sample)
    separator = SILENCE * 8
    assert separator in audio
    assert 5 * 32000 <= len(audio) <= 15 * 32000
    assert capture.stats()["reference_spliced"] is True
    KittenVoiceProfiles._validate_sample(sample)


async def test_twenty_three_two_second_phrases_finish_by_forty_five_voiced_seconds():
    now = 0.0
    manager = VoiceSpeechBank(clock=lambda: now, pre_roll_seconds=0)
    capture = manager.start(100, 10, "Target")
    index = 0
    for _ in range(23):
        for _ in range(100):
            index += 1
            manager.feed(100, 10, index.to_bytes(2, "little") * 320, True)
        now += 20
    assert capture.complete and capture.voiced_seconds == 45
    assert capture.reference_spliced
    audio = read_pcm(await capture.finish())
    # Silence makes boundaries audible; no other account or duplicated padding
    # pretends to contribute to the forty-five seconds of speech.
    voice_frames = [
        audio[offset : offset + 640]
        for offset in range(0, len(audio), 640)
        if audio[offset : offset + 640] != SILENCE
    ]
    assert len(voice_frames) >= 250
    assert len(set(voice_frames)) == len(voice_frames)
    assert all(0 < int.from_bytes(frame[:2], "little") <= 2250 for frame in voice_frames)
    assert audio.count(SILENCE * 8) == capture.reference_segment_count - 1
    assert 5 * 32000 <= len(audio) <= 15 * 32000


async def test_minimum_qualified_runs_also_produce_a_bounded_reference_at_target():
    manager = VoiceSpeechBank(pre_roll_seconds=0)
    capture = manager.start(100, 10, "Target", duration_seconds=30)
    for _ in range(300):
        feed(manager, 5)
        feed(manager, 20, pcm=SILENCE, voiced=False)
        assert capture.buffered_bytes < 1_500_000
    assert capture.complete and capture.voiced_seconds == 30
    assert capture.reference_spliced
    sample = await capture.finish()
    KittenVoiceProfiles._validate_sample(sample)


async def test_retained_pcm_is_bounded_during_a_long_capture():
    manager = VoiceSpeechBank()
    capture = manager.start(100, 10, "Target", duration_seconds=60)
    for _ in range(4000):
        manager.feed(100, 10, VOICE, True)
        assert capture.buffered_bytes < 1_000_000
    assert capture.voiced_seconds == 60 and capture.complete
    assert len(read_pcm(await capture.finish())) == 750 * 640


async def test_expiry_purges_an_uncollected_completed_bank():
    now = 0.0
    manager = VoiceSpeechBank(clock=lambda: now)
    capture = manager.start(100, 10, "Target", duration_seconds=30)
    feed(manager, 30 * 50)
    assert capture.complete
    now = 901.0
    with pytest.raises(EnrollmentTimeoutError):
        await capture.finish()
    assert capture.buffered_bytes == 0 and manager.active_count == 0


async def test_timer_expires_without_new_frames_or_a_collector():
    manager = VoiceSpeechBank(expiry_seconds=0.01)
    capture = manager.start(100, 10, "Target")
    feed(manager, 10)
    await asyncio.sleep(0.03)
    assert capture.buffered_bytes == 0 and manager.active_count == 0
    with pytest.raises(EnrollmentTimeoutError):
        await capture.finish()


async def test_cancelling_one_capture_does_not_affect_another():
    manager = VoiceSpeechBank()
    first = manager.start(100, 10, "First")
    second = manager.start(200, 20, "Second")
    feed(manager, 10)
    feed(manager, 10, user=200, channel=20)
    assert manager.cancel(channel_id=10) == 1
    assert first.buffered_bytes == 0 and second.buffered_bytes > 0
    with pytest.raises(asyncio.CancelledError):
        await first.finish()
    assert manager.cancel_all() == 1 and second.buffered_bytes == 0


async def test_moving_channel_cancels_target_and_discards_audio():
    manager = VoiceSpeechBank()
    capture = manager.start(100, 10, "Target")
    feed(manager, 20)
    manager.feed(100, 20, OTHER, True)
    assert capture.buffered_bytes == 0 and manager.active_count == 0
    with pytest.raises(asyncio.CancelledError):
        await capture.finish()


async def test_collector_timeout_cancellation_and_duplicate_waiter_cleanup():
    manager = VoiceSpeechBank()
    capture = manager.start(100, 10, "Target")
    with pytest.raises(EnrollmentTimeoutError):
        await capture.finish(timeout=0)
    assert capture.buffered_bytes == 0 and manager.active_count == 0
    capture = manager.start(100, 10, "Target")
    collector = asyncio.create_task(capture.finish())
    await asyncio.sleep(0)
    with pytest.raises(EnrollmentError, match="waiting collector"):
        await capture.finish()
    collector.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collector
    assert capture.buffered_bytes == 0 and manager.active_count == 0


async def test_capacity_duplicate_and_invalid_duration_limits():
    manager = VoiceSpeechBank()
    manager.start(100, 10, "First")
    for user in (200, 300, 400):
        manager.start(user, 10, "Other")
    with pytest.raises(EnrollmentError, match="already"):
        manager.start(100, 10, "Again")
    with pytest.raises(EnrollmentError, match="slots"):
        manager.start(500, 10, "Fifth")
    manager.cancel_all()
    for duration in (0, 29.9, 60.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            manager.start(100, 10, "Target", duration_seconds=duration)
    for kwargs in ({"max_captures": 5}, {"expiry_seconds": 901}, {"pre_roll_seconds": 1}):
        with pytest.raises(ValueError):
            VoiceSpeechBank(**kwargs)


async def test_bad_frames_fail_only_the_affected_bank():
    manager = VoiceSpeechBank()
    first = manager.start(100, 10, "First")
    second = manager.start(200, 10, "Second")
    manager.feed(100, 10, b"bad", True)
    assert first.buffered_bytes == 0 and manager.get(200) is second
    with pytest.raises(EnrollmentError, match="20ms"):
        await first.finish()
    manager.cancel_all()
