import asyncio
import io
import wave

import pytest

from bentebot.enrollment import EnrollmentError, EnrollmentTimeoutError, VoiceEnrollment

VOICE = b"\x11\x01" * 320
OTHER = b"\x77\x07" * 320
SILENCE = b"\0" * 640


def pcm(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as value:
        assert value.getframerate() == 16000
        assert value.getnchannels() == 1 and value.getsampwidth() == 2
        return value.readframes(value.getnframes())


async def test_capture_isolates_target_even_with_overlapping_users():
    manager = VoiceEnrollment()
    capture = manager.start(100, 10, "Bente", duration_seconds=5)
    for _ in range(250):
        manager.feed(200, 10, OTHER, True)
        manager.feed(100, 10, VOICE, True)
    result = await capture.finish()
    assert pcm(result) == VOICE * 250
    assert capture.voiced_seconds == 5 and manager.active_count == 0
    assert capture.buffered_bytes == 0
    with pytest.raises(EnrollmentError, match="already been collected"):
        await capture.finish()


async def test_preserves_short_pre_roll_and_actual_pauses():
    manager = VoiceEnrollment(pre_roll_seconds=0.2)
    capture = manager.start(100, 10, "Bente", duration_seconds=6)
    for _ in range(50):
        manager.feed(100, 10, SILENCE, False)
    for _ in range(125):
        manager.feed(100, 10, VOICE, True)
    for _ in range(40):
        manager.feed(100, 10, SILENCE, False)
    for _ in range(125):
        manager.feed(100, 10, VOICE, True)
    assert pcm(await capture.finish()) == SILENCE * 10 + VOICE * 125 + SILENCE * 40 + VOICE * 125
    assert capture.audio_seconds == 6 and capture.voiced_seconds == 5


async def test_target_duration_can_extend_to_meet_voiced_minimum():
    manager = VoiceEnrollment()
    capture = manager.start(100, 10, "Bente", duration_seconds=5)
    for _ in range(10):
        manager.feed(100, 10, SILENCE, False)
    for _ in range(250):
        manager.feed(100, 10, VOICE, True)
    assert len(pcm(await capture.finish())) == 260 * 640
    assert capture.voiced_seconds == 5


async def test_maximum_audio_storage_is_thirty_seconds_and_requires_actual_speech():
    manager = VoiceEnrollment()
    capture = manager.start(100, 10, "Bente", duration_seconds=30)
    manager.feed(100, 10, VOICE, True)
    for _ in range(2000):
        manager.feed(100, 10, SILENCE, False)
        assert capture.buffered_bytes <= 30 * 32000
    assert capture.buffered_bytes == 0 and manager.active_count == 0
    with pytest.raises(EnrollmentError, match="at least five seconds"):
        await capture.finish()


async def test_capture_expiry_erases_pre_roll_without_new_frames():
    manager = VoiceEnrollment(expiry_seconds=0.02)
    capture = manager.start(100, 10, "Bente")
    manager.feed(100, 10, SILENCE, False)
    await asyncio.sleep(0.04)
    assert capture.buffered_bytes == 0 and capture._ready is None and manager.active_count == 0
    with pytest.raises(EnrollmentTimeoutError):
        await capture.finish()


async def test_finish_timeout_and_waiter_cancellation_erase_audio():
    manager = VoiceEnrollment()
    capture = manager.start(100, 10, "Bente")
    manager.feed(100, 10, VOICE, True)
    with pytest.raises(EnrollmentTimeoutError):
        await capture.finish(timeout=0.01)
    assert capture.buffered_bytes == 0 and manager.active_count == 0
    capture = manager.start(100, 10, "Bente")
    manager.feed(100, 10, VOICE, True)
    task = asyncio.create_task(capture.finish())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert capture.buffered_bytes == 0 and capture._ready is None and manager.active_count == 0


async def test_cancel_user_channel_and_all_do_not_cross_sources():
    manager = VoiceEnrollment()
    first = manager.start(100, 10, "First")
    second = manager.start(200, 10, "Second")
    third = manager.start(300, 20, "Third")
    for capture in (first, second, third):
        manager.feed(capture.user_id, capture.channel_id, VOICE, True)
    assert manager.cancel(user_id=100) == 1 and first.buffered_bytes == 0
    assert manager.cancel(channel_id=10) == 1 and second.buffered_bytes == 0
    assert third.buffered_bytes == 640
    assert manager.cancel_all() == 1 and third.buffered_bytes == 0
    for capture in (first, second, third):
        with pytest.raises(asyncio.CancelledError):
            await capture.finish()


async def test_target_moving_to_another_channel_cancels_original_capture():
    manager = VoiceEnrollment()
    capture = manager.start(100, 10, "Bente")
    manager.feed(100, 10, VOICE, True)
    manager.feed(100, 20, OTHER, True)
    assert capture.buffered_bytes == 0 and manager.active_count == 0
    with pytest.raises(asyncio.CancelledError):
        await capture.finish()


async def test_completed_but_uncollected_reference_can_still_be_cancelled():
    manager = VoiceEnrollment()
    capture = manager.start(100, 10, "Bente", duration_seconds=5)
    for _ in range(250):
        manager.feed(100, 10, VOICE, True)
    assert capture.complete and manager.active_count == 1
    manager.cancel(user_id=100)
    assert capture.buffered_bytes == 0 and capture._ready is None
    with pytest.raises(asyncio.CancelledError):
        await capture.finish()


async def test_capacity_duplicate_target_and_invalid_reference_limits():
    manager = VoiceEnrollment(max_captures=1)
    manager.start(100, 10, "Bente")
    try:
        with pytest.raises(EnrollmentError, match="already"):
            manager.start(100, 20, "Again")
        with pytest.raises(EnrollmentError, match="slots"):
            manager.start(200, 10, "Other")
    finally:
        manager.cancel_all()
    for duration in (0, 4.9, 30.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            manager.start(100, 10, "Bente", duration_seconds=duration)
    for name in ("", "\n", "x" * 81):
        with pytest.raises(ValueError):
            manager.start(100, 10, name)


async def test_malformed_audio_fails_only_the_reference_not_other_users():
    manager = VoiceEnrollment()
    first = manager.start(100, 10, "First")
    second = manager.start(200, 10, "Second")
    manager.feed(100, 10, b"bad data", True)
    assert first.buffered_bytes == 0 and manager.get(200) is second
    with pytest.raises(EnrollmentError, match="20ms"):
        await first.finish()
    manager.cancel_all()


async def test_two_overlapping_captures_never_mix_their_targets():
    manager = VoiceEnrollment()
    first = manager.start(100, 10, "First", duration_seconds=5)
    second = manager.start(200, 10, "Second", duration_seconds=5)
    for _ in range(250):
        manager.feed(100, 10, VOICE, True)
        manager.feed(200, 10, OTHER, True)
    assert pcm(await first.finish()) == VOICE * 250
    assert pcm(await second.finish()) == OTHER * 250


async def test_expired_completed_reference_is_not_delivered_after_deadline():
    current = 0.0
    manager = VoiceEnrollment(clock=lambda: current)
    capture = manager.start(100, 10, "Bente", duration_seconds=5)
    for _ in range(250):
        manager.feed(100, 10, VOICE, True)
    current = 61.0
    with pytest.raises(EnrollmentTimeoutError):
        await capture.finish()
    assert capture.buffered_bytes == 0 and manager.active_count == 0
