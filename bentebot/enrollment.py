"""Short, isolated voice-reference capture. Consent and provider upload belong to callers.

Feed this manager on its asyncio loop with chronological 20ms, 16kHz mono s16le frames.
Supply silence frames for receive gaps so brief pauses retain their original duration.
There are no filesystem writes, API calls, model imports, or cross-user audio mixes.
"""

from __future__ import annotations

import asyncio
import logging
import math
import struct
import time
from collections import deque
from collections.abc import Callable

_FRAME_BYTES = 640
_FRAMES_PER_SECOND = 50
_MIN_VOICED_FRAMES = 5 * _FRAMES_PER_SECOND
_MAX_FRAMES = 30 * _FRAMES_PER_SECOND

log = logging.getLogger(__name__)


class EnrollmentError(ValueError):
    """The reference could not be collected safely or did not meet speech requirements."""


class EnrollmentTimeoutError(TimeoutError):
    """A reference was not collected before its deadline."""


class VoiceCapture:
    """One user's reference; finish returns it once and discards internal audio."""

    def __init__(self, manager: VoiceEnrollment, user_id: int, channel_id: int, name: str, duration: float):
        self.user_id, self.channel_id, self.name = user_id, channel_id, name
        self.duration_seconds = duration
        self._manager = manager
        self._target_frames = math.ceil(duration * _FRAMES_PER_SECOND)
        self._audio = bytearray()
        self._pre_roll: deque[bytes] = deque(maxlen=manager.pre_roll_frames)
        self._frames = 0
        self._voiced_frames = 0
        self._started = False
        self._complete = False
        self._collected = False
        self._finishing = False
        self._error: BaseException | None = None
        self._ready: asyncio.Future[None] | None = manager.loop.create_future()
        self._deadline = manager.clock() + manager.expiry_seconds
        self._timer = manager.loop.call_later(manager.expiry_seconds, self._expire)

    @property
    def audio_seconds(self) -> float:
        return self._frames / _FRAMES_PER_SECOND

    @property
    def voiced_seconds(self) -> float:
        return self._voiced_frames / _FRAMES_PER_SECOND

    @property
    def buffered_bytes(self) -> int:
        return len(self._audio) + sum(len(frame) for frame in self._pre_roll)

    @property
    def complete(self) -> bool:
        return self._complete and self._error is None and not self._collected

    def _erase(self) -> None:
        self._audio.clear()
        self._pre_roll.clear()

    def _release(self) -> None:
        self._timer.cancel()
        if self._manager.get(self.user_id) is self:
            self._manager._captures.pop(self.user_id, None)

    def _fail(self, error: BaseException) -> None:
        if self._collected or self._error is not None:
            return
        self._error = error
        self._erase()
        ready, self._ready = self._ready, None
        if ready is not None and not ready.done():
            if isinstance(error, asyncio.CancelledError):
                ready.cancel()
            else:
                # Errors are raised by finish, not stored as unobserved Future exceptions.
                ready.set_result(None)
        self._release()

    def _expire(self) -> None:
        self._fail(EnrollmentTimeoutError("Voice-reference capture expired before collection."))

    def cancel(self) -> None:
        self._fail(asyncio.CancelledError("Voice-reference capture cancelled."))

    def _feed(self, pcm: bytes, voiced: bool) -> None:
        if self._complete or self._error is not None or self._collected:
            return
        if self._manager.clock() >= self._deadline:
            self._expire()
            return
        if len(pcm) != _FRAME_BYTES:
            self._fail(EnrollmentError("Reference audio requires 20ms, 16kHz mono PCM frames."))
            return
        if not self._started:
            if not voiced:
                self._pre_roll.append(bytes(pcm))
                return
            self._started = True
            for frame in self._pre_roll:
                self._audio.extend(frame)
                self._frames += 1
            self._pre_roll.clear()
        self._audio.extend(pcm)
        self._frames += 1
        self._voiced_frames += bool(voiced)
        enough_speech = self._voiced_frames >= _MIN_VOICED_FRAMES
        if self._frames >= self._target_frames and enough_speech:
            self._complete = True
            if self._ready is not None and not self._ready.done():
                self._ready.set_result(None)
        elif self._frames >= _MAX_FRAMES:
            self._fail(EnrollmentError("Speak for at least five seconds within a thirty-second reference."))

    async def finish(self, timeout: float = 60) -> bytes:  # noqa: ASYNC109 - public bounded capture API
        """Wait for a usable reference and return a PCM WAV exactly once.

        The configured duration is a target; capture may extend to obtain five seconds
        of actual voiced input, but retained audio can never exceed thirty seconds.
        Cancelling this await also cancels capture. Returned immutable bytes belong to
        the caller and cannot subsequently be revoked by the manager.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Capture wait timeout must be finite and nonnegative")
        if self._finishing:
            raise EnrollmentError("This reference already has a waiting collector.")
        if self._collected:
            raise EnrollmentError("This reference has already been collected.")
        if self._manager.clock() >= self._deadline:
            self._expire()
        if self._error is not None:
            raise self._error
        ready = self._ready
        if ready is None:
            raise EnrollmentError("This reference is no longer available.")
        self._finishing = True
        try:
            try:
                await asyncio.wait_for(asyncio.shield(ready), timeout=timeout)
            except TimeoutError:
                self._fail(EnrollmentTimeoutError("Timed out waiting for enough reference speech."))
            except asyncio.CancelledError:
                self.cancel()
                raise
            if self._manager.clock() >= self._deadline:
                self._expire()
            if self._error is not None:
                raise self._error
            if not self._complete or not _MIN_VOICED_FRAMES <= self._voiced_frames <= _MAX_FRAMES:
                raise EnrollmentError("The reference does not contain enough actual speech.")
            if not _MIN_VOICED_FRAMES <= self._frames <= _MAX_FRAMES:
                raise EnrollmentError("The reference must contain five to thirty seconds of audio.")
            size = len(self._audio)
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + size,
                b"WAVE",
                b"fmt ",
                16,
                1,
                1,
                16000,
                32000,
                2,
                16,
                b"data",
                size,
            )
            wav = header + self._audio
            self._collected = True
            self._ready = None
            self._erase()
            self._release()
            return wav
        finally:
            self._finishing = False


class VoiceEnrollment:
    """Bounded per-user captures. Start only after the target user's explicit consent."""

    def __init__(
        self,
        *,
        expiry_seconds: float = 60,
        pre_roll_seconds: float = 0.2,
        max_captures: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not math.isfinite(expiry_seconds) or expiry_seconds <= 0:
            raise ValueError("Capture expiry must be finite and positive")
        if not math.isfinite(pre_roll_seconds) or not 0 <= pre_roll_seconds <= 1:
            raise ValueError("Reference pre-roll must be between zero and one second")
        if max_captures < 1:
            raise ValueError("At least one reference-capture slot is required")
        self.loop = asyncio.get_running_loop()
        self.clock = clock
        self.expiry_seconds = expiry_seconds
        self.pre_roll_frames = math.ceil(pre_roll_seconds * _FRAMES_PER_SECOND)
        self.max_captures = max_captures
        self._captures: dict[int, VoiceCapture] = {}

    def get(self, user_id: int) -> VoiceCapture | None:
        return self._captures.get(user_id)

    @property
    def active_count(self) -> int:
        return len(self._captures)

    def start(self, user_id: int, channel_id: int, name: str, duration_seconds: float = 15) -> VoiceCapture:
        if user_id <= 0 or channel_id <= 0:
            raise ValueError("Reference capture requires valid user and channel IDs")
        name = name.strip()
        if not name or len(name) > 80 or not name.isprintable():
            raise ValueError("Voice-profile name must contain one to eighty printable characters")
        if not math.isfinite(duration_seconds) or not 5 <= duration_seconds <= 30:
            raise ValueError("Reference duration must be between five and thirty seconds")
        # Reap expired references even if a test/fake clock advanced ahead of real timers.
        for capture in list(self._captures.values()):
            if self.clock() >= capture._deadline:
                capture._expire()
        if user_id in self._captures:
            raise EnrollmentError("This user already has a voice-reference capture in progress.")
        if len(self._captures) >= self.max_captures:
            raise EnrollmentError("All voice-reference capture slots are occupied.")
        capture = VoiceCapture(self, user_id, channel_id, name, duration_seconds)
        self._captures[user_id] = capture
        return capture

    def feed(self, user_id: int, channel_id: int, pcm16k20ms: bytes, voiced: bool) -> None:
        capture = self._captures.get(user_id)
        if capture is None:
            return
        if capture.channel_id != channel_id:
            capture.cancel()
            return
        capture._feed(pcm16k20ms, voiced)

    def cancel(self, *, user_id: int | None = None, channel_id: int | None = None) -> int:
        """Cancel one user, one channel, their intersection, or all captures if unspecified."""
        captures = [
            capture
            for capture in self._captures.values()
            if (user_id is None or capture.user_id == user_id)
            and (channel_id is None or capture.channel_id == channel_id)
        ]
        for capture in captures:
            capture.cancel()
        return len(captures)

    def cancel_all(self) -> int:
        return self.cancel()


class SpeechBankCapture:
    """Collect voiced duration across pauses and keep the best continuous reference.

    VAD runs shorter than 100ms do not count. A reference contains at least five
    voiced seconds and at most fifteen seconds of original continuous audio.
    Pauses longer than 300ms start another candidate. If no suitable continuous
    reference exists when the voiced target is met, qualified same-speaker runs
    are assembled with explicit 160ms silence boundaries. The full forty-five
    seconds need not be retained by the external provider.
    """

    _MIN_RUN = 5
    _MIN_REFERENCE = 5 * _FRAMES_PER_SECOND
    _MAX_REFERENCE = 15 * _FRAMES_PER_SECOND
    _MAX_GAP = 15
    _REFERENCE_SEPARATOR = 8

    def __init__(self, manager: VoiceSpeechBank, user_id: int, channel_id: int, name: str, duration: float):
        self.user_id, self.channel_id, self.name = user_id, channel_id, name
        self.duration_seconds = duration
        self._manager = manager
        self._target_frames = math.ceil(duration * _FRAMES_PER_SECOND)
        self._voiced_frames = 0
        self._audio_frames = 0
        self._candidate: deque[tuple[bytes, bool]] = deque()
        self._candidate_voiced = 0
        self._best: tuple[bytes, ...] = ()
        self._best_score = (0, 0)
        self._reference_frames = 0
        self._reference_segment_count = 0
        self._fallback: deque[tuple[bytes, ...]] = deque()
        self._fallback_run: deque[bytes] = deque(maxlen=self._MAX_REFERENCE)
        self._fallback_frames = 0
        self._pre_roll: deque[bytes] = deque(maxlen=manager.pre_roll_frames)
        self._gap: deque[bytes] = deque()
        self._pending: list[bytes] = []
        self._run_confirmed = False
        self._last_frame_at: float | None = None
        self._next_progress = 5.0
        self._complete = False
        self._collected = False
        self._finishing = False
        self._error: BaseException | None = None
        self._ready: asyncio.Future[None] | None = manager.loop.create_future()
        self._deadline = manager.clock() + manager.expiry_seconds
        self._timer = manager.loop.call_later(manager.expiry_seconds, self._expire)

    @property
    def voiced_seconds(self) -> float:
        return self._voiced_frames / _FRAMES_PER_SECOND

    @property
    def audio_seconds(self) -> float:
        """Total accepted candidate audio, including brief preserved pauses."""
        return self._audio_frames / _FRAMES_PER_SECOND

    @property
    def reference_seconds(self) -> float:
        return self._reference_frames / _FRAMES_PER_SECOND

    @property
    def reference_segment_count(self) -> int:
        return self._reference_segment_count

    @property
    def reference_spliced(self) -> bool:
        return self._reference_segment_count > 1

    @property
    def buffered_bytes(self) -> int:
        # Deliberately count shared immutable frame references twice. This is an
        # upper bound on retained PCM, not a fragile estimate of Python allocations.
        return _FRAME_BYTES * (
            len(self._candidate)
            + len(self._best)
            + len(self._pre_roll)
            + len(self._gap)
            + len(self._pending)
            + sum(len(segment) for segment in self._fallback)
            + len(self._fallback_run)
        )

    @property
    def complete(self) -> bool:
        return self._complete and self._error is None and not self._collected

    @property
    def status(self) -> str:
        if self._collected:
            return "collected"
        if self._error is not None:
            return "cancelled" if isinstance(self._error, asyncio.CancelledError) else "failed"
        if self._complete:
            return "ready"
        if self._voiced_frames >= self._target_frames:
            return "waiting_for_continuous_reference"
        return "collecting" if self._voiced_frames else "waiting_for_speech"

    def stats(self) -> dict:
        return {
            "status": self.status,
            "target_seconds": self.duration_seconds,
            "voiced_seconds": round(self.voiced_seconds, 2),
            "audio_seconds": round(self.audio_seconds, 2),
            "reference_seconds": round(self.reference_seconds, 2),
            "reference_segment_count": self.reference_segment_count,
            "reference_spliced": self.reference_spliced,
            "buffered_bytes": self.buffered_bytes,
            "expires_in": max(0, round(self._deadline - self._manager.clock(), 1)),
        }

    def _erase(self) -> None:
        self._candidate.clear()
        self._candidate_voiced = 0
        self._best = ()
        self._pre_roll.clear()
        self._gap.clear()
        self._pending.clear()
        self._fallback.clear()
        self._fallback_run.clear()
        self._fallback_frames = 0

    def _release(self) -> None:
        self._timer.cancel()
        if self._manager.get(self.user_id) is self:
            self._manager._captures.pop(self.user_id, None)

    def _fail(self, error: BaseException) -> None:
        if self._collected or self._error is not None:
            return
        self._error = error
        self._erase()
        ready, self._ready = self._ready, None
        if ready is not None and not ready.done():
            if isinstance(error, asyncio.CancelledError):
                ready.cancel()
            else:
                ready.set_result(None)
        self._release()
        log.info(
            "voice_reference_bank_ended user=%s channel=%s outcome=%s voiced_seconds=%.2f",
            self.user_id,
            self.channel_id,
            type(error).__name__,
            self.voiced_seconds,
        )

    def _expire(self) -> None:
        self._fail(
            EnrollmentTimeoutError("Voice collection expired before enough usable speech was collected.")
        )

    def cancel(self) -> None:
        self._fail(asyncio.CancelledError("Voice-reference collection cancelled."))

    def _consider_reference(self) -> None:
        # Every retained candidate is a continuous slice of one utterance, with
        # only original short pauses. Do not turn minutes of silence into splices.
        score = (self._candidate_voiced, -len(self._candidate))
        if (
            self._candidate_voiced >= self._MIN_REFERENCE
            and self._candidate_voiced >= len(self._candidate) * 0.6
            and score > self._best_score
        ):
            self._best = tuple(frame for frame, _ in self._candidate)
            self._best_score = score
            self._reference_frames = len(self._best)
            self._reference_segment_count = 1
        # A clean run can become suitable before earlier pauses roll out of the
        # candidate window. Prefer its real continuous waveform to any splice.
        if len(self._fallback_run) >= self._MIN_REFERENCE:
            score = (len(self._fallback_run), -len(self._fallback_run))
            if score > self._best_score:
                self._best = tuple(self._fallback_run)
                self._best_score = score
                self._reference_frames = len(self._best)
                self._reference_segment_count = 1
        if self._best:
            self._fallback.clear()
            self._fallback_run.clear()
            self._fallback_frames = 0

    def _finish_run(self) -> None:
        if self._fallback_run and not self._best:
            segment = tuple(self._fallback_run)
            self._fallback.append(segment)
            self._fallback_frames += len(segment)
            while self._fallback and (
                self._fallback_frames + self._REFERENCE_SEPARATOR * (len(self._fallback) - 1)
                > self._MAX_REFERENCE
            ):
                self._fallback_frames -= len(self._fallback.popleft())
        self._fallback_run.clear()

    def _assemble_reference(self) -> None:
        if self._best:
            return
        self._finish_run()
        if self._fallback_frames < self._MIN_REFERENCE:
            return
        separator = (bytes(_FRAME_BYTES),) * self._REFERENCE_SEPARATOR
        frames: list[bytes] = []
        for segment in self._fallback:
            if frames:
                frames.extend(separator)
            frames.extend(segment)
        self._best = tuple(frames)
        self._reference_frames = len(self._best)
        self._reference_segment_count = len(self._fallback)

    def _end_segment(self) -> None:
        self._consider_reference()
        self._finish_run()
        self._candidate.clear()
        self._candidate_voiced = 0
        self._gap.clear()
        self._pending.clear()
        self._run_confirmed = False

    def _push(self, pcm: bytes, voiced: bool) -> None:
        if len(self._candidate) == self._MAX_REFERENCE:
            _, previous_voiced = self._candidate.popleft()
            self._candidate_voiced -= previous_voiced
        self._candidate.append((pcm, voiced))
        self._candidate_voiced += voiced
        self._audio_frames += 1

    def _silence(self, pcm: bytes) -> None:
        self._pre_roll.append(pcm)
        if self._candidate:
            self._gap.append(pcm)
            if len(self._gap) > self._MAX_GAP:
                self._end_segment()

    def _feed(self, pcm: bytes, voiced: bool) -> None:
        if self._complete or self._error is not None or self._collected:
            return
        now = self._manager.clock()
        if now >= self._deadline:
            self._expire()
            return
        if not isinstance(pcm, (bytes, bytearray)) or len(pcm) != _FRAME_BYTES:
            self._fail(EnrollmentError("Reference audio requires 20ms, 16kHz mono PCM frames."))
            return
        if self._last_frame_at is not None:
            elapsed = now - self._last_frame_at
            if elapsed > 0.35:
                self._end_segment()
                self._pre_roll.clear()
            elif elapsed > 0.04:
                # Some transports omit receive gaps. Preserve short missing gaps
                # with silence; never make separated speech appear consecutive.
                self._finish_run()
                self._run_confirmed = False
                self._pending.clear()
                for _ in range(min(self._MAX_GAP + 1, max(0, round(elapsed * 50) - 1))):
                    self._silence(bytes(_FRAME_BYTES))
        self._last_frame_at = now
        pcm = bytes(pcm)
        if voiced:
            if self._run_confirmed:
                self._push(pcm, True)
                if not self._best:
                    self._fallback_run.append(pcm)
                self._voiced_frames += 1
            else:
                self._pending.append(pcm)
                if len(self._pending) < self._MIN_RUN:
                    return
                pause = self._gap if self._candidate else self._pre_roll
                for frame in pause:
                    self._push(frame, False)
                self._gap.clear()
                self._pre_roll.clear()
                for frame in self._pending:
                    self._push(frame, True)
                    if not self._best:
                        self._fallback_run.append(frame)
                self._voiced_frames += len(self._pending)
                self._pending.clear()
                self._run_confirmed = True
            if self._voiced_frames % 25 == 0 or self._voiced_frames >= self._target_frames:
                self._consider_reference()
            if self._voiced_frames >= self._target_frames:
                self._assemble_reference()
            if self.voiced_seconds >= self._next_progress:
                log.info(
                    "voice_reference_bank_progress user=%s channel=%s voiced_seconds=%.2f target_seconds=%.2f",
                    self.user_id,
                    self.channel_id,
                    self.voiced_seconds,
                    self.duration_seconds,
                )
                self._next_progress = self.voiced_seconds + 5
            if self._voiced_frames >= self._target_frames and self._best:
                self._complete = True
                if self._ready is not None and not self._ready.done():
                    self._ready.set_result(None)
                log.info(
                    "voice_reference_bank_ready user=%s channel=%s voiced_seconds=%.2f reference_seconds=%.2f",
                    self.user_id,
                    self.channel_id,
                    self.voiced_seconds,
                    self.reference_seconds,
                )
                # All unused candidates disappear as soon as the provider-sized
                # reference is ready. Keep metadata, not the rest of the bank.
                self._candidate.clear()
                self._candidate_voiced = 0
                self._pre_roll.clear()
                self._gap.clear()
                self._pending.clear()
                self._fallback.clear()
                self._fallback_run.clear()
                self._fallback_frames = 0
        else:
            if self._pending:
                # Discard tiny VAD bursts instead of counting clicks as speech or
                # including their waveform in a clean voice reference.
                for _ in self._pending:
                    self._silence(bytes(_FRAME_BYTES))
                self._pending.clear()
            self._run_confirmed = False
            self._finish_run()
            self._silence(pcm)

    async def finish(self, timeout: float = 900) -> bytes:  # noqa: ASYNC109 - bounded public API
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Capture wait timeout must be finite and nonnegative")
        if self._finishing:
            raise EnrollmentError("This reference already has a waiting collector.")
        if self._collected:
            raise EnrollmentError("This reference has already been collected.")
        if self._manager.clock() >= self._deadline:
            self._expire()
        if self._error is not None:
            raise self._error
        ready = self._ready
        if ready is None:
            raise EnrollmentError("This reference is no longer available.")
        self._finishing = True
        try:
            try:
                await asyncio.wait_for(asyncio.shield(ready), timeout=timeout)
            except TimeoutError:
                self._fail(EnrollmentTimeoutError("Timed out waiting for enough usable reference speech."))
            except asyncio.CancelledError:
                self.cancel()
                raise
            if self._manager.clock() >= self._deadline:
                self._expire()
            if self._error is not None:
                raise self._error
            if not self._complete or not self._MIN_REFERENCE <= len(self._best) <= self._MAX_REFERENCE:
                raise EnrollmentError("The reference does not contain enough continuous speech.")
            pcm = b"".join(self._best)
            size = len(pcm)
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + size,
                b"WAVE",
                b"fmt ",
                16,
                1,
                1,
                16000,
                32000,
                2,
                16,
                b"data",
                size,
            )
            self._collected = True
            self._ready = None
            self._erase()
            self._release()
            return header + pcm
        finally:
            self._finishing = False


class VoiceSpeechBank:
    """At most four live target-speaker collectors, each retaining under 1.5MB PCM.

    Feed after speaker separation and VAD. Permission and provider checks remain
    the caller's responsibility. No STT suppression or playback pause is required.
    """

    def __init__(
        self,
        *,
        expiry_seconds: float = 900,
        pre_roll_seconds: float = 0.2,
        max_captures: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not math.isfinite(expiry_seconds) or not 0 < expiry_seconds <= 900:
            raise ValueError("Collection expiry must be between zero and fifteen minutes")
        if not math.isfinite(pre_roll_seconds) or not 0 <= pre_roll_seconds <= 0.2:
            raise ValueError("Reference pre-roll must be between zero and 0.2 seconds")
        if not 1 <= max_captures <= 4:
            raise ValueError("Choose between one and four voice-collection slots")
        self.loop = asyncio.get_running_loop()
        self.clock, self.expiry_seconds = clock, expiry_seconds
        self.pre_roll_frames = math.ceil(pre_roll_seconds * _FRAMES_PER_SECOND)
        self.max_captures = max_captures
        self._captures: dict[int, SpeechBankCapture] = {}

    def get(self, user_id: int) -> SpeechBankCapture | None:
        return self._captures.get(user_id)

    @property
    def active_count(self) -> int:
        return len(self._captures)

    def start(
        self, user_id: int, channel_id: int, name: str, duration_seconds: float = 45
    ) -> SpeechBankCapture:
        if user_id <= 0 or channel_id <= 0:
            raise ValueError("Reference capture requires valid user and channel IDs")
        name = name.strip()
        if not name or len(name) > 80 or not name.isprintable():
            raise ValueError("Voice-profile name must contain one to eighty printable characters")
        if not math.isfinite(duration_seconds) or not 30 <= duration_seconds <= 60:
            raise ValueError("Collect between thirty and sixty seconds of actual speech")
        for capture in list(self._captures.values()):
            if self.clock() >= capture._deadline:
                capture._expire()
        if user_id in self._captures:
            raise EnrollmentError("This user already has a voice-reference capture in progress.")
        if len(self._captures) >= self.max_captures:
            raise EnrollmentError("All voice-reference capture slots are occupied.")
        capture = SpeechBankCapture(self, user_id, channel_id, name, duration_seconds)
        self._captures[user_id] = capture
        log.info(
            "voice_reference_bank_started user=%s channel=%s target_seconds=%.2f expiry_seconds=%.1f",
            user_id,
            channel_id,
            duration_seconds,
            self.expiry_seconds,
        )
        return capture

    def feed(self, user_id: int, channel_id: int, pcm16k20ms: bytes, voiced: bool) -> None:
        capture = self._captures.get(user_id)
        if capture is None:
            return
        if capture.channel_id != channel_id:
            capture.cancel()
            return
        capture._feed(pcm16k20ms, voiced)

    def cancel(self, *, user_id: int | None = None, channel_id: int | None = None) -> int:
        captures = [
            capture
            for capture in self._captures.values()
            if (user_id is None or capture.user_id == user_id)
            and (channel_id is None or capture.channel_id == channel_id)
        ]
        for capture in captures:
            capture.cancel()
        return len(captures)

    def cancel_all(self) -> int:
        return self.cancel()
