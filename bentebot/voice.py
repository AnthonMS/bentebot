from __future__ import annotations

import asyncio
import audioop
import logging
import queue
import threading
import time
import traceback
import weakref
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime

import discord
import webrtcvad
from discord.ext import voice_recv

from .conversation import VoiceConversation
from .providers import ProviderError, StreamingSTT, STTStreamError

log = logging.getLogger(__name__)
PCM_FRAME = 640  # 20 ms, 16 kHz mono, s16le


class TrackedPCM(discord.AudioSource):
    """Discord reads 20ms frames on its audio thread; progress is an estimate of sent words."""

    def __init__(self, pcm: bytes, text: str, progress, loop, *, lead_in_frames: int = 0):
        self.pcm, self.text, self.progress, self.loop = pcm, text, progress, loop
        self.position = 0
        self.lead_in_frames = lead_in_frames
        self.gate = threading.Event()
        self.gate.set()

    def read(self) -> bytes:
        if not self.gate.is_set() or self.position >= len(self.pcm):
            return b""
        if self.lead_in_frames:
            self.lead_in_frames -= 1
            return b"\0" * 3840
        frame = self.pcm[self.position : self.position + 3840]
        if self.position == 0:
            self.loop.call_soon_threadsafe(self.progress, "")
        self.position += len(frame)
        if self.position % (3840 * 5) == 0 or self.position >= len(self.pcm):
            words = self.text.split()
            count = int(len(words) * self.position / len(self.pcm))
            self.loop.call_soon_threadsafe(self.progress, " ".join(words[:count]))
        return frame.ljust(3840, b"\0")

    def cleanup(self) -> None:
        self.gate.clear()


class DiscordPlayback:
    def __init__(self, client: discord.VoiceClient, *, mixed=False, music_volume=0.35):
        self.client = client
        self.source: TrackedPCM | None = None
        self.last_finished = float("-inf")
        from .music_audio import Mixer, SpeechGate

        self.mixer = Mixer(volume=music_volume) if mixed else None
        self.speech_gate = SpeechGate() if mixed else None
        self.duck_humans = True

    def _start_mixer(self):
        if self.mixer and self.mixer.closed:
            raise ProviderError("Discord music output ended. Rejoin the voice call to recover.")
        self._sync_mixer()

    def _sync_mixer(self):
        if not self.mixer or self.mixer.closed:
            return
        paused = getattr(self.client, "is_paused", lambda: False)()
        if self.mixer.has_output():
            if paused:
                self.client.resume()
            elif not self.client.is_playing():
                self.client.play(
                    self.mixer, after=lambda error: self.mixer.cleanup(), bitrate=192, signal_type="music"
                )
        elif self.client.is_playing():
            self.client.pause()

    async def play_music(self, path):
        from .music_audio import BufferedMusic

        if not self.mixer:
            raise ProviderError("Music playback is disabled for this voice session.")
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def finish(error):
            def resolve():
                if not done.done():
                    done.set_result(None)

            loop.call_soon_threadsafe(resolve)

        building = asyncio.create_task(asyncio.to_thread(BufferedMusic, path))
        try:
            source = await asyncio.shield(building)
        except asyncio.CancelledError:
            source = await building
            await asyncio.to_thread(source.cleanup)
            raise
        try:
            self.mixer.set_music(source, finish)
            self._start_mixer()
            await done
            if source.played_frames < 10 or source.failed:
                raise ProviderError("The recording could not be decoded.")
        finally:
            self.mixer.take_music()
            self._sync_mixer()
            await asyncio.to_thread(source.cleanup)

    def pause_music(self, paused):
        if self.mixer:
            self.mixer.paused = paused
            self._sync_mixer()

    def set_listener_presence(self, occupied):
        if self.mixer:
            self.mixer.occupied = occupied
            if not occupied:
                self.speech_gate.users.clear()
            self._sync_mixer()

    def human_speaking(self):
        if self.mixer:
            self.mixer.speaking()

    def human_audio(self, user_id, frame, voiced):
        if self.duck_humans and self.speech_gate and self.speech_gate.observe(user_id, frame, voiced):
            self.human_speaking()

    def recognized_speech(self, user_id, final=False):
        if self.duck_humans and self.speech_gate and self.speech_gate.confirm(user_id, final=final):
            self.human_speaking()

    def close(self):
        if self.mixer:
            self.mixer.cleanup()
        self.client.stop_playing()

    async def play(self, pcm: bytes, text: str, progress) -> None:
        if not pcm:
            return
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        lead_in = 4 if time.monotonic() - self.last_finished > 1 else 0
        source = self.source = TrackedPCM(pcm, text, progress, loop, lead_in_frames=lead_in)

        def finish(error):
            def set_result():
                if not done.done():
                    if error:
                        done.set_exception(ProviderError("Discord audio playback failed."))
                    else:
                        done.set_result(None)

            loop.call_soon_threadsafe(set_result)

        try:
            if self.mixer:
                self.mixer.set_speech(source, finish)
                self._start_mixer()
            else:
                self.client.play(source, after=finish)
            await done
        finally:
            self.last_finished = time.monotonic()
            source.gate.clear()
            if self.source is source:
                self.source = None
            if self.mixer:
                self.mixer.stop_speech()
                self._sync_mixer()
            elif self.client.is_playing():
                self.client.stop_playing()

    def stop(self) -> None:
        if self.source:
            self.source.gate.clear()
        if self.mixer:
            self.mixer.stop_speech()
            self._sync_mixer()
        else:
            self.client.stop_playing()  # Receive must continue while output is cancelled.


class ReceiveSink(voice_recv.AudioSink):
    def __init__(self, incoming: queue.Queue, metrics):
        super().__init__()
        self.incoming, self.metrics = incoming, metrics
        self.ignored_user_ids: frozenset[int] = frozenset()
        self.allowed_user_ids: frozenset[int] | None = None

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data) -> None:
        if (
            user is None
            or user.bot
            or user.id in self.ignored_user_ids
            or not data.pcm
            or self.allowed_user_ids is not None
            and user.id not in self.allowed_user_ids
        ):
            return
        try:
            self.incoming.put_nowait((user.id, user.display_name, bytes(data.pcm)))
        except queue.Full:
            self.metrics.dropped_frames += 1

    def cleanup(self) -> None:
        pass


@dataclass
class Speaker:
    user_id: int
    name: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    vad: webrtcvad.Vad = field(default_factory=lambda: webrtcvad.Vad(2))
    resample_state: tuple | None = None
    partial_pcm: bytes = b""
    pending: bytearray = field(default_factory=bytearray)
    last_packet: float = field(default_factory=time.monotonic)
    last_voice: float = 0.0
    retry_after: float = 0.0
    task: asyncio.Task | None = None
    frames_since_send: int = 0
    active: bool = True


class VoiceReceiver:
    """Bounded cross-thread ingress and independent per-speaker live STT streams."""

    def __init__(
        self, client, conversation: VoiceConversation, stt: StreamingSTT, notify, on_frame=None, language=None
    ):
        self.client, self.conversation, self.stt, self.notify = client, conversation, stt, notify
        self.incoming: queue.Queue = queue.Queue(maxsize=300)
        self.speakers: dict[int, Speaker] = {}
        self.sink = ReceiveSink(self.incoming, conversation.metrics)
        self.task: asyncio.Task | None = None
        self.closed = False
        self.last_error = ""
        self._last_notice = 0.0
        self.on_frame = on_frame
        self.language = language
        self.on_recognized_speech = None

    def _observe(self, user_id: int, frame: bytes, voiced: bool):
        if audio := getattr(self.conversation, "audio", None):
            audio.observe(user_id, frame, voiced)
        if self.on_frame:
            self.on_frame(user_id, self.client.channel.id, frame, voiced)

    def start(self):
        self.client.listen(self.sink)
        runner = getattr(getattr(self.client, "_connection", None), "_runner", None)
        if isinstance(runner, asyncio.Task):
            reference = weakref.ref(self)

            def voice_control_stopped(task):
                receiver = reference()
                if receiver is None or receiver.closed or task.cancelled():
                    return
                error = task.exception()
                if error is not None:
                    kind = type(error).__name__
                    receiver.last_error = f"Discord voice control stopped ({kind}). Rejoin voice to recover."
                    # Exception messages can contain provider/session data.
                    log.error("Discord voice control reader stopped (%s)", kind)

            runner.add_done_callback(voice_control_stopped)
        self.task = asyncio.create_task(self._pump(), name="discord-audio-ingress")

    async def _audio(self, speaker: Speaker):
        while (frame := await speaker.queue.get()) is not None:
            yield frame

    async def _transcribe(self, speaker: Speaker):
        try:
            audio = self._audio(speaker)
            stream = (
                self.stt.transcribe(audio, language=self.language())
                if self.language
                else self.stt.transcribe(audio)
            )
            async with aclosing(stream):
                await self._receive_updates(speaker, stream)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = exc.reason if isinstance(exc, STTStreamError) else "unknown"
            close_code = exc.close_code if isinstance(exc, STTStreamError) else None
            metrics = self.conversation.metrics
            metrics.stt_failures += 1
            metrics.stt_last_failure_reason = reason
            metrics.stt_last_close_code = close_code
            metrics.stt_last_failure_utc = datetime.now(UTC).isoformat(timespec="seconds")
            log.warning(
                "stt_stream channel=%s user=%s outcome=failed error=%s reason=%s close_code=%s cause=%s locations=%s",
                self.conversation.source.channel_id,
                speaker.user_id,
                type(exc).__name__,
                reason,
                close_code,
                type(exc.__cause__).__name__ if exc.__cause__ else "none",
                [
                    f"{frame.name}:{frame.lineno}"
                    for frame in traceback.extract_tb((exc.__cause__ or exc).__traceback__)[-5:]
                ],
            )
            self.last_error = "Speech recognition failed. Check STT service logs; new speech will retry."
            if reason == "out_of_memory":
                self.last_error = "Speech recognition ran out of memory; new speech will retry."
            elif reason == "overloaded":
                self.last_error = "Speech recognition is overloaded; new speech will retry."
            self.conversation.metrics.provider_errors += 1
            self.conversation.partial.pop(speaker.user_id, None)
            speaker.retry_after = time.monotonic() + 3
            if time.monotonic() - self._last_notice > 30:
                self._last_notice = time.monotonic()
                try:
                    await self.notify(self.last_error)
                except Exception as notice_error:
                    log.warning("stt_notice outcome=failed error=%s", type(notice_error).__name__)

    async def _receive_updates(self, speaker: Speaker, stream):
        async for update in stream:
            if self.closed or not speaker.active:
                return
            self.last_error = ""
            if update.text.strip() and self.on_recognized_speech:
                self.on_recognized_speech(user_id=speaker.user_id, final=update.final)
            try:
                await self.conversation.transcript(speaker.user_id, speaker.name, update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A local turn-handler failure is not a broken STT connection.
                # Keep receiving rather than dropping subsequent commands.
                locations = [
                    f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(exc.__traceback__)[-5:]
                ]
                log.error(
                    "voice_transcript_handler_failed channel=%s user=%s error=%s locations=%s",
                    self.conversation.source.channel_id,
                    speaker.user_id,
                    type(exc).__name__,
                    locations,
                )
                self.last_error = "Voice turn handling failed; see bot logs. Recognition remains connected."
                self.conversation.metrics.provider_errors += 1
                self.conversation.partial.pop(speaker.user_id, None)

    async def _pump(self):
        last_send = time.monotonic()
        while not self.closed:
            # A fixed batch prevents a busy channel from monopolizing the event loop.
            for _ in range(100):
                try:
                    user_id, name, pcm = self.incoming.get_nowait()
                except queue.Empty:
                    break
                self.conversation.metrics.received_frames += 1
                if self.sink.allowed_user_ids is not None and user_id not in self.sink.allowed_user_ids:
                    self.conversation.metrics.dropped_frames += 1
                    continue
                speaker = self.speakers.get(user_id)
                if speaker is None:
                    if len(self.speakers) >= max(32, self.conversation.settings.max_speakers * 4):
                        idle = [
                            item for item in self.speakers.values() if item.task is None or item.task.done()
                        ]
                        if not idle:
                            self.conversation.metrics.dropped_frames += 1
                            continue
                        self.speakers.pop(min(idle, key=lambda item: item.last_packet).user_id)
                    speaker = self.speakers[user_id] = Speaker(user_id, name)
                if speaker.task and speaker.task.done():
                    # Reconnect after a bounded backoff, dropping stale audio rather than replaying it.
                    if time.monotonic() < speaker.retry_after:
                        continue
                    speaker.queue = asyncio.Queue(maxsize=100)
                    speaker.pending.clear()
                    speaker.partial_pcm = b""
                    speaker.resample_state = None
                    speaker.last_voice = 0.0
                    speaker.task = None
                speaker.last_packet = time.monotonic()
                try:
                    mono = audioop.tomono(pcm, 2, 0.5, 0.5)
                    frame, speaker.resample_state = audioop.ratecv(
                        mono,
                        2,
                        1,
                        48000,
                        16000,
                        speaker.resample_state,
                    )
                    speaker.partial_pcm += frame
                    while len(speaker.partial_pcm) >= PCM_FRAME:
                        frame, speaker.partial_pcm = (
                            speaker.partial_pcm[:PCM_FRAME],
                            speaker.partial_pcm[PCM_FRAME:],
                        )
                        voiced = speaker.vad.is_speech(frame, 16000)
                        self._observe(user_id, frame, voiced)
                        speaker.frames_since_send += 1
                        if voiced:
                            speaker.last_voice = time.monotonic()
                            self.conversation.speech_activity(user_id)
                        speaker.pending.extend(frame)
                    if speaker.task is None and speaker.last_voice:
                        active = [
                            item for item in self.speakers.values() if item.task and not item.task.done()
                        ]
                        if len(active) >= self.conversation.settings.max_speakers:
                            old = [
                                item
                                for item in active
                                if time.monotonic() - item.last_voice > 5
                                and item.user_id not in self.conversation.partial
                            ]
                            if old:
                                replaced = min(old, key=lambda item: item.last_voice)
                                replaced.task.cancel()
                                await asyncio.gather(replaced.task, return_exceptions=True)
                                active.remove(replaced)
                        if len(active) < self.conversation.settings.max_speakers:
                            speaker.task = asyncio.create_task(
                                self._transcribe(speaker), name=f"stt-user-{user_id}"
                            )
                except (audioop.error, ValueError):
                    self.conversation.metrics.dropped_frames += 1
            now = time.monotonic()
            if now - last_send >= 0.1:
                # Keep a stable audio clock instead of accumulating event-loop scheduling drift.
                last_send += 0.1
                for user_id, speaker in list(self.speakers.items()):
                    for _ in range(max(0, 5 - speaker.frames_since_send)):
                        self._observe(user_id, b"\0" * PCM_FRAME, False)
                    speaker.frames_since_send = 0
                    if now - speaker.last_packet > 60:
                        if speaker.task:
                            speaker.task.cancel()
                            await asyncio.gather(speaker.task, return_exceptions=True)
                        self.conversation.partial.pop(user_id, None)
                        del self.speakers[user_id]
                        continue
                    if speaker.task is None or speaker.task.done():
                        speaker.pending = speaker.pending[-16000:]  # Up to 0.5s pre-roll.
                        continue
                    # Discord omits silence packets. Pad every batch to advance STT's audio clock.
                    frame = bytes(speaker.pending[:3200]).ljust(3200, b"\0")
                    del speaker.pending[:3200]
                    if len(speaker.pending) > 16000:
                        speaker.pending = speaker.pending[-16000:]
                        self.conversation.metrics.dropped_frames += 1
                    try:
                        speaker.queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        self.conversation.metrics.dropped_frames += 1
            await asyncio.sleep(0.02)

    def set_members(self, user_ids):
        self.sink.allowed_user_ids = frozenset(user_ids)

    async def drop_speaker(self, user_id: int):
        speaker = self.speakers.pop(user_id, None)
        # Atomically filter the bounded cross-thread ingress queue so a rapid
        # rejoin cannot replay packets from the previous membership period.
        with self.incoming.mutex:
            kept = [item for item in self.incoming.queue if item[0] != user_id]
            removed = len(self.incoming.queue) - len(kept)
            self.incoming.queue.clear()
            self.incoming.queue.extend(kept)
            self.incoming.unfinished_tasks -= removed
            self.incoming.not_full.notify_all()
            if not self.incoming.unfinished_tasks:
                self.incoming.all_tasks_done.notify_all()
        if speaker:
            speaker.active = False
            if speaker.task and speaker.task is not asyncio.current_task():
                speaker.task.cancel()
                await asyncio.gather(speaker.task, return_exceptions=True)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.sink.allowed_user_ids = frozenset()
        try:
            self.client.stop_listening()
        finally:
            tasks = [speaker.task for speaker in self.speakers.values() if speaker.task]
            for speaker in self.speakers.values():
                speaker.active = False
            if self.task:
                tasks.append(self.task)
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self.speakers.clear()
