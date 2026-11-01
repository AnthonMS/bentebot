"""One Discord PCM source for independent speech and music lifetimes."""

from __future__ import annotations

import audioop
import os
import queue
import threading
import time
from collections import deque

import discord

SILENCE = b"\0" * 3840


class SpeechGate:
    """Require recognized words AND sustained, audible VAD before ducking."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.users = {}

    def observe(self, user_id, pcm, voiced):
        now = self.clock()
        if user_id not in self.users:
            if len(self.users) >= 32:
                self.users.pop(next(iter(self.users)))
            self.users[user_id] = [deque(maxlen=12), 0.0, float("-inf")]
        state = self.users[user_id]
        audible = voiced and audioop.rms(pcm, 2) >= 180
        state[0].append(audible)
        if audible:
            state[2] = now
        return bool(audible and state[1] > now and sum(state[0]) >= 8)

    def confirm(self, user_id, *, final=False):
        state = self.users.get(user_id)
        if not state:
            return False
        now = self.clock()
        state[1] = now + (0.15 if final else 2)
        return now - state[2] < 0.18 and sum(state[0]) >= 8


class BufferedMusic(discord.AudioSource):
    """Decode local files off Discord's audio thread, with two seconds of RAM."""

    def __init__(self, path):
        with open(os.devnull, "wb") as errors:
            self.decoder = discord.FFmpegPCMAudio(
                str(path),
                before_options="-nostdin -loglevel error -threads 1 -protocol_whitelist file,pipe",
                options="-vn -sn -dn",
                stderr=errors,
            )
        self.frames = queue.Queue(maxsize=100)
        self.closed = threading.Event()
        self.played_frames = 0
        self.underruns = 0
        self.failed = False
        self.thread = threading.Thread(target=self._decode, name="music-decoder", daemon=True)
        self.thread.start()

    def _decode(self):
        try:
            while not self.closed.is_set():
                frame = self.decoder.read()
                if not frame:
                    self.failed = bool(getattr(self.decoder, "_current_error", None))
                    return
                while not self.closed.is_set():
                    try:
                        self.frames.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except (OSError, ValueError, AttributeError):
            pass  # EOF/error is delivered below without filenames in logs.
        finally:
            if not self.closed.is_set():
                # An exception is an EOF too; the controller checks progress.
                while not self.closed.is_set():
                    try:
                        self.frames.put(b"", timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def read(self):
        try:
            frame = self.frames.get_nowait()
            if frame:
                self.played_frames += 1
            return frame
        except queue.Empty:
            self.underruns += 1
            return SILENCE

    def cleanup(self):
        self.closed.set()
        self.decoder.cleanup()
        self.thread.join(timeout=2)


class Mixer(discord.AudioSource):
    def __init__(self, *, volume=0.35):
        self.lock = threading.Lock()
        self.speech = self.music = None
        self.speech_done = self.music_done = None
        self.volume = volume
        self.paused = False
        self.occupied = True
        self.duck_until = 0.0
        self.gain = volume
        self.closed = False
        self.music_frames = 0

    def speaking(self):
        self.duck_until = time.monotonic() + 0.45

    def has_output(self):
        with self.lock:
            return bool(
                not self.closed
                and self.occupied
                and (self.speech or self.music and self.music_done and not self.paused)
            )

    def set_speech(self, source, done):
        self.stop_speech()
        with self.lock:
            self.speech, self.speech_done = source, done

    def stop_speech(self):
        with self.lock:
            source, done = self.speech, self.speech_done
            self.speech = self.speech_done = None
        if source:
            source.cleanup()
        if done:
            done(None)

    def set_music(self, source, done):
        with self.lock:
            self.music, self.music_done = source, done
            self.music_frames = 0

    def take_music(self):
        with self.lock:
            source, done = self.music, self.music_done
            self.music = self.music_done = None
        if done:
            done(None)
        return source

    def read(self):
        callbacks = []
        with self.lock:
            if self.closed:
                return b""
            if not self.occupied:
                return SILENCE
            speech = self.speech.read() if self.speech else b""
            if self.speech and not speech:
                callbacks.append(self.speech_done)
                self.speech = self.speech_done = None
            music = b""
            if self.music and not self.paused:
                music = self.music.read()
                if not music:
                    callbacks.append(self.music_done)
                    self.music_done = None
                    # Ownership/cleanup stays with play_music's finally block.
                elif music != SILENCE:
                    self.music_frames += 1
            if speech and audioop.rms(speech, 2) > 20:
                self.speaking()
            target = self.volume * (0.4 if time.monotonic() < self.duck_until else 1)
            self.gain += (target - self.gain) * (0.15 if target < self.gain else 0.035)
            frame = audioop.mul(music, 2, self.gain) if music else SILENCE
            if speech:
                frame = audioop.add(frame, audioop.mul(speech, 2, 0.9), 2)
        for callback in callbacks:
            if callback:
                callback(None)
        return frame

    def cleanup(self):
        self.closed = True
        self.stop_speech()
        self.take_music()
