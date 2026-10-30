from __future__ import annotations

import asyncio
import audioop
import io
import re
import struct
import wave
from collections.abc import Awaitable, Callable
from contextlib import aclosing

from .domain import ResponseRecord
from .providers import ProviderError
from .voice_style import VoiceFollowupFilter


class VoiceLinkFilter:
    """Remove URL targets before sentence splitting, including split or very long URLs."""

    prefixes = ("http://", "https://", "www.")

    def __init__(self):
        self.pending = ""
        self.in_url = False

    def feed(self, text: str, *, final: bool = False) -> str:
        output = []
        for char in text:
            if self.in_url:
                if char.isspace() or char in "<>)]}\"'`":
                    self.in_url = False
                    output.append(char)
                continue
            self.pending += char
            while self.pending:
                prefix = self.pending.casefold()
                if prefix in self.prefixes:
                    self.pending = ""
                    self.in_url = True
                    break
                if any(item.startswith(prefix) for item in self.prefixes):
                    break
                output.append(self.pending[0])
                self.pending = self.pending[1:]
        if final:
            output.append(self.pending)
            self.pending = ""
            self.in_url = False
        return "".join(output)


class SentenceBuffer:
    """Sentence-level synthesis; fenced code remains text-only, including split fences."""

    def __init__(self, max_chars: int = 320, suppress_after_code: bool = False):
        self.buffer = ""
        self.in_code = False
        self.had_code = False
        self.max_chars = max_chars
        self.suppress_after_code = suppress_after_code
        self.marker = "```"

    def feed(self, text: str, *, final: bool = False) -> list[str]:
        if self.suppress_after_code and self.had_code:
            return []
        self.buffer += text
        sentences = []
        while self.buffer:
            match = re.search(r"`{3,}|~{3,}", self.buffer)
            fence = match.start() if match else -1
            if self.in_code:
                closing = re.search(
                    r"(?m)^ {0,3}"
                    + re.escape(self.marker[0])
                    + "{"
                    + str(len(self.marker))
                    + r",}[ \t]*(?:\r?\n|$)",
                    self.buffer,
                )
                if closing is None or (
                    closing.end() == len(self.buffer) and not final and not self.buffer.endswith("\n")
                ):
                    if final:
                        self.buffer = ""
                    break
                self.buffer = self.buffer[closing.end() :]
                self.in_code = False
                continue
            boundary = None
            for end in re.finditer(r"[.!?](?:[\"')\]]*)(?=\s|$)", self.buffer):
                if end.end() == len(self.buffer) and not final:
                    continue  # A following token may turn a period into a decimal or abbreviation.
                word = re.search(r"([\w.]+)\.$", self.buffer[: end.end()])
                if word and word.group(1).casefold() in {
                    "mr",
                    "mrs",
                    "ms",
                    "dr",
                    "prof",
                    "e.g",
                    "i.e",
                    "vs",
                    "approx",
                }:
                    continue
                boundary = end.end()
                break
            if fence >= 0 and (boundary is None or fence < boundary):
                prefix = self.buffer[:fence]
                if prefix.strip():
                    sentences.append(self.clean(prefix))
                self.marker = match.group()
                self.buffer = self.buffer[fence + len(self.marker) :]
                self.in_code = self.had_code = True
                if self.suppress_after_code:
                    self.buffer = ""
                    break
                continue
            if (boundary is None or boundary > self.max_chars) and len(self.buffer) > self.max_chars:
                cut = self.buffer.rfind(" ", 0, self.max_chars)
                boundary = cut if cut > 0 else self.max_chars
            if boundary:
                sentences.append(self.clean(self.buffer[:boundary]))
                self.buffer = self.buffer[boundary:].lstrip()
            elif final:
                sentences.append(self.clean(self.buffer))
                self.buffer = ""
            else:
                break
        return [sentence for sentence in sentences if sentence]

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r"<@!?\d+>|<@&\d+>|<#\d+>", "", text)
        text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
        text = text.replace("<>", "")
        if re.fullmatch(r"\s*(?:sources?|references?)\s*:?\s*", text, re.I):
            return ""
        return re.sub(r"\s+", " ", re.sub(r"[*_`#]", "", text)).strip()


def wav_to_pcm(data: bytes, max_seconds: int = 120) -> bytes:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, width, rate, frames = (
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getframerate(),
                wav.getnframes(),
            )
            if channels not in {1, 2} or width not in {1, 2, 3, 4} or not 8000 <= rate <= 96000:
                raise ValueError("Unsupported WAV format")
            offset, unknown_length = 12, False
            while offset + 8 <= len(data):
                chunk, size = struct.unpack_from("<4sI", data, offset)
                if chunk == b"data":
                    unknown_length = size == 0xFFFFFFFF
                    break
                if size == 0xFFFFFFFF:
                    raise ValueError("Unknown non-audio WAV chunk length")
                offset += 8 + size + (size & 1)
            if not unknown_length and frames > rate * max_seconds:
                raise ValueError("TTS segment too long")
            pcm = wav.readframes(rate * max_seconds + 1 if unknown_length else frames)
            if len(pcm) > rate * max_seconds * channels * width:
                raise ValueError("TTS segment too long")
            if len(pcm) % (channels * width) or (
                not unknown_length and len(pcm) != frames * channels * width
            ):
                raise ValueError("Truncated WAV")
        if width == 1:
            pcm = audioop.bias(pcm, 1, -128)
        if width != 2:
            pcm = audioop.lin2lin(pcm, width, 2)
        if channels == 2:
            pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 48000, None)
        return audioop.tostereo(pcm, 2, 1, 1)
    except (wave.Error, ValueError, audioop.error, EOFError) as exc:
        raise ProviderError("TTS returned unsupported or invalid PCM WAV audio.") from exc


async def speak_response(
    tokens,
    synthesize: Callable[[str], Awaitable[bytes]],
    play: Callable[[bytes, str, Callable[[str], None]], Awaitable[None]],
    allowed: Callable[[], Awaitable[bool]],
    record: ResponseRecord,
    *,
    max_chars: int = 320,
    capture_generated: bool = True,
    suppress_after_code: bool = False,
    filter_stock_followups: bool = False,
    suppress_urls: bool = False,
) -> None:
    """Bounded producer -> sentence synthesis -> playback. Any cancellation closes all three."""
    sentences: asyncio.Queue[tuple[str, float] | None] = asyncio.Queue(maxsize=3)
    audio: asyncio.Queue[tuple[str, bytes] | None] = asyncio.Queue(maxsize=2)
    started = asyncio.get_running_loop().time()

    async def produce():
        generation_started = asyncio.get_running_loop().time()
        splitter = SentenceBuffer(max_chars, suppress_after_code=suppress_after_code)
        links = VoiceLinkFilter() if suppress_urls else None
        filter_sentence = VoiceFollowupFilter() if filter_stock_followups else lambda text: text
        async with aclosing(tokens):
            async for token in tokens:
                record.timings.setdefault(
                    "chat_first_token_ms", (asyncio.get_running_loop().time() - generation_started) * 1000
                )
                if capture_generated:
                    record.generated += token
                for sentence in splitter.feed(links.feed(token) if links else token):
                    sentence = filter_sentence(sentence)
                    if not sentence:
                        continue
                    record.timings.setdefault(
                        "first_sentence_ms", (asyncio.get_running_loop().time() - generation_started) * 1000
                    )
                    await sentences.put((sentence, asyncio.get_running_loop().time()))
        for sentence in splitter.feed(links.feed("", final=True) if links else "", final=True):
            sentence = filter_sentence(sentence)
            if not sentence:
                continue
            record.timings.setdefault(
                "first_sentence_ms", (asyncio.get_running_loop().time() - generation_started) * 1000
            )
            await sentences.put((sentence, asyncio.get_running_loop().time()))
        await sentences.put(None)

    async def synth():
        while (item := await sentences.get()) is not None:
            sentence, queued_at = item
            permission_started = asyncio.get_running_loop().time()
            record.timings.setdefault("tts_first_queue_wait_ms", (permission_started - queued_at) * 1000)
            if not await allowed():
                raise ProviderError("The audience changed; this response was stopped.")
            record.timings.setdefault(
                "tts_first_permission_ms", (asyncio.get_running_loop().time() - permission_started) * 1000
            )
            synthesis_started = asyncio.get_running_loop().time()
            wav = await synthesize(sentence)
            decoded_at = asyncio.get_running_loop().time()
            record.timings.setdefault("tts_first_ms", (decoded_at - synthesis_started) * 1000)
            pcm = await asyncio.to_thread(wav_to_pcm, wav)
            record.timings.setdefault(
                "decode_first_ms", (asyncio.get_running_loop().time() - decoded_at) * 1000
            )
            duration_ms = len(pcm) / 192
            record.timings.setdefault("tts_first_audio_duration_ms", duration_ms)
            await audio.put((sentence, pcm))
        await audio.put(None)

    async def playback():
        while (item := await audio.get()) is not None:
            sentence, pcm = item
            if not await allowed():
                raise ProviderError("The audience changed; this response was stopped.")
            if record.first_audio_ms is None:
                record.first_audio_ms = (asyncio.get_running_loop().time() - started) * 1000
                record.timings["pipeline_first_audio_ms"] = record.first_audio_ms
            prefix = record.spoken

            def progress(text: str, prefix: str = prefix):
                record.spoken = " ".join(part for part in (prefix, text) if part)

            await play(pcm, sentence, progress)

    try:

        async def observed(stage, operation):
            try:
                await operation
            except Exception:
                record.error_stage = record.error_stage or stage
                raise

        async with asyncio.TaskGroup() as group:
            group.create_task(observed("chat_stream", produce()), name="llm-stream")
            group.create_task(observed("tts_synthesis_decode", synth()), name="tts-synthesis")
            group.create_task(observed("playback", playback()), name="tts-playback")
    except asyncio.CancelledError:
        record.interrupted = True
        raise
    except ExceptionGroup as exc:
        record.error = "Response pipeline failed. Check provider configuration and availability."
        raise ProviderError(record.error) from exc
