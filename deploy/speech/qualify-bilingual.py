"""Reproducible WAV-to-live-STT checks. Does not change speech service settings."""

from __future__ import annotations

import argparse
import asyncio
import audioop
import json
import re
import time
import wave
from contextlib import aclosing
from pathlib import Path

from bentebot.config import Settings
from bentebot.providers import StreamingSTT


def words(text):
    return re.findall(r"\w+", text.casefold())


def word_error(reference, actual):
    expected, recognized = words(reference), words(actual)
    previous = list(range(len(recognized) + 1))
    for i, left in enumerate(expected, 1):
        row = [i]
        for j, right in enumerate(recognized, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = row
    return round(previous[-1] / max(1, len(expected)), 3)


async def qualify(stt, filename, reference):
    with wave.open(filename, "rb") as wav:
        raw, width, channels, rate = (
            wav.readframes(wav.getnframes()),
            wav.getsampwidth(),
            wav.getnchannels(),
            wav.getframerate(),
        )
    if channels == 2:
        raw = audioop.tomono(raw, width, 0.5, 0.5)
    elif channels != 1:
        raise ValueError("Use mono/stereo WAV samples")
    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
    pcm, _ = audioop.ratecv(raw, 2, 1, rate, 16000, None)
    ended = None
    began = time.monotonic()
    first = None
    text, final, error = "", False, None

    async def audio():
        nonlocal ended
        for start in range(0, len(pcm), 3200):
            yield pcm[start : start + 3200]
            await asyncio.sleep(0.1)
        ended = time.monotonic()
        for _ in range(20):
            yield b"\0" * 3200
            await asyncio.sleep(0.1)

    try:
        async with asyncio.timeout(30):
            async with aclosing(stt.transcribe(audio(), language="auto")) as stream:
                async for update in stream:
                    first = first or time.monotonic()
                    text, final = update.text, update.final
                    if final:
                        break
    except Exception as exc:
        error = type(exc).__name__
    return {
        "file": Path(filename).name,
        "reference": reference,
        "recognized": text,
        "final": final,
        "word_error_rate": word_error(reference, text),
        "error": error,
        "first_update_ms": round((first - began) * 1000, 1) if first else None,
        "speech_end_to_final_ms": round((time.monotonic() - ended) * 1000, 1) if ended and final else None,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt-url", required=True)
    parser.add_argument("--case", nargs=2, action="append", required=True, metavar=("WAV", "EXPECTED_TEXT"))
    parser.add_argument("--report")
    args = parser.parse_args()
    stt = StreamingSTT(Settings(stt_url=args.stt_url, stt_language="auto", turn_silence_ms=800))
    results = [await qualify(stt, filename, reference) for filename, reference in args.case]
    report = {"endpointing_ms": 800, "language": "auto", "cases": results}
    if args.report:
        await asyncio.to_thread(
            Path(args.report).write_text,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
