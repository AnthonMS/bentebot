"""Measure synthetic speech against external APIs without Discord or stored history."""

from __future__ import annotations

import argparse
import asyncio
import audioop
import json
import time
import traceback
from contextlib import aclosing
from pathlib import Path

from .config import Settings
from .providers import OpenAITTS, StreamingSTT
from .speech import wav_to_pcm


async def measure(
    settings: Settings,
    phrase: str,
    *,
    repeats: int = 3,
    padding_seconds: float = 3,
    overlap_tts: bool = False,
) -> list[dict]:
    if not 1 <= repeats <= 10 or not 0 <= padding_seconds <= 5:
        raise ValueError("Use 1-10 runs and 0-5 seconds of silence padding")
    if not 1 <= len(phrase) <= 300:
        raise ValueError("Use a short synthetic phrase of at most 300 characters")
    tts, stt = OpenAITTS(settings), StreamingSTT(settings)
    results = []
    try:
        for run in range(1, repeats + 1):
            started = time.monotonic()
            wav = await tts.synthesize(phrase)
            synthesis_ms = (time.monotonic() - started) * 1000
            pcm = wav_to_pcm(wav)
            mono = audioop.tomono(pcm, 2, 0.5, 0.5)
            mono, _ = audioop.ratecv(mono, 2, 1, 48000, 16000, None)
            report = {
                "run": run,
                "tts_ms": round(synthesis_ms, 1),
                "audio_ms": round(len(pcm) / 192, 1),
                "updates": [],
            }
            started = time.monotonic()
            overlap = None

            async def synthesize_while_listening():
                began = time.monotonic()
                await tts.synthesize(phrase)
                return round((time.monotonic() - began) * 1000, 1)

            async def audio(mono=mono, report=report, started=started):
                nonlocal overlap
                if overlap_tts:
                    overlap = asyncio.create_task(synthesize_while_listening(), name="probe-overlap-tts")
                for offset in range(0, len(mono), 3200):
                    yield mono[offset : offset + 3200].ljust(3200, b"\0")
                    await asyncio.sleep(0.1)
                report["real_audio_sent_ms"] = round((time.monotonic() - started) * 1000, 1)
                for _ in range(round(padding_seconds * 10)):
                    yield b"\0" * 3200
                    await asyncio.sleep(0.1)

            try:
                async with asyncio.timeout(settings.request_timeout):
                    async with aclosing(stt.transcribe(audio(), language="auto")) as stream:
                        async for update in stream:
                            report["updates"].append(
                                {
                                    "at_ms": round((time.monotonic() - started) * 1000, 1),
                                    "final": update.final,
                                    "characters": len(update.text),
                                }
                            )
                    if overlap is not None:
                        report["overlap_tts_ms"] = await overlap
            finally:
                if overlap is not None:
                    overlap.cancel()
                    await asyncio.gather(overlap, return_exceptions=True)
            updates = report["updates"]
            end = report.get("real_audio_sent_ms", report["audio_ms"])
            report["interim_before_speech_end"] = any(
                not item["final"] and item["at_ms"] < end for item in updates
            )
            finals = [item for item in updates if item["final"] and item["at_ms"] >= end]
            report["final_after_speech_ms"] = round(finals[0]["at_ms"] - end, 1) if finals else None
            results.append(report)
    finally:
        await tts.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt-url", required=True)
    parser.add_argument("--tts-base-url", required=True)
    parser.add_argument("--tts-model", default="speaches-ai/Kokoro-82M-v1.0-ONNX")
    parser.add_argument("--tts-voice", default="af_heart")
    parser.add_argument("--phrase", default="Hello, how are you doing today?")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overlap-tts", action="store_true", help="Synthesize another sentence during STT")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings(
        stt_url=args.stt_url,
        tts_base_url=args.tts_base_url,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
    )
    try:
        settings.validate(login=False)  # Local audio only; no paid or cloud fallback.
        results = asyncio.run(
            measure(settings, args.phrase, repeats=args.repeats, overlap_tts=args.overlap_tts)
        )
    except (ValueError, TimeoutError, RuntimeError) as exc:
        # Locations identify the failing stage without dumping provider messages,
        # URLs, credentials, audio or recognized text.
        failure = {
            "synthetic_audio": True,
            "ok": False,
            "error": type(exc).__name__,
            "cause": type(exc.__cause__).__name__ if exc.__cause__ else "none",
            "locations": [
                f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(exc.__traceback__)
            ],
        }
        output = json.dumps(failure, indent=2)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
        return 1
    output = json.dumps({"synthetic_audio": True, "ok": True, "language": "auto", "runs": results}, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
