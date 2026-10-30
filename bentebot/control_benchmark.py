"""Synthetic command qualification. Never executes Discord actions or saves audio."""

from __future__ import annotations

import argparse
import asyncio
import audioop
import json
import time
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .providers import OpenAIChat, OpenAITTS, StreamingSTT
from .speech import wav_to_pcm
from .voice_control import control_candidate, spoken_control

CASES = (
    ("stop", "Please stop talking Bentebot.", "stop"),
    ("pause", "Can you pause for a sec Bente?", "stop"),
    ("leave", "Please leave the call Bentebot.", "leave"),
    ("style", "Stop speaking like a robot.", None),
)


async def qualify(settings: Settings, *, compare_hints=False, include_intent=False) -> list[dict]:
    """At most eight sequential short utterances; one provider failure ends the run."""
    tts = OpenAITTS(settings)
    chat = OpenAIChat(settings) if include_intent else None
    rows = []
    hints = ("", settings.stt_context) if compare_hints and settings.stt_context else (settings.stt_context,)
    try:
        for case_id, phrase, expected in CASES:
            wav = await tts.synthesize(phrase)
            pcm = wav_to_pcm(wav)
            mono = audioop.tomono(pcm, 2, 0.5, 0.5)
            mono, _ = audioop.ratecv(mono, 2, 1, 48000, 16000, None)
            for hint in hints:
                row = {"case": case_id, "expected": expected, "hint_enabled": bool(hint)}
                started = time.monotonic()
                speech_end = None
                finals = []

                async def audio(mono=mono):
                    nonlocal speech_end
                    for pos in range(0, len(mono), 3200):
                        yield mono[pos : pos + 3200].ljust(3200, b"\0")
                        await asyncio.sleep(0.1)
                    speech_end = time.monotonic()
                    for _ in range(25):
                        yield b"\0" * 3200
                        await asyncio.sleep(0.1)

                try:
                    async with (
                        asyncio.timeout(min(settings.request_timeout, 20)),
                        aclosing(
                            StreamingSTT(replace(settings, stt_context=hint)).transcribe(audio())
                        ) as stream,
                    ):
                        async for update in stream:
                            if spoken_control(update.text) or control_candidate(update.text):
                                row.setdefault(
                                    "first_control_ms", round((time.monotonic() - started) * 1000, 1)
                                )
                            if update.final:
                                finals.append(update.text)
                                if speech_end:
                                    row.setdefault(
                                        "final_lag_ms", round((time.monotonic() - speech_end) * 1000, 1)
                                    )
                    text = " ".join(finals)
                    action = spoken_control(text)
                    row["method"] = "direct" if action else "unmatched"
                    if action is None and chat and control_candidate(text):
                        began = time.monotonic()
                        action = await chat.resolve_spoken_control(text, settings.chat_model)
                        row["intent_ms"] = round((time.monotonic() - began) * 1000, 1)
                        row["method"] = "intent"
                    row.update(action=action, recognized=bool(text), passed=bool(text) and action == expected)
                    # Test data is fixed synthetic speech. Do not log normal caller transcripts.
                    row["synthetic_transcript"] = text[:500]
                    rows.append(row)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    row.update(error=type(exc).__name__, passed=False)
                    rows.append(row)
                    return rows
    finally:
        await tts.close()
        if chat:
            await chat.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-hints", action="store_true")
    parser.add_argument(
        "--include-intent", action="store_true", help="Use the configured chat alias to classify; no actions"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.validate(login=False)
    if not settings.voice_configured:
        parser.error("Configure existing STT and TTS APIs first")
    try:
        rows = asyncio.run(
            qualify(settings, compare_hints=args.compare_hints, include_intent=args.include_intent)
        )
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(json.dumps({"synthetic_audio": True, "error": type(exc).__name__}))
        return 1
    result = {"synthetic_audio": True, "discord_actions_executed": False, "runs": rows}
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if rows and all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
