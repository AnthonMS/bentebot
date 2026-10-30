"""Run INSIDE an external STT container; no network unless --wav loads a missing model.

Preflight imports/runtime visibility do not prove model inference works. --wav benchmarks
a user-supplied WAV locally and may download the selected Whisper weights on first use.
It never calls an inference API or reads Bentebot's credentials.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--wav", type=Path, help="Optional local PCM WAV to transcribe twice for a timing benchmark"
    )
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default="en", help="en, da, or auto")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument(
        "--segment-language-redetection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Additional per-segment language detection; auto still detects when this is disabled",
    )
    parser.add_argument(
        "--compute-type", default="auto", help="Try int8_float32 on Pascal, float16 on 4090, int8 on CPU"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--show-text", action="store_true", help="Explicitly print your test recording's transcript"
    )
    args = parser.parse_args()
    report: dict = {"device_requested": args.device, "packages": {}, "inference_tested": False}
    try:
        import ctranslate2
        import torch

        for package in (
            "whisperlivekit",
            "faster-whisper",
            "av",
            "ctranslate2",
            "torch",
            "torchaudio",
            "nvidia-cudnn-cu12",
        ):
            try:
                report["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        report["ctranslate2_cuda_devices"] = ctranslate2.get_cuda_device_count()
        report["torch_cuda_available"] = torch.cuda.is_available()
        report["compute_types"] = sorted(ctranslate2.get_supported_compute_types(args.device))
        if args.device == "cuda" and not report["ctranslate2_cuda_devices"]:
            raise RuntimeError("CTranslate2 cannot see a CUDA device; check NVIDIA container passthrough")
        if args.wav:
            if args.threads <= 0 or args.beam_size <= 0:
                raise ValueError("threads and beam size must be positive")
            with wave.open(str(args.wav), "rb") as audio:
                seconds = audio.getnframes() / audio.getframerate()
            if not 1 <= seconds <= 120:
                raise ValueError("Use a 1–120 second WAV for the benchmark")
            from faster_whisper import WhisperModel

            started = time.perf_counter()
            model = WhisperModel(
                args.model, device=args.device, compute_type=args.compute_type, cpu_threads=args.threads
            )
            report["model_load_seconds"] = round(time.perf_counter() - started, 3)
            report["actual_compute_type"] = model.model.compute_type
            report["audio_seconds"] = round(seconds, 3)
            report["runs"] = []
            for _ in range(2):
                started = time.perf_counter()
                segments, info = model.transcribe(
                    str(args.wav),
                    language=None if args.language == "auto" else args.language,
                    beam_size=args.beam_size,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    multilingual=args.language == "auto" and args.segment_language_redetection,
                )
                text = " ".join(segment.text.strip() for segment in segments)
                elapsed = time.perf_counter() - started
                run = {
                    "seconds": round(elapsed, 3),
                    "real_time_factor": round(elapsed / seconds, 3),
                    "word_count": len(text.split()),
                    "detected_language": info.language,
                    "language_probability": round(info.language_probability, 3),
                    "segment_language_redetection": args.segment_language_redetection,
                }
                if args.show_text:
                    run["text"] = text
                report["runs"].append(run)
            report["inference_tested"] = True
        report["ok"] = True
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report["ok"] = False
        report["error"] = type(exc).__name__
        report["detail"] = str(exc)[:500]
        print(json.dumps(report, indent=2), file=sys.stderr)
        print(
            "Runtime preflight failed. Check local provider logs; no API fallback was attempted.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
