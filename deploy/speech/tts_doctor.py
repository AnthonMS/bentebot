"""Run a synthetic Kokoro inference test inside Speaches, without starting its API.

Reads an existing model cache. Saves no audio or transcripts and makes no API calls.
CUDA mode rejects CPU fallback and verifies actual CUDA kernels using ORT profiling.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/models"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--memory-test", action="store_true", help="Qualify the optional 2 GiB CUDA arena")
    args = parser.parse_args()
    report = {"device_requested": args.device, "inference_tested": False}
    session = None
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        report["packages"] = {
            name: importlib.metadata.version(name) for name in ("kokoro-onnx", "onnxruntime-gpu")
        }
        report["available_providers"] = ort.get_available_providers()
        if args.device == "cuda":
            cudnn = ctypes.CDLL("libcudnn.so.9")
            cudnn.cudnnGetVersion.restype = ctypes.c_size_t
            report["cudnn_version"] = cudnn.cudnnGetVersion()
        snapshots = args.cache / "models--speaches-ai--Kokoro-82M-v1.0-ONNX" / "snapshots"
        models = sorted(
            path for path in snapshots.glob("*/model.onnx") if (path.parent / "voices.bin").is_file()
        )
        if not models:
            raise ValueError("Cached Kokoro model/voices missing; check the mounted CPU TTS cache")
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = "/tmp/bentebot-tts-profile"
        cuda_options = {"cudnn_conv_algo_search": "DEFAULT"}
        if args.memory_test:
            cuda_options.update(
                arena_extend_strategy="kSameAsRequested",
                gpu_mem_limit="2147483648",
                cudnn_conv_use_max_workspace="0",
            )
        providers = (
            [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
            if args.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        started = time.perf_counter()
        session = ort.InferenceSession(str(models[-1]), sess_options=options, providers=providers)
        report["active_providers"] = session.get_providers()
        report["active_provider_options"] = session.get_provider_options()
        if args.device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA initialization failed; refusing to benchmark a CPU fallback")
        session.disable_fallback()
        model = Kokoro.from_session(session, str(models[-1].parent / "voices.bin"))
        report["model_load_seconds"] = round(time.perf_counter() - started, 3)
        report["runs"] = []
        for _ in range(2):
            started = time.perf_counter()
            audio, rate = model.create("The capital of Sweden is Stockholm.", voice="af_heart", lang="en-us")
            report["runs"].append(
                {
                    "synthesis_seconds": round(time.perf_counter() - started, 3),
                    "audio_seconds": round(len(audio) / rate, 3),
                }
            )
        events = json.loads(Path(session.end_profiling()).read_text())
        counts: dict[str, int] = {}
        for event in events:
            provider = event.get("args", {}).get("provider")
            if provider:
                counts[provider] = counts.get(provider, 0) + 1
        report["executed_kernel_events"] = counts
        if args.device == "cuda" and not counts.get("CUDAExecutionProvider"):
            raise RuntimeError("No CUDA kernel execution recorded; GPU inference is unverified")
        report["inference_tested"] = True
        report["ok"] = True
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report.update(ok=False, error=type(exc).__name__, detail=str(exc)[:500])
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
