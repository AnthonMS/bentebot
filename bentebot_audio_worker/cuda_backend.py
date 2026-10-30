"""FP32 ONNX classifiers with verified CUDA computation and bounded arenas."""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path


def cuda_options(memory_mb):
    if not 64 <= memory_mb <= 4096:
        raise ValueError("GPU arena budget must be 64-4096 MiB.")
    return {
        "device_id": 0,
        "gpu_mem_limit": memory_mb * 1024 * 1024,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT",
        "cudnn_conv_use_max_workspace": "0",
        "do_copy_in_default_stream": "1",
        "use_tf32": "0",
    }


def verified_session(ort, path, feed, memory_mb):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA audio requested, but the CUDA execution provider is unavailable.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    # Shape/frontend operations may need CPU. Require real CUDA compute in each
    # model rather than assuming the advertised provider proves GPU execution.
    options.enable_profiling = True
    with tempfile.TemporaryDirectory(prefix="bentebot-cuda-") as folder:
        options.profile_file_prefix = str(Path(folder) / "startup")
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", cuda_options(memory_mb)), "CPUExecutionProvider"],
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDA audio initialization fell back to CPU.")
        session.disable_fallback()
        session.run(None, {session.get_inputs()[0].name: feed})
        profile = Path(session.end_profiling())
        events = json.loads(profile.read_text())
        gpu_compute = sum(
            event.get("args", {}).get("provider") == "CUDAExecutionProvider"
            and event.get("args", {}).get("op_name") in {"Conv", "FusedConv", "MatMul", "Gemm"}
            for event in events
        )
        providers = Counter(event.get("args", {}).get("provider") for event in events)
        if not gpu_compute:
            raise RuntimeError("The audio model did not execute CUDA compute during startup.")
    return session, {"cuda_compute_nodes": gpu_compute, "cpu_nodes": providers["CPUExecutionProvider"]}


class CudaModels:
    def __init__(self, directory, ort, np, memory_mb=1024):
        if not 768 <= memory_mb <= 4096:
            raise ValueError("Combined sound/tone GPU arena budget must be 768-4096 MiB.")
        # One quarter for the small sound model, the rest for the tone backbone.
        # Arena limits do not include every driver/cuDNN allocation.
        sound_mb = max(128, memory_mb // 4)
        tone_mb = memory_mb - sound_mb
        self.sound, sound_status = verified_session(
            ort,
            directory / "yamnet.onnx",
            np.zeros(48000, dtype=np.float32),
            sound_mb,
        )
        tone_path = directory / "emotion2vec_plus_base.onnx"
        # The pinned backbone input is fixed by its manifest/export contract.
        self.tone, tone_status = verified_session(
            ort,
            tone_path,
            np.zeros((1, 48000), dtype=np.float32),
            tone_mb,
        )
        with (directory / "yamnet_class_map.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 521 or [int(row["index"]) for row in rows] != list(range(521)):
            raise ValueError("Unexpected YAMNet class map.")
        self.labels = [row["display_name"].lower() for row in rows]
        self.details = {
            "device": "cuda",
            "sound_provider": "CUDAExecutionProvider",
            "tone_provider": "CUDAExecutionProvider",
            "sound_startup": sound_status,
            "tone_startup": tone_status,
            "gpu_arena_budget_mb": memory_mb,
        }

    def classify(self, samples):
        scores = self.sound.run([self.sound.get_outputs()[0].name], {"waveform": samples})[0]
        if scores.ndim != 2 or scores.shape[1] != 521:
            raise ValueError("Unexpected YAMNet score shape.")
        return dict(zip(self.labels, scores.max(axis=0).tolist(), strict=True))
