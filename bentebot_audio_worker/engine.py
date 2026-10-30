"""Pinned YAMNet and emotion2vec ONNX inference without downloaded Python code."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

EVENTS = {
    "laughter": "laugh",
    "giggle": "laugh",
    "chuckle, chortle": "laugh",
    "snicker": "laugh",
    "belly laugh": "laugh",
    "cough": "cough",
    "sneeze": "sneeze",
    "gasp": "gasp",
    "sigh": "sigh",
    "screaming": "scream",
    "crying, sobbing": "cry",
}


class Engine:
    def __init__(self, directory, *, device="cpu", gpu_memory_mb=1024):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        if device not in {"cpu", "cuda"}:
            raise ValueError("Audio device must be cpu or cuda.")
        self.device = device
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        for item in manifest:
            path = directory / item["name"]
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError("Audio model checksum does not match its manifest.")
        if device == "cuda":
            from .cuda_backend import CudaModels

            self.cuda = CudaModels(directory, ort, np, gpu_memory_mb)
            self.tone = self.cuda.tone
            self.details = self.cuda.details
        else:
            import mediapipe as mp

            self.mp = mp
            options = mp.tasks.audio.AudioClassifierOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(directory / "yamnet.tflite")),
                max_results=521,
            )
            self.sound = mp.tasks.audio.AudioClassifier.create_from_options(options)
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 2
            session_options.inter_op_num_threads = 1
            self.tone = ort.InferenceSession(
                str(directory / "emotion2vec_plus_base.onnx"),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self.details = {
                "device": "cpu",
                "sound_provider": "MediaPipeCPU",
                "tone_provider": "CPUExecutionProvider",
            }
        head = json.loads((directory / "emotion2vec_head.json").read_text())
        self.labels = head["labels"]
        self.weight = np.asarray(head["weight"], dtype=np.float32)
        self.bias = np.asarray(head["bias"], dtype=np.float32)
        if self.weight.shape != (9, 768) or self.bias.shape != (9,):
            raise ValueError("Unexpected emotion classifier head shape.")

    def analyze(self, pcm, tone=True):
        np = self.np
        began = time.monotonic()
        if len(pcm) % 2 or not 16000 <= len(pcm) <= 192000:
            raise ValueError("Audio must be 0.5-6 seconds of 16kHz signed mono PCM16.")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms < 0.003:
            return {
                "events": [],
                "tone": "unknown",
                "tone_score": 0,
                "speech_score": 0,
                "tone_status": "low_volume",
                "tone_candidate": "unknown",
                "top_sound": None,
                "device": self.device,
                "elapsed_ms": round((time.monotonic() - began) * 1000, 1),
            }
        if self.device == "cuda":
            labels = self.cuda.classify(samples)
        else:
            data = self.mp.tasks.components.containers.AudioData.create_from_array(samples, 16000)
            labels = {}
            for result in self.sound.classify(data):
                for group in result.classifications:
                    for category in group.categories:
                        label = category.category_name.lower()
                        labels[label] = max(labels.get(label, 0), float(category.score))
        events = {}
        for label, event in EVENTS.items():
            events[event] = max(events.get(event, 0), labels.get(label, 0))
        speech = max(labels.get(k, 0) for k in ("speech", "conversation", "narration, monologue"))
        affect, confidence, margin = "unknown", 0.0, 0.0
        candidate = "unknown"
        tone_status = (
            "not_requested" if not tone else "insufficient_audio" if len(samples) < 32000 else "no_speech"
        )
        if tone and len(samples) >= 32000 and speech >= 0.35:
            features = self.tone.run(None, {self.tone.get_inputs()[0].name: samples.reshape(1, -1)})[0]
            logits = self.weight @ features[0].mean(axis=0) + self.bias
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            ranking = np.argsort(probabilities)
            confidence = float(probabilities[ranking[-1]])
            margin = float(probabilities[ranking[-1]] - probabilities[ranking[-2]])
            candidate = self.labels[int(ranking[-1])]
            tone_status = (
                "low_confidence" if confidence < 0.7 else "low_margin" if margin < 0.2 else "candidate"
            )
            if confidence >= 0.7 and margin >= 0.2:
                affect = candidate
        top_sound = max(events, key=events.get)
        return {
            "events": [
                {"label": label, "score": round(score, 4)} for label, score in events.items() if score >= 0.65
            ],
            "tone": affect,
            "tone_score": round(confidence, 4),
            "tone_margin": round(margin, 4),
            "tone_status": tone_status,
            "tone_candidate": candidate,
            "top_sound": {"label": top_sound, "score": round(events[top_sound], 4)},
            "device": self.device,
            "speech_score": round(speech, 4),
            "elapsed_ms": round((time.monotonic() - began) * 1000, 1),
        }
