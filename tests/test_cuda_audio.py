import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bentebot_audio_worker.cuda_backend import cuda_options, verified_session


def runtime(*, available=True, fallback=False, gpu_compute=True):
    sessions = []

    class Session:
        def __init__(self, path, *, sess_options, providers):
            self.options, self.providers = sess_options, providers
            self.fallback_disabled = False
            sessions.append(self)

        def get_providers(self):
            return ["CPUExecutionProvider"] if fallback else ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_inputs(self):
            return [SimpleNamespace(name="waveform")]

        def run(self, outputs, feed):
            assert self.fallback_disabled and feed == {"waveform": "synthetic"}

        def end_profiling(self):
            path = Path(self.options.profile_file_prefix + ".json")
            path.write_text(
                json.dumps(
                    [
                        {
                            "args": {
                                "provider": "CUDAExecutionProvider"
                                if gpu_compute
                                else "CPUExecutionProvider",
                                "op_name": "Conv",
                            }
                        },
                        {"args": {"provider": "CPUExecutionProvider", "op_name": "Shape"}},
                    ]
                )
            )
            return str(path)

    return SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider"] if available else ["CPUExecutionProvider"],
        SessionOptions=SimpleNamespace,
        InferenceSession=Session,
    ), sessions


@pytest.mark.parametrize("mode", ["unavailable", "fallback", "cpu_compute", "cuda"])
def test_cuda_startup_requires_real_gpu_compute_and_removes_profile(mode):
    ort, sessions = runtime(
        available=mode != "unavailable", fallback=mode == "fallback", gpu_compute=mode != "cpu_compute"
    )
    if mode == "cuda":
        session, details = verified_session(ort, "model.onnx", "synthetic", 256)
        assert details == {"cuda_compute_nodes": 1, "cpu_nodes": 1}
        assert not Path(session.options.profile_file_prefix).parent.exists()
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            verified_session(ort, "model.onnx", "synthetic", 256)
    if sessions:
        assert sessions[0].providers[0][1]["gpu_mem_limit"] == 256 * 1024**2


def test_gpu_options_use_small_pascal_compatible_workspace():
    options = cuda_options(768)
    assert options["cudnn_conv_algo_search"] == "DEFAULT"
    assert options["cudnn_conv_use_max_workspace"] == "0"
    assert options["arena_extend_strategy"] == "kSameAsRequested"
    assert options["use_tf32"] == "0"
    with pytest.raises(ValueError):
        cuda_options(0)
