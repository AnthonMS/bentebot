import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = """class Processor:
    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def get_audio_buffer_end_time(self):
        return self.end

    def start_silence(self):
        if self.audio_buffer.size == 0:
            return [], self.get_audio_buffer_end_time()
        return self.process_iter()

    def process_iter(self):
        current_audio_processed_upto = self.get_audio_buffer_end_time()
        self.calls += 1
        committed_tokens = self.transcript_buffer.flush()
        return committed_tokens, current_audio_processed_upto
"""


def patch():
    return runpy.run_path(str(Path(__file__).parents[1] / "deploy/speech/tune_silence.py"))[
        "tuned_silence_source"
    ]


@pytest.mark.parametrize(
    "enabled,prior_decode,pending,append,expected_calls",
    [
        (True, True, False, False, 1),
        (False, True, False, False, 2),
        (True, False, False, False, 1),
        (True, True, True, False, 2),
        (True, True, False, True, 2),
        (True, True, True, True, 2),
    ],
)
def test_silence_reuse_requires_decoded_unchanged_audio_and_no_provisional_words(
    monkeypatch, enabled, prior_decode, pending, append, expected_calls
):
    monkeypatch.setenv("WLK_REUSE_COMMITTED_SILENCE", str(enabled).lower())
    scope = {"logger": SimpleNamespace(info=lambda *args: None)}
    exec(patch()(SOURCE), scope)
    processor = scope["Processor"]()
    processor.end = 2.4
    processor.calls = 0
    processor.committed = ["already verified"]
    processor.audio_buffer = SimpleNamespace(size=38400)
    processor.transcript_buffer = SimpleNamespace(buffer=[], flush=lambda: ["verified"])
    if prior_decode:
        processor.process_iter()
    if pending:
        processor.transcript_buffer.buffer = ["not yet verified"]
    if append:
        processor.end += 0.1
    result, end = processor.start_silence()
    assert processor.calls == expected_calls and end == processor.end
    if expected_calls == 1 and prior_decode:
        assert result == []  # No duplicate committed words or invented final.
    else:
        assert result == ["verified"]


def test_patch_rejects_changed_or_already_patched_processor():
    tune = patch()
    with pytest.raises(ValueError, match="changed"):
        tune(SOURCE.replace("self.transcript_buffer.flush()", "self.other.flush()"))
    with pytest.raises(ValueError, match="changed"):
        tune(tune(SOURCE))


@pytest.mark.parametrize("new_audio", [False, True])
def test_reuse_is_disabled_without_committed_words_or_after_new_input(monkeypatch, new_audio):
    monkeypatch.setenv("WLK_REUSE_COMMITTED_SILENCE", "true")
    scope = {
        "logger": SimpleNamespace(info=lambda *args: None),
        "np": SimpleNamespace(append=lambda old, audio: old),
    }
    exec(patch()(SOURCE), scope)
    processor = scope["Processor"]()
    processor.end = 2.4
    processor.calls = 0
    processor.audio_buffer = SimpleNamespace(size=38400)
    processor.committed = ["stable"] if new_audio else []
    processor.transcript_buffer = SimpleNamespace(buffer=[], flush=lambda: ["verified"])
    processor.process_iter()
    if new_audio:
        processor.insert_audio_chunk([0.5])  # Even if the reported end has not advanced yet.
    processor.start_silence()
    assert processor.calls == 2
