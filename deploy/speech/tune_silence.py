"""Opt-in reuse of a fully committed, unchanged LocalAgreement silence boundary."""

from pathlib import Path


def tuned_silence_source(source: str) -> str:
    start = """    def start_silence(self):
        if self.audio_buffer.size == 0:
            return [], self.get_audio_buffer_end_time()
        return self.process_iter()
"""
    commit = "        committed_tokens = self.transcript_buffer.flush()\n"
    append = "        self.audio_buffer = np.append(self.audio_buffer, audio)\n"
    if (
        source.count(start) != 1
        or source.count(commit) != 1
        or source.count(append) != 1
        or "_silence_decoded_end" in source
    ):
        raise ValueError("Pinned LocalAgreement processor changed; review silence reuse")
    replacement = """    def start_silence(self):
        if self.audio_buffer.size == 0:
            return [], self.get_audio_buffer_end_time()
        end = self.get_audio_buffer_end_time()
        if (
            os.environ.get("WLK_REUSE_COMMITTED_SILENCE", "false").lower() == "true"
            and getattr(self, "_silence_decoded_end", None) == end
            and not self.transcript_buffer.buffer
            and self.committed
        ):
            logger.info("asr_silence_reuse audio_end=%.3f outcome=unchanged_committed", end)
            return [], end
        return self.process_iter()
"""
    source = (
        source.replace(start, replacement)
        .replace(commit, commit + "        self._silence_decoded_end = current_audio_processed_upto\n")
        .replace(append, append + "        self._silence_decoded_end = None\n")
    )
    return "import os\n" + source


def main():
    import importlib.util

    spec = importlib.util.find_spec("whisperlivekit")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("WhisperLiveKit is not installed")
    target = Path(next(iter(spec.submodule_search_locations))) / "local_agreement" / "online_asr.py"
    original = target.read_text(encoding="utf-8")
    patched = tuned_silence_source(original)
    compile(patched, str(target), "exec")
    target.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
