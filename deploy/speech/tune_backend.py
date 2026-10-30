"""Apply a deliberately narrow patch to the pinned external WhisperLiveKit backend.

This is part of the separate speech images, never the bot process. Fail at image build
if upstream changes the expected contract instead of silently applying a partial patch.
"""

from __future__ import annotations

from pathlib import Path


def tuned_source(source: str) -> str:
    beam = "            beam_size=5,"
    context = "            condition_on_previous_text=True,\n            **self.transcribe_kargs,"
    compute = '        compute_type = "auto" # Allow CTranslate2 to decide faster compute type'
    if source.count(beam) != 1 or source.count(context) != 1 or source.count(compute) != 1:
        raise ValueError("Pinned WhisperLiveKit backend changed; review the tuning patch")
    source = source.replace(beam, '            beam_size=int(os.environ.get("WLK_BEAM_SIZE", "1")),')
    source = source.replace(compute, '        compute_type = os.environ.get("WLK_COMPUTE_TYPE", "auto")')
    source = source.replace(
        context,
        "            condition_on_previous_text=False,\n"
        '            multilingual=self.original_language is None and os.environ.get("WLK_MULTILINGUAL", "true").lower() == "true",\n'
        "            **self.transcribe_kargs,",
    )
    return "import os\n" + source


def main():
    import importlib.util

    spec = importlib.util.find_spec("whisperlivekit")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("WhisperLiveKit is not installed")
    target = Path(next(iter(spec.submodule_search_locations))) / "local_agreement" / "backends.py"
    target.write_text(tuned_source(target.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
