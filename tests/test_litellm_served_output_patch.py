import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

PATCH = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/patch_litellm_served_output.py"))

SOURCE = """from __future__ import annotations
def _chat_stream_choice_text(chunks, index):
    contents = tuple(choice.delta.content for chunk in chunks for choice in chunk.choices
                     if choice.index == index and isinstance(choice.delta.content, str))
    return "".join(contents) if contents else None

def _chat_stream_texts(chunks):
    choice_count = max((choice.index + 1 for chunk in chunks for choice in chunk.choices), default=0)
    texts = tuple(_chat_stream_choice_text(chunks, index) for index in range(choice_count))
    return texts if any(text is not None for text in texts) else None
"""


def test_reproduces_reported_vendor_crash_and_repairs_only_logging():
    chunks = [SimpleNamespace(choices=[SimpleNamespace(index=None, delta=SimpleNamespace(content=None))])]
    original = PATCH["helper_namespace"](SOURCE)["_chat_stream_texts"]
    with pytest.raises(TypeError, match="NoneType"):
        original(chunks)
    updated = PATCH["patched_source"](SOURCE)
    assert updated.replace(PATCH["GUARDED"], PATCH["ORIGINAL"]) == SOURCE
    assert PATCH["helper_namespace"](updated)["_chat_stream_texts"](chunks) is None
    assert chunks[0].choices[0].index is None
    PATCH["verify"](updated)


def test_preserves_text_and_choice_order_with_null_terminal_chunk():
    def chunk(index, text):
        return SimpleNamespace(choices=[SimpleNamespace(index=index, delta=SimpleNamespace(content=text))])

    helper = PATCH["helper_namespace"](PATCH["patched_source"](SOURCE))["_chat_stream_texts"]
    assert helper([chunk(1, "B"), chunk(0, "A"), chunk(None, None), chunk(0, "C")]) == (
        "AC",
        "B",
    )


def test_patch_is_idempotent():
    updated = PATCH["patched_source"](SOURCE)
    assert PATCH["patched_source"](updated) == updated


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace(PATCH["ORIGINAL"], "0"),
        SOURCE + "\nother = " + PATCH["ORIGINAL"],
        SOURCE.replace("def _chat_stream_texts(", "def changed_vendor_function("),
    ],
)
def test_unknown_vendor_source_fails_before_writing(source):
    with pytest.raises(ValueError):
        PATCH["patched_source"](source)
