"""Guard a confirmed LiteLLM logging crash without changing streamed responses."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Final

ORIGINAL = "max((choice.index + 1 for chunk in chunks for choice in chunk.choices), default=0)"
GUARDED = (
    "max((choice.index + 1 for chunk in chunks for choice in chunk.choices "
    "if type(choice.index) is int and choice.index >= 0), default=0)"
)


def patched_source(source: str) -> str:
    """Fail closed on unfamiliar source; accept an already applied exact guard."""
    tree = ast.parse(source)
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_chat_stream_texts"
    ]
    if len(matches) != 1:
        raise ValueError("Expected one _chat_stream_texts helper")
    function = ast.get_source_segment(source, matches[0])
    if GUARDED in function and ORIGINAL not in source:
        return source
    if source.count(ORIGINAL) != 1 or function.count(ORIGINAL) != 1:
        raise ValueError("Installed helper differs from the reviewed source; no file changed")
    result = source.replace(ORIGINAL, GUARDED, 1)
    compile(result, "served_output_texts.py", "exec")
    return result


def helper_namespace(source: str) -> dict:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_chat_stream_texts", "_chat_stream_choice_text"}
    ]
    if len(functions) != 2:
        raise ValueError("Expected both reviewed stream logging helpers")
    namespace = {"Sequence": Sequence, "Final": Final, "ModelResponseStream": object}
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), "logging-helper-check", "exec"), namespace)
    return namespace


def verify(source: str) -> None:
    helper = helper_namespace(source)["_chat_stream_texts"]

    def chunk(*items):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(index=index, delta=SimpleNamespace(content=content))
                for index, content in items
            ]
        )

    cases = [
        ([], None),
        ([chunk()], None),
        ([chunk((None, None))], None),
        ([chunk((0, "hello")), chunk((None, None))], ("hello",)),
        ([chunk((0, None)), chunk((None, None))], None),
        ([chunk((0, "a"), (1, "b")), chunk((0, "c"), (1, "d"))], ("ac", "bd")),
        ([chunk((None, "unassigned")), chunk((0, "assigned"))], ("assigned",)),
        ([chunk((-1, None), (True, None), ("0", None), (1.5, None))], None),
    ]
    for chunks, expected in cases:
        if helper(chunks) != expected:
            raise AssertionError("Stream logging regression check failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", type=Path, help="Restore a reviewed original backup")
    args = parser.parse_args()
    if args.apply and args.restore:
        parser.error("Choose apply or restore")
    module = importlib.import_module("litellm.litellm_core_utils.served_output_texts")
    target = Path(module.__file__).resolve()
    before = target.read_bytes()
    source = before.decode("utf-8")
    if args.restore:
        # Refuse to overwrite an unrelated vendor file or restore unreviewed source.
        if GUARDED not in source:
            raise ValueError("Current source does not contain this exact guard")
        original = args.restore.read_bytes()
        if patched_source(original.decode("utf-8")) != source:
            raise ValueError("Backup does not match the installed guarded source")
        after = original
    else:
        updated = patched_source(source)
        verify(updated)
        # Also exercise the actual installed LiteLLM model classes.
        real_chunk = module.ModelResponseStream(
            choices=[{"index": 0, "delta": {"content": None}, "finish_reason": "tool_calls"}]
        )
        real_chunk.choices[0].index = None
        if helper_namespace(updated)["_chat_stream_texts"]([real_chunk]) is not None:
            raise AssertionError("Real null-index tool chunk check failed")
        after = updated.encode("utf-8")
        if not args.apply:
            print("Preflight passed: eight logging cases and actual LiteLLM null-index chunk")
            return
    if after != before:
        # Backup stays beside the installed file and survives an ordinary restart.
        backup = target.with_suffix(target.suffix + ".before-bentebot-index-guard")
        if not args.restore and not backup.exists():
            backup.write_bytes(before)
        temporary = target.with_suffix(target.suffix + ".bentebot-tmp")
        temporary.write_bytes(after)
        os.chmod(temporary, target.stat().st_mode)
        os.replace(temporary, target)
    print("Restored original helper" if args.restore else "Applied stream logging index guard")
    print("SHA256=" + hashlib.sha256(after).hexdigest())


if __name__ == "__main__":
    main()
