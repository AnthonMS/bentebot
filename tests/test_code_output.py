import pytest

from bentebot.code_output import OutputPart, render_discord_output


def test_small_code_has_language_and_preserves_indentation():
    parts = render_discord_output("Here is the example.\n```py\ndef hello():\n    return 1\n```\nDone.")
    assert [part.content for part in parts] == [
        "Here is the example.",
        "```python\ndef hello():\n    return 1\n```",
        "Done.",
    ]
    assert all(part.data is None for part in parts)


def test_large_source_is_an_attachment_with_exact_bytes_and_no_duplicate_code():
    code = "def hello():\r\n\treturn 'æøå 😀'  \r\n" * 100
    parts = render_discord_output("Example:\r\n```python\r\n" + code + "```\r\nAsk if you need details.")
    assert len(parts) == 3
    attachment = parts[1]
    assert attachment.filename == "snippet-1.py"
    assert attachment.data == code.encode("utf-8")
    assert "return" not in "".join(part.content for part in parts)


def test_long_code_line_is_not_wrapped_or_truncated():
    code = "value = '" + "x" * 4000 + "'\n"
    part = render_discord_output("```python\n" + code + "```")[0]
    assert part.filename == "snippet-1.py"
    assert part.data == code.encode()
    assert part.data.count(b"\n") == 1


def test_mixed_blocks_use_stable_safe_names_and_do_not_repeat_prose():
    large = "const value = 1;\n" * 100
    parts = render_discord_output("First.\n~~~python\nprint(1)\n~~~\nSecond.\n```js\n" + large + "```\nEnd.")
    assert [part.content for part in parts if part.filename is None] == [
        "First.",
        "```python\nprint(1)\n```",
        "Second.",
        "End.",
    ]
    assert [part.filename for part in parts if part.filename] == ["snippet-2.js"]


def test_variable_length_fences_preserve_nested_fences():
    code = "```python\nprint(1)\n```\n"
    part = render_discord_output("````markdown\n" + code + "`````\n")[0]
    assert part.filename is None
    assert part.content.startswith("````markdown\n")
    assert code in part.content
    assert part.content.endswith("````")


@pytest.mark.parametrize("opening,closing", [("~~~~", "~~~~~"), ("```", "```   "), ("`````", "``````")])
def test_matching_fence_character_and_minimum_length(opening, closing):
    part = render_discord_output(f"{opening}json\n{{}}\n{closing}\n")[0]
    assert part.filename is None and "{}" in part.content


def test_malformed_closing_is_retained_in_an_incomplete_source_attachment():
    code = "print(1)\n``\nthis was never closed\n"
    part = render_discord_output("```python\n" + code)[0]
    assert part.filename == "snippet-1.py" and part.data == code.encode()
    assert "incomplete" in part.content


def test_different_fence_does_not_close_source():
    code = "print(1)\n~~~\n"
    part = render_discord_output("```python\n" + code)[0]
    assert part.data == code.encode()


@pytest.mark.parametrize("info", ["../../evil.py", "C:\\Windows\\evil.ps1", "python/../../evil", "<@123>"])
def test_untrusted_info_cannot_choose_attachment_path(info):
    part = render_discord_output(f"```{info}\n" + "x" * 1600 + "\n```")[0]
    assert part.filename == "snippet-1.txt"


def test_prose_and_emoji_messages_stay_below_discord_limit():
    text = ("😀 words " * 1000) + "\nlast sentence"
    parts = render_discord_output(text)
    assert len(parts) > 1
    assert all(len(part.content.encode("utf-16-le")) // 2 <= 1900 for part in parts)
    assert all(part.filename is None for part in parts)


def test_inline_code_remains_prose_and_empty_input_is_empty():
    assert render_discord_output("Use `value` here.") == [OutputPart("Use `value` here.")]
    assert render_discord_output(" \n\t ") == []


def test_attachment_contract_rejects_path_traversal_and_missing_data():
    with pytest.raises(ValueError):
        OutputPart("Code", "../../file.py", b"print(1)")
    with pytest.raises(ValueError):
        OutputPart("Code", "snippet-1.py")
