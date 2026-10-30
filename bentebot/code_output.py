"""Format generated prose/code for Discord without rewriting downloadable source."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MESSAGE_LIMIT = 1900
_INLINE_CODE_LIMIT = 1500
_LANGUAGES = {
    "py": ("python", "py"),
    "python": ("python", "py"),
    "js": ("javascript", "js"),
    "javascript": ("javascript", "js"),
    "jsx": ("jsx", "jsx"),
    "ts": ("typescript", "ts"),
    "typescript": ("typescript", "ts"),
    "tsx": ("tsx", "tsx"),
    "json": ("json", "json"),
    "yaml": ("yaml", "yaml"),
    "yml": ("yaml", "yaml"),
    "bash": ("bash", "sh"),
    "sh": ("bash", "sh"),
    "shell": ("bash", "sh"),
    "zsh": ("bash", "sh"),
    "powershell": ("powershell", "ps1"),
    "ps1": ("powershell", "ps1"),
    "c": ("c", "c"),
    "cpp": ("cpp", "cpp"),
    "c++": ("cpp", "cpp"),
    "cs": ("cs", "cs"),
    "csharp": ("cs", "cs"),
    "java": ("java", "java"),
    "go": ("go", "go"),
    "rust": ("rust", "rs"),
    "rs": ("rust", "rs"),
    "ruby": ("ruby", "rb"),
    "rb": ("ruby", "rb"),
    "php": ("php", "php"),
    "html": ("html", "html"),
    "css": ("css", "css"),
    "sql": ("sql", "sql"),
    "toml": ("toml", "toml"),
    "ini": ("ini", "ini"),
    "dockerfile": ("dockerfile", "dockerfile"),
    "makefile": ("makefile", "mk"),
    "xml": ("xml", "xml"),
    "md": ("markdown", "md"),
    "markdown": ("markdown", "md"),
    "diff": ("diff", "diff"),
    "kotlin": ("kotlin", "kt"),
    "swift": ("swift", "swift"),
    "vue": ("vue", "vue"),
    "svelte": ("svelte", "svelte"),
    "text": ("", "txt"),
    "txt": ("", "txt"),
}
_EXTENSIONS = {extension for _, extension in _LANGUAGES.values()}
_OPEN_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)(?:\r\n|\n|\r)?$")


def _message_length(text: str) -> int:
    # A conservative limit also handles non-BMP emoji on UTF-16-based clients.
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


@dataclass(frozen=True, slots=True)
class OutputPart:
    content: str
    filename: str | None = None
    data: bytes | None = None

    def __post_init__(self):
        if _message_length(self.content) > _MESSAGE_LIMIT:
            raise ValueError("Discord output content exceeds its message budget")
        if (self.filename is None) != (self.data is None):
            raise ValueError("An attachment requires both a filename and bytes")
        if self.filename is not None:
            match = re.fullmatch(r"snippet-[1-9]\d*\.([a-z0-9]+)", self.filename)
            if not match or match.group(1) not in _EXTENSIONS:
                raise ValueError("Attachment filenames must be generated safe snippet names")


def _prose_parts(text: str) -> list[OutputPart]:
    text = text.strip()
    parts = []
    while text:
        units = 0
        end = 0
        for character in text:
            units += 2 if ord(character) > 0xFFFF else 1
            if units > _MESSAGE_LIMIT:
                break
            end += 1
        if end < len(text):
            natural = max(text.rfind("\n", 0, end), text.rfind(" ", 0, end))
            if natural >= end // 2:
                end = natural
        value = text[:end].strip()
        if value:
            parts.append(OutputPart(value))
        text = text[end:].lstrip()
    return parts


def _code_part(code: str, info: str, number: int, *, closed: bool) -> OutputPart:
    language = info.split(maxsplit=1)[0].casefold() if info.strip() else ""
    tag, extension = _LANGUAGES.get(language, ("", "txt"))
    longest = max((len(run.group()) for run in re.finditer(r"`+", code)), default=0)
    fence = "`" * max(3, longest + 1)
    rendered = f"{fence}{tag}\n{code}"
    if not rendered.endswith(("\n", "\r")):
        rendered += "\n"
    rendered += fence
    if closed and len(code) <= _INLINE_CODE_LIMIT and _message_length(rendered) <= _MESSAGE_LIMIT:
        return OutputPart(rendered)
    suffix = " (incomplete code block)" if not closed else ""
    return OutputPart(
        f"Code snippet {number}{suffix}:",
        filename=f"snippet-{number}.{extension}",
        data=code.encode("utf-8"),
    )


def render_discord_output(text: str) -> list[OutputPart]:
    """Render small fenced code inline and large/incomplete code as exact UTF-8 attachments.

    Both backtick and tilde fences are supported. A close must use the same delimiter,
    at least the opening length, and no trailing non-whitespace characters. Anything
    between unmatched fences remains source rather than being silently discarded.
    """
    parts = []
    prose: list[str] = []
    code: list[str] | None = None
    delimiter, info = "", ""
    number = 0
    for line in text.splitlines(keepends=True):
        if code is not None:
            candidate = line.rstrip("\r\n")
            if re.fullmatch(
                r" {0,3}" + re.escape(delimiter[0]) + "{" + str(len(delimiter)) + r",}[ \t]*", candidate
            ):
                parts.append(_code_part("".join(code), info, number, closed=True))
                code = None
            else:
                code.append(line)
            continue
        opening = _OPEN_FENCE.fullmatch(line)
        # Backticks in a backtick fence's info string make it ordinary prose.
        if opening and not (opening["fence"][0] == "`" and "`" in opening["info"]):
            parts.extend(_prose_parts("".join(prose)))
            prose.clear()
            delimiter, info = opening["fence"], opening["info"]
            code = []
            number += 1
        else:
            prose.append(line)
    if code is not None:
        parts.append(_code_part("".join(code), info, number, closed=False))
    parts.extend(_prose_parts("".join(prose)))
    return parts
