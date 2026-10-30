from __future__ import annotations


def discord_chunks(text: str, limit: int = 1900) -> list[str]:
    """Split Discord output, closing and reopening fenced code across messages."""
    chunks, current = [], ""
    fence = ""
    for line in text.splitlines(keepends=True):
        is_fence = line.lstrip().startswith("```")
        # Hard-wrap exceptionally long lines while keeping room for fence markers.
        pieces = [line[i : i + limit - 40] for i in range(0, len(line), limit - 40)] or [""]
        for piece in pieces:
            if len(current) + len(piece) + (5 if fence else 0) > limit:
                chunks.append(current.rstrip() + ("\n```" if fence else ""))
                current = f"{fence}\n" if fence else ""
            current += piece
        if is_fence:
            fence = "" if fence else line.strip()
    if current:
        if fence:
            current = current.rstrip() + "\n```"
        chunks.append(current)
    return chunks
