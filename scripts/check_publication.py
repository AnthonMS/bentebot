"""Check tracked paths before publishing. Use a secret scanner for file contents too."""

from __future__ import annotations

import subprocess
from pathlib import PurePosixPath


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    forbidden = []
    for name in filter(None, tracked):
        path = PurePosixPath(name.lower())
        if (
            any(part in {"ai", "artifacts", "data", "secrets", ".test-deps", ".venv"} for part in path.parts)
            or path.suffix in {".md", ".markdown"}
            and name.lower() != "readme.md"
            or (path.name == ".env" or path.suffix == ".env" or path.name.startswith(".env."))
            and path.name != "example.env"
            or path.suffix in {".private", ".pem", ".key", ".p12", ".pfx", ".sqlite3", ".db"}
        ):
            forbidden.append(name)
    if forbidden:
        print("Private working files must not be tracked:\n" + "\n".join(forbidden))
        return 1
    print(
        "Tracked path policy passed: only the public README is Markdown; private working files are excluded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
