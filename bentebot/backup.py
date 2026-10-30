"""Consistent SQLite backup without starting the bot or migrating its schema."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path


def backup_database(source: str, destination: str) -> Path:
    original, target = Path(source).resolve(), Path(destination).resolve()
    if not original.is_file():
        raise ValueError("The source database does not exist")
    if original == target or target.exists():
        raise ValueError("Choose a new backup file; existing files are never overwritten")
    if not target.parent.is_dir():
        raise ValueError("The backup directory must already exist")
    # Exclusive reservation prevents an accidental concurrent backup from overwriting a file.
    with target.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(original.as_uri() + "?mode=ro", uri=True, timeout=10)) as src:
            with closing(sqlite3.connect(target)) as dst:
                src.backup(dst, pages=256, sleep=0.05)
                if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Backup integrity check failed")
        return target
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    try:
        result = backup_database(args.source, args.destination)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Backup failed: {exc}\n")
    print(f"Consistent database backup created: {result.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
