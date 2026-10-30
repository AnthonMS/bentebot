from __future__ import annotations

import json
import re
import sqlite3
import time
import zlib
from dataclasses import asdict
from pathlib import Path

from .domain import Event, Source


class Store:
    """One bot process owns this local SQLite database. No raw audio is persisted."""

    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        has_version = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        row = self.db.execute("SELECT MAX(version) FROM schema_version").fetchone() if has_version else None
        version = (row[0] or 0) if row else 0
        if version not in {0, 1, 2, 3}:
            self.db.close()
            raise ValueError("Unsupported database schema; use a matching Bentebot version")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, guild_id INTEGER, owner_id INTEGER,
                channel_id INTEGER NOT NULL, author_id INTEGER NOT NULL,
                created_at REAL NOT NULL, payload BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_time ON events(created_at);
            CREATE INDEX IF NOT EXISTS events_guild ON events(guild_id, created_at);
            CREATE INDEX IF NOT EXISTS events_owner ON events(owner_id, created_at);
            CREATE TABLE IF NOT EXISTS event_sources (
                event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                channel_id INTEGER NOT NULL,
                PRIMARY KEY(event_id, channel_id)
            );
            CREATE INDEX IF NOT EXISTS sources_channel ON event_sources(channel_id);
            CREATE TABLE IF NOT EXISTS settings (scope TEXT, key TEXT, value TEXT NOT NULL,
                PRIMARY KEY(scope, key));
        """)
        if version < 2:
            with self.db:
                for event_id, payload in self.db.execute("SELECT id,payload FROM events"):
                    event = self._decode(payload)
                    self.db.executemany(
                        "INSERT OR IGNORE INTO event_sources VALUES(?,?)",
                        [(event_id, origin.channel_id) for origin in (event.source, *event.provenance)],
                    )
                self.db.execute("DELETE FROM schema_version")
                self.db.execute("INSERT INTO schema_version VALUES(2)")
        self.db.create_function("bentebot_search_text", 1, self._index_payload, deterministic=True)
        try:
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                # Traditional contentless FTS works on Debian Bookworm's SQLite 3.40 as
                # well as newer Windows SQLite. No original text or word positions are stored.
                self.db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS event_search USING fts5(
                    terms, content='', detail=none, tokenize='unicode61 remove_diacritics 2')""")
                if version < 3:
                    self.db.execute("INSERT INTO event_search(event_search) VALUES('delete-all')")
                    for rowid, payload in self.db.execute("SELECT rowid,payload FROM events"):
                        self.db.execute(
                            "INSERT INTO event_search(rowid,terms) VALUES(?,?)",
                            (
                                rowid,
                                self._index_payload(payload),
                            ),
                        )
                self.db.execute("""CREATE TRIGGER IF NOT EXISTS events_search_insert AFTER INSERT ON events BEGIN
                    INSERT INTO event_search(rowid,terms) VALUES(new.rowid,bentebot_search_text(new.payload)); END""")
                self.db.execute("""CREATE TRIGGER IF NOT EXISTS events_search_delete AFTER DELETE ON events BEGIN
                    INSERT INTO event_search(event_search,rowid,terms)
                    VALUES('delete',old.rowid,bentebot_search_text(old.payload)); END""")
                self.db.execute("""CREATE TRIGGER IF NOT EXISTS events_search_update AFTER UPDATE ON events BEGIN
                    INSERT INTO event_search(event_search,rowid,terms)
                    VALUES('delete',old.rowid,bentebot_search_text(old.payload));
                    INSERT INTO event_search(rowid,terms) VALUES(new.rowid,bentebot_search_text(new.payload)); END""")
                self.db.execute("DELETE FROM schema_version")
                self.db.execute("INSERT INTO schema_version VALUES(3)")
        except (sqlite3.Error, ValueError, TypeError, UnicodeError, zlib.error) as exc:
            self.db.close()
            raise ValueError(
                "Search index initialization failed; verify SQLite FTS5 support and database integrity"
            ) from exc

    def save(self, event: Event) -> None:
        payload = zlib.compress(json.dumps(asdict(event), ensure_ascii=False).encode(), level=6)
        with self.db:
            self.db.execute(
                """INSERT INTO events VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                    guild_id=excluded.guild_id,owner_id=excluded.owner_id,channel_id=excluded.channel_id,
                    author_id=excluded.author_id,created_at=excluded.created_at,payload=excluded.payload""",
                (
                    event.id,
                    event.source.guild_id,
                    event.source.owner_id,
                    event.source.channel_id,
                    event.author_id,
                    event.created_at,
                    payload,
                ),
            )
            self.db.execute("DELETE FROM event_sources WHERE event_id=?", (event.id,))
            self.db.executemany(
                "INSERT OR IGNORE INTO event_sources VALUES(?,?)",
                [(event.id, origin.channel_id) for origin in (event.source, *event.provenance)],
            )

    def event(self, event_id: str) -> Event | None:
        """Fetch one event by ID; callers must check its source permissions."""
        row = self.db.execute("SELECT payload FROM events WHERE id=?", (event_id,)).fetchone()
        return self._decode(row[0]) if row else None

    @staticmethod
    def _decode(payload: bytes) -> Event:
        data = json.loads(zlib.decompress(payload))

        def source(value):
            value["audience_ids"] = tuple(value.get("audience_ids", ()))
            return Source(**value)

        data["source"] = source(data["source"])
        data["provenance"] = tuple(source(value) for value in data["provenance"])
        return Event(**data)

    @staticmethod
    def _terms(text: str) -> list[str]:
        # Identical normalization is used for inserts, deletes and query terms. Changing it
        # requires rebuilding the index in a future schema migration.
        text = re.sub(r"data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[@#][!&]?\d+>", " ", text)
        return [
            term.casefold()
            for term in re.findall(r"[^\W_]+", text)
            if 2 <= len(term) <= 64 and not term.isnumeric()
        ]

    @classmethod
    def _index_payload(cls, payload: bytes) -> str:
        event = cls._decode(payload)
        if event.source.kind == "screen":
            return ""  # Raw frame/source events are never searchable; derived text inherits ACLs.
        text = "\n".join((event.content, event.generated, event.spoken, event.written))
        return " ".join(sorted(set(cls._terms(text))))

    def search(
        self, query: str, guild_id: int | None, owners: frozenset[int], limit: int = 50
    ) -> list[Event]:
        """Scoped keyword candidates only. Callers MUST apply current source/provenance ACLs."""
        stopwords = {
            "the",
            "what",
            "when",
            "where",
            "how",
            "about",
            "remember",
            "did",
            "you",
            "we",
            "me",
            "my",
            "our",
            "an",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "for",
            "in",
            "on",
            "at",
            "it",
            "that",
            "this",
            "can",
            "could",
            "would",
            "please",
            "do",
            "does",
            "as",
            "with",
            "jeg",
            "du",
            "vi",
            "det",
            "den",
            "de",
            "en",
            "et",
            "og",
            "eller",
            "er",
            "var",
            "har",
            "have",
            "hvad",
            "hvor",
            "hvordan",
            "kan",
            "om",
            "til",
            "på",
            "som",
        }
        terms = list(dict.fromkeys(term for term in self._terms(query[:1000]) if term not in stopwords))[:12]
        if not terms or limit <= 0:
            return []
        clauses, args = [], []
        if guild_id is not None:
            clauses.append("e.guild_id=?")
            args.append(guild_id)
        if owners:
            clauses.append(f"e.owner_id IN ({','.join('?' for _ in owners)})")
            args.extend(sorted(owners))
        if not clauses:
            return []
        # Quoted extracted tokens cannot inject FTS operators, column names or SQL syntax.
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.db.execute(
            f"""SELECT e.payload FROM event_search JOIN events e ON e.rowid=event_search.rowid
                WHERE event_search MATCH ? AND ({" OR ".join(clauses)})
                ORDER BY bm25(event_search),e.created_at DESC,e.rowid DESC LIMIT ?""",
            [expression, *args, min(limit, 200)],
        ).fetchall()
        return [self._decode(row[0]) for row in rows]

    def recent(self, guild_id: int | None, owners: frozenset[int], limit: int = 500) -> list[Event]:
        # Include only this server and candidate recipients' DMs; ACL filtering follows.
        clauses, args = [], []
        if guild_id is not None:
            clauses.append("guild_id=?")
            args.append(guild_id)
        if owners:
            clauses.append(f"owner_id IN ({','.join('?' for _ in owners)})")
            args.extend(sorted(owners))
        if not clauses:
            return []
        rows = self.db.execute(
            f"SELECT payload FROM events WHERE {' OR '.join(clauses)} ORDER BY created_at DESC, rowid DESC LIMIT ?",
            [*args, limit],
        ).fetchall()
        return [self._decode(row[0]) for row in reversed(rows)]

    def set(self, scope: str, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES(?,?,?)", (scope, key, value))

    def get(self, scope: str, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM settings WHERE scope=? AND key=?", (scope, key)).fetchone()
        return row[0] if row else default

    def set_many(self, values) -> None:
        """Persist related settings and ID allocation in one transaction."""
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO settings VALUES(?,?,?)", values)

    def remove_setting(self, scope: str, key: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM settings WHERE scope=? AND key=?", (scope, key))

    def forget_channel(self, channel_id: int) -> int:
        # A derived reply/summary can itself contain information from a forgotten source.
        with self.db:
            cursor = self.db.execute(
                """DELETE FROM events WHERE id IN
                (SELECT event_id FROM event_sources WHERE channel_id=?)""",
                (channel_id,),
            )
        return cursor.rowcount

    def delete(self, event_id: str) -> None:
        # Discord deletion also removes derivatives that inherited the same channel's content.
        with self.db:
            self.db.execute("DELETE FROM events WHERE id=?", (event_id,))

    def prune(self, retention_days: int) -> int:
        with self.db:
            cursor = self.db.execute(
                "DELETE FROM events WHERE created_at<?", (time.time() - retention_days * 86400,)
            )
        return cursor.rowcount

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def close(self) -> None:
        self.db.close()
