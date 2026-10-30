import sqlite3

import pytest

from bentebot.backup import backup_database
from bentebot.domain import Event, Source
from bentebot.storage import Store


def test_backup_preserves_wal_history_and_search_while_database_is_open(tmp_path):
    source, destination = tmp_path / "live.sqlite3", tmp_path / "backup.sqlite3"
    store = Store(str(source))
    store.save(Event(Source("channel", 10, 1), 100, "Me", "unraid waveform issue"))
    try:
        backup_database(str(source), str(destination))
        copied = Store(str(destination))
        try:
            assert copied.count() == 1
            assert copied.search("waveform", 1, frozenset())[0].content == "unraid waveform issue"
        finally:
            copied.close()
        with pytest.raises(ValueError, match="never overwritten"):
            backup_database(str(source), str(destination))
    finally:
        store.close()


def test_backup_does_not_migrate_an_older_schema(tmp_path):
    source, destination = tmp_path / "old.sqlite3", tmp_path / "backup.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE schema_version(version INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO schema_version VALUES(2)")
    backup_database(str(source), str(destination))
    for path in (source, destination):
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 2


def test_missing_source_does_not_create_empty_database(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        backup_database(str(tmp_path / "missing"), str(tmp_path / "backup"))
    assert list(tmp_path.iterdir()) == []
