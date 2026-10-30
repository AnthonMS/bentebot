import sqlite3

import pytest

from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.storage import Store

PUBLIC = Source("channel", 10, 1)
PRIVATE = Source("channel", 20, 1)
DM = Source("dm", 30, owner_id=100)


async def readable(user, source):
    return source == PUBLIC or (source == PRIVATE and user == 100)


def test_keyword_index_is_contentless_and_compressed_payload_is_authoritative(store):
    store.save(Event(PUBLIC, 100, "Me", "The zirconium spindle was calibrated", id="old"))
    assert [event.id for event in store.search("zirconium spindle", 1, frozenset())] == ["old"]
    assert store.db.execute("SELECT terms FROM event_search").fetchall() == [(None,)]
    assert not store.db.execute("SELECT 1 FROM sqlite_master WHERE name='event_search_content'").fetchone()
    assert store.db.execute("SELECT payload FROM events").fetchone()[0].startswith(b"x")


async def test_query_retrieves_old_history_beyond_recent_window_without_displacing_latest_turn(store):
    store.save(Event(PUBLIC, 100, "Me", "The zirconium spindle was calibrated at 72 degrees.", created_at=1))
    for i in range(510):
        store.save(Event(PUBLIC, 100, "Me", "ordinary unrelated update", created_at=10 + i))
    store.save(Event(PUBLIC, 100, "Me", "What did we decide about the zirconium spindle?", created_at=1000))
    memory = Memory(store, AccessPolicy(readable), 2400)
    audience = Audience(frozenset({100}), 1)
    ordinary, _ = await memory.context(audience)
    assert "72 degrees" not in str(ordinary)
    messages, sources = await memory.context(
        audience, query="What did we decide about the zirconium spindle?"
    )
    assert "72 degrees" in str(messages)
    assert "Retrieved earlier context" in str(messages)
    assert "What did we decide" in messages[-1]["content"]
    assert sum(len(message["content"]) for message in messages[1:]) <= 2400
    assert sources == (PUBLIC,)


async def test_current_turn_only_search_does_not_reduce_recent_context(store):
    for i in range(10):
        store.save(Event(PUBLIC, 100, "Me", f"update number {i}", created_at=i))
    store.save(Event(PUBLIC, 100, "Me", "unique current question", created_at=100))
    memory = Memory(store, AccessPolicy(readable), 1000)
    audience = Audience(frozenset({100}), 1)
    ordinary, origins = await memory.context(audience)
    queried, queried_origins = await memory.context(audience, query="unique current question")
    assert queried == ordinary and queried_origins == origins


async def test_search_and_context_filter_private_summaries_dm_and_derived_drafts(store):
    store.save(Event(PRIVATE, 100, "Me", "zirconium secret detail", created_at=1))
    store.save(Event(DM, 100, "Me", "zirconium DM detail", created_at=2))
    store.save(
        Event(
            PUBLIC,
            999,
            "Bot",
            "zirconium summary detail",
            kind="summary",
            provenance=(PRIVATE,),
            created_at=3,
        )
    )
    store.save(
        Event(
            PUBLIC,
            999,
            "Bot",
            "",
            kind="assistant",
            generated="zirconium private draft",
            delivery="voice",
            provenance=(PRIVATE,),
            created_at=4,
        )
    )
    memory = Memory(store, AccessPolicy(readable), 4000)
    group = Audience(frozenset({100, 200}), 1)
    assert await memory.search(group, "zirconium") == []
    assert "secret detail" not in str((await memory.context(group, query="zirconium"))[0])
    solo = Audience(frozenset({100}), 1)
    assert len(await memory.search(solo, "zirconium")) == 4
    assert "private draft" in str((await memory.context(solo, query="zirconium"))[0])


async def test_permission_revocation_removes_old_search_results(store):
    permitted = True

    async def check(user, source):
        return permitted

    store.save(Event(PRIVATE, 100, "Me", "zirconium private history"))
    memory = Memory(store, AccessPolicy(check), 4000)
    audience = Audience(frozenset({100}), 1)
    assert len(await memory.search(audience, "zirconium")) == 1
    permitted = False
    assert await memory.search(audience, "zirconium") == []


def test_edits_delete_old_tokens_and_update_provenance_index(store):
    store.save(Event(PUBLIC, 100, "Me", "oldkeyword", id="same", provenance=(PRIVATE,)))
    assert len(store.search("oldkeyword", 1, frozenset())) == 1
    store.save(Event(PUBLIC, 100, "Me", "newkeyword", id="same"))
    assert store.search("oldkeyword", 1, frozenset()) == []
    assert len(store.search("newkeyword", 1, frozenset())) == 1
    assert store.forget_channel(PRIVATE.channel_id) == 0
    store.delete("same")
    assert store.search("newkeyword", 1, frozenset()) == []


def test_forget_and_prune_remove_searchable_derivatives(store):
    store.save(Event(PRIVATE, 100, "Me", "secretkeyword", id="source"))
    store.save(Event(PUBLIC, 999, "Bot", "secretkeyword derived", id="derived", provenance=(PRIVATE,)))
    assert store.forget_channel(PRIVATE.channel_id) == 2
    assert store.search("secretkeyword", 1, frozenset()) == []
    store.save(Event(PUBLIC, 100, "Me", "expiredkeyword", created_at=1))
    assert store.prune(90) == 1
    assert store.search("expiredkeyword", 1, frozenset()) == []


@pytest.mark.parametrize(
    "query", ['" OR * NEAR(:)', "'; DROP TABLE events;--", "\0\0\0", "x" * 5000, "what did we do"]
)
def test_fts_syntax_and_sql_are_never_accepted_from_query_text(store, query):
    store.save(Event(PUBLIC, 100, "Me", "ordinary history"))
    assert isinstance(store.search(query, 1, frozenset()), list)
    assert store.count() == 1


def test_other_guilds_and_unrelated_dms_are_not_candidates(store):
    store.save(Event(Source("channel", 90, 2), 100, "Me", "zirconium elsewhere"))
    store.save(Event(Source("dm", 91, owner_id=999), 999, "Other", "zirconium elsewhere"))
    assert store.search("zirconium", 1, frozenset({100})) == []


def test_visual_sources_and_data_uris_are_not_indexed(store):
    screen = Source("screen", 10, 1, owner_id=100, session_id="session", audience_ids=(100,))
    store.save(Event(screen, 100, "Me", "rawframekeyword data:image/png;base64,YWJjZGVmZ2hpamtsbW5vcA=="))
    store.save(Event(PUBLIC, 100, "Me", "data:image/png;base64,YWJjZGVmZ2hpamtsbW5vcA=="))
    assert store.search("rawframekeyword", 1, frozenset({100})) == []
    assert store.search("YWJjZGVmZ2hpamtsbW5vcA", 1, frozenset({100})) == []


def downgrade_index_to_v2(db):
    db.db.executescript("""
        DROP TRIGGER events_search_insert;
        DROP TRIGGER events_search_update;
        DROP TRIGGER events_search_delete;
        DROP TABLE event_search;
        UPDATE schema_version SET version=2;
    """)


def test_v2_migration_backfills_search_without_rewriting_compressed_history(tmp_path):
    path = str(tmp_path / "v2.sqlite3")
    db = Store(path)
    event = Event(PUBLIC, 100, "Me", "zirconium legacy", provenance=(PRIVATE,))
    db.save(event)
    before = db.db.execute("SELECT payload FROM events").fetchone()[0]
    downgrade_index_to_v2(db)
    db.close()
    db = Store(path)
    assert db.db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 3
    assert db.db.execute("SELECT payload FROM events").fetchone()[0] == before
    assert db.search("zirconium", 1, frozenset())[0] == event
    db.close()


def test_failed_v2_migration_rolls_back_version_and_partial_search_schema(tmp_path):
    path = str(tmp_path / "corrupt-v2.sqlite3")
    db = Store(path)
    db.save(Event(PUBLIC, 100, "Me", "legacy", id="broken"))
    downgrade_index_to_v2(db)
    db.db.execute("UPDATE events SET payload=? WHERE id='broken'", (b"not compressed",))
    db.db.commit()
    db.close()
    with pytest.raises(ValueError, match="Search index initialization failed"):
        Store(path)
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT version FROM schema_version").fetchall() == [(2,)]
    assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name='event_search'").fetchone()
    connection.close()
