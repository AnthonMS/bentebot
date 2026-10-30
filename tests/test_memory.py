import sqlite3
import time

import pytest

from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.storage import Store

PUBLIC = Source("channel", 10, 1)
PRIVATE = Source("channel", 20, 1)
DM = Source("dm", 30, owner_id=100)


async def can_read(user, source):
    return source == PUBLIC or (source == PRIVATE and user == 100)


@pytest.mark.parametrize(
    "audience,source,expected",
    [
        (Audience(frozenset({100}), 1), PRIVATE, True),
        (Audience(frozenset({100, 200}), 1), PRIVATE, False),
        (Audience(frozenset({100, 200}), 1), PUBLIC, True),
        (Audience(frozenset({100}), 1), DM, True),
        (Audience(frozenset({100, 200}), 1), DM, False),
        (Audience(frozenset({100}), 1, 10), DM, False),
        (Audience(frozenset({100}), 2), PUBLIC, False),
        (Audience(frozenset(), 1), PUBLIC, False),
    ],
)
async def test_acl(audience, source, expected):
    assert await AccessPolicy(can_read).allows(source, audience) == expected


async def test_public_cross_channel_private_draft_and_dm_filtering(store):
    store.save(Event(PUBLIC, 200, "Other", "public topic", created_at=1))
    store.save(Event(PRIVATE, 100, "Me", "private topic", created_at=2))
    store.save(Event(DM, 100, "Me", "DM topic", created_at=3))
    store.save(
        Event(
            PUBLIC,
            999,
            "Bot",
            "",
            kind="assistant",
            provenance=(PRIVATE,),
            generated="private intention",
            delivery="voice",
            interrupted=True,
            created_at=4,
        )
    )
    memory = Memory(store, AccessPolicy(can_read), 10000)
    messages, sources = await memory.context(Audience(frozenset({100, 200}), 1))
    text = str(messages)
    assert "public topic" in text
    assert "private topic" not in text and "DM topic" not in text and "private intention" not in text
    assert sources == (PUBLIC,)
    messages, sources = await memory.context(Audience(frozenset({100}), 1))
    assert "private intention" in str(messages)
    assert "Unspoken draft, never delivered" in str(messages)
    assert "DM topic" in str(messages)
    assert PRIVATE in sources and DM in sources


async def test_permission_revocation_is_evaluated_each_time(store):
    allowed = True

    async def permission(user, source):
        return allowed

    memory = Memory(store, AccessPolicy(permission), 10000)
    store.save(Event(PRIVATE, 100, "Me", "secret"))
    audience = Audience(frozenset({100}), 1)
    assert "secret" in str((await memory.context(audience))[0])
    allowed = False
    assert "secret" not in str((await memory.context(audience))[0])


def test_compressed_persistence_and_unspoken_state(tmp_path):
    path = str(tmp_path / "db.sqlite3")
    event = Event(
        PUBLIC,
        999,
        "Bot",
        "Hello.",
        kind="assistant",
        generated="Hello. More to say.",
        spoken="Hello.",
        provenance=(PRIVATE,),
        interrupted=True,
        delivery="voice",
    )
    db = Store(path)
    db.save(event)
    db.close()
    db = Store(path)
    assert db.recent(1, frozenset())[0] == event
    assert db.db.execute("SELECT payload FROM events").fetchone()[0].startswith(b"x")
    db.close()


def test_forget_removes_derivatives_and_retention(store):
    store.save(Event(PRIVATE, 100, "Me", "old secret", created_at=1))
    store.save(Event(PUBLIC, 999, "Bot", "derived", provenance=(PRIVATE,)))
    store.save(Event(PUBLIC, 100, "Me", "keep"))
    assert store.forget_channel(PRIVATE.channel_id) == 2
    assert store.count() == 1
    store.save(Event(PUBLIC, 100, "Me", "expired", created_at=time.time() - 100 * 86400))
    assert store.prune(90) == 1


async def test_context_budget_and_chronology(store):
    for i in range(20):
        store.save(Event(PUBLIC, 100, "Me", f"message-{i}:" + "x" * 100, created_at=i))
    messages, _ = await Memory(store, AccessPolicy(can_read), 600).context(Audience(frozenset({100}), 1))
    assert sum(len(message["content"]) for message in messages[1:]) <= 600
    assert "message-19" in messages[-1]["content"]
    assert "message-0:" not in str(messages)


def test_invalid_sources_rejected():
    with pytest.raises(ValueError):
        Source("dm", 1)
    with pytest.raises(ValueError):
        Source("channel", 1)


def test_equal_timestamps_preserve_insertion_order_instead_of_random_uuid_order(store):
    store.save(Event(PUBLIC, 100, "Me", "first", id="z", created_at=1))
    store.save(Event(PUBLIC, 999, "Bot", "second", id="a", created_at=1))
    assert [event.content for event in store.recent(1, frozenset())] == ["first", "second"]


def test_source_index_migration_preserves_existing_history(tmp_path):
    path = str(tmp_path / "legacy.sqlite3")
    db = Store(path)
    db.save(Event(PUBLIC, 999, "Bot", "derived", provenance=(PRIVATE,)))
    # Simulate the initial schema, before provenance had its own lookup index.
    db.db.execute("DROP TABLE event_sources")
    db.db.execute("UPDATE schema_version SET version=1")
    db.db.commit()
    db.close()
    db = Store(path)
    assert db.count() == 1
    assert db.forget_channel(PRIVATE.channel_id) == 1
    assert db.db.execute("SELECT COUNT(*) FROM event_sources").fetchone()[0] == 0
    db.close()


def test_unknown_future_schema_is_rejected_without_changing_it(tmp_path):
    path = str(tmp_path / "future.sqlite3")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    connection.execute("INSERT INTO schema_version VALUES(99)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="Unsupported database schema"):
        Store(path)
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT version FROM schema_version").fetchall() == [(99,)]
    connection.close()
