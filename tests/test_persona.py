import json
from unittest.mock import AsyncMock

import pytest

from bentebot.domain import Audience
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.persona import PersonaStore, defaults


def test_persona_scope_ids_undo_reset_and_persistence(store):
    p = PersonaStore(store)
    changed, _ = p.change(1, None, "add", text="Prefer dry jokes.")
    ident = changed["rules"][-1]["id"]
    assert PersonaStore(store).effective(1, 10)[0]["rules"][-1]["text"] == "Prefer dry jokes."
    p.change(1, 10, "edit", rule_id=ident, text="Prefer occasional Danish jokes.")
    assert p.effective(1, 20)[0]["rules"][-1]["text"] == "Prefer dry jokes."
    p.change(1, 10, "remove", rule_id=ident)
    p.change(1, 10, "undo")
    assert p.effective(1, 10)[0]["rules"][-1]["text"] == "Prefer occasional Danish jokes."
    p.change(1, 10, "reset")
    assert p.effective(1, 10)[0]["rules"] == p.effective(1)[0]["rules"]
    p.change(1, 10, "undo")
    assert p.effective(1, 10)[0]["rules"][-1]["id"] == ident
    with pytest.raises(ValueError, match="another server"):
        p.effective(2, 10)
    assert p.effective(2, 20)[1] == "built-in defaults"


def test_persona_stale_edit_and_bounds_leave_settings_untouched(store):
    p = PersonaStore(store)
    p.change(1, None, "add", text="A reasonable trait.")
    before = store.get("guild:1", p.key)
    with pytest.raises(ValueError, match="changed"):
        p.change(1, None, "edit", rule_id=1, text="stale", revision=0)
    with pytest.raises(ValueError):
        p.change(1, None, "add", text="x" * 701)
    assert store.get("guild:1", p.key) == before


async def test_memory_resolves_live_channel_persona_without_copying_between_channels(store):
    memory = Memory(store, AccessPolicy(AsyncMock(return_value=True)), 8000)
    memory.personas.change(1, None, "add", text="Server custom rule.")
    memory.personas.change(1, 10, "add", text="Channel-only custom rule.")
    messages, _ = await memory.context(Audience(frozenset({100}), 1), persona_channel_id=10)
    assert "Channel-only custom rule." in messages[0]["content"]
    elsewhere, _ = await memory.context(Audience(frozenset({100}), 1), persona_channel_id=20)
    assert "Channel-only custom rule." not in elsewhere[0]["content"]
    assert "Server custom rule." in elsewhere[0]["content"]


def test_channel_undo_never_replays_parent_history(store):
    personas = PersonaStore(store)
    personas.change(1, None, "add", text="Server trait.")
    inherited = personas.effective(1, 10)[0]["rules"]
    with pytest.raises(ValueError, match="no persona change"):
        personas.change(1, 10, "undo")
    assert not store.get("channel:10", personas.key)
    personas.change(1, 10, "add", text="Channel trait.")
    restored, _ = personas.change(1, 10, "undo")
    assert restored["rules"] == inherited
    with pytest.raises(ValueError, match="no persona change"):
        personas.change(1, 10, "undo")
    assert personas.effective(1, 20)[0]["rules"] == inherited


def test_live_inheritance_server_edits_replace_same_rule_but_keep_other_local_changes(store):
    p = PersonaStore(store)
    server, _ = p.change(1, None, "add", text="Server rule A.")
    ident = server["rules"][-1]["id"]
    p.change(1, 10, "edit", rule_id=ident, text="Channel variation A.")
    local, _ = p.change(1, 10, "add", text="Channel-only rule.")
    local_id = local["rules"][-1]["id"]
    p.change(1, 20, "remove", rule_id=ident)
    p.change(1, None, "add", text="New global rule.")
    assert p.effective(1, 10)[0]["rules"][-2]["text"] == "New global rule."
    p.change(1, None, "edit", rule_id=ident, text="Server rule A revised.")
    for channel in (10, 20, 30):
        rules = {r["id"]: r["text"] for r in p.effective(1, channel)[0]["rules"]}
        assert rules[ident] == "Server rule A revised."
    assert p.effective(1, 10)[0]["rules"][-1] == {"id": local_id, "text": "Channel-only rule."}
    assert local_id not in {r["id"] for r in p.effective(1)[0]["rules"]}
    # Undoing a channel edit cannot resurrect an outdated parent override.
    p.change(1, 10, "undo")
    p.change(1, 10, "undo")
    assert (
        next(r["text"] for r in p.effective(1, 10)[0]["rules"] if r["id"] == ident)
        == "Server rule A revised."
    )


def test_server_reset_clears_all_channels_and_undo_restores_untouched_overlays(store):
    p = PersonaStore(store)
    p.change(1, None, "add", text="Server addition.")
    p.change(1, 10, "edit", rule_id=1, text="Channel edit.")
    p.change(1, 20, "add", text="Channel addition.")
    before = [p.effective(1, c)[0]["rules"] for c in (10, 20)]
    reset, _ = p.change(1, None, "reset")
    assert reset["rules"] == defaults()
    for channel in (10, 20, 30):
        assert p.effective(1, channel)[0]["rules"] == reset["rules"]
    with pytest.raises(ValueError, match="no persona change"):
        p.change(1, 10, "undo")
    p.change(1, None, "undo")
    assert [p.effective(1, c)[0]["rules"] for c in (10, 20)] == before


def test_server_update_invalidates_channel_modal_and_channel_ids_never_collide(store):
    p = PersonaStore(store)
    first, _ = p.change(1, 10, "add", text="Channel rule.")
    token = first["edit_token"]
    added, _ = p.change(1, None, "add", text="Server rule.")
    assert first["rules"][-1]["id"] != added["rules"][-1]["id"]
    with pytest.raises(ValueError, match="changed"):
        p.change(1, 10, "edit", rule_id=1, text="Stale modal.", revision=token)
    p.change(1, 10, "reset")
    assert p.effective(1, 10)[0]["rules"] == added["rules"]


def test_legacy_persona_migration_keeps_local_rules_and_inherits_later_server_additions(store):
    base = defaults()
    store.set(
        "guild:1",
        "persona_v1",
        json.dumps(
            {
                "rules": base + [{"id": 8, "text": "Later server rule."}],
                "next_id": 9,
                "revision": 1,
                "undo": [base],
            }
        ),
    )
    local = [dict(r) for r in base]
    local[0]["text"] = "Local first rule."
    local.append({"id": 8, "text": "Local addition with colliding old ID."})
    store.set(
        "channel:10", "persona_v1", json.dumps({"rules": local, "next_id": 9, "revision": 2, "undo": [base]})
    )
    p = PersonaStore(store)
    rules = p.effective(1, 10)[0]["rules"]
    assert rules[0]["text"] == "Local first rule."
    assert {r["text"] for r in rules} >= {"Later server rule.", "Local addition with colliding old ID."}
    assert len({r["id"] for r in rules}) == len(rules)
    p.change(1, None, "edit", rule_id=1, text="Server correction.")
    assert p.effective(1, 10)[0]["rules"][0]["text"] == "Server correction."
    p.change(1, None, "reset")
    assert p.effective(1, 10)[0]["rules"] == base


def test_server_reset_also_covers_unmigrated_legacy_channels(store):
    rules = defaults() + [{"id": 8, "text": "Old custom channel rule."}]
    store.set(
        "channel:99",
        "persona_v1",
        json.dumps({"rules": rules, "next_id": 9, "revision": 1, "undo": [defaults()]}),
    )
    p = PersonaStore(store)
    p.change(1, None, "reset")
    assert p.effective(1, 99)[0]["rules"] == defaults()


def test_channel_allocation_and_invalid_edit_are_atomic_and_server_reset_stays_in_its_guild(store):
    p = PersonaStore(store)
    p.change(1, 10, "add", text="First server channel.")
    p.change(2, 20, "add", text="Other server channel.")
    before_parent = store.get("guild:1", p.key)
    before_channel = store.get("channel:10", p.key)
    with pytest.raises(ValueError):
        p.change(1, 10, "add", text="x" * 701)
    assert store.get("guild:1", p.key) == before_parent
    assert store.get("channel:10", p.key) == before_channel
    other = p.effective(2, 20)[0]["rules"]
    p.change(1, None, "reset")
    assert p.effective(1, 10)[0]["rules"] == defaults()
    assert p.effective(2, 20)[0]["rules"] == other


def test_parent_additions_always_reach_a_channel_with_twenty_local_additions(store):
    p = PersonaStore(store)
    for number in range(20):
        p.change(1, 10, "add", text=f"Local rule {number}.")
    p.change(1, None, "add", text="Server-wide addition.")
    rules = p.effective(1, 10)[0]["rules"]
    assert len(rules) == 28 and any(r["text"] == "Server-wide addition." for r in rules)
