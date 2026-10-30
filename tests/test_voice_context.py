import json

from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy

SOURCE = Source("channel", 112233445566778899, 1)


async def yes(*args):
    return True


async def test_voice_budget_keeps_durable_history_and_text_context(store):
    for index in range(35):
        store.save(Event(SOURCE, 100, "Caller", f"Topic {index}: " + "ordinary discussion " * 10))
    memory = Memory(store, AccessPolicy(yes), 10000, voice_max_chars=1500)
    audience = Audience(frozenset({100}), 1)
    voice, _ = await memory.context(audience, voice=True)
    text, _ = await memory.context(audience)
    assert sum(len(m["content"]) for m in voice[1:]) <= 1500
    assert sum(len(m["content"]) for m in text[1:]) > 1500
    assert "Topic 34" in str(voice) and "Topic 0:" not in str(voice)
    assert store.count() == 35


async def test_small_voice_budget_still_retrieves_older_permitted_topic(store):
    old = Event(SOURCE, 100, "Caller", "The telescope calibration password is not a secret: stargazer.")
    store.save(old)
    for index in range(35):
        store.save(Event(SOURCE, 100, "Caller", f"Recent topic {index}: " + "unrelated discussion " * 10))
    memory = Memory(store, AccessPolicy(yes), 10000, voice_max_chars=1500)
    messages, sources = await memory.context(
        Audience(frozenset({100}), 1), query="telescope calibration", voice=True
    )
    assert "stargazer" in str(messages) and "Recent topic 34" in str(messages)
    assert sum(len(m["content"]) for m in messages[1:]) <= 1500
    assert sources == (SOURCE,) and store.event(old.id) == old


async def test_small_voice_budget_rechecks_permissions_for_older_retrieval(store):
    private = Source("channel", 123, 1)
    store.save(Event(private, 100, "Caller", "telescope calibration: private details"))
    for index in range(12):
        store.save(Event(SOURCE, 100, "Caller", f"Public discussion {index}: " + "hello " * 30))

    async def permission(user, source):
        return source == SOURCE

    memory = Memory(store, AccessPolicy(permission), 10000, voice_max_chars=1500)
    messages, sources = await memory.context(
        Audience(frozenset({100, 200}), 1), query="telescope calibration", voice=True
    )
    assert "private details" not in str(messages) and private not in sources


async def test_voice_budget_does_not_shrink_pinned_paused_reply(store):
    event = Event(
        SOURCE,
        999,
        "Bentebot",
        "Beginning of the story.",
        kind="assistant",
        generated="Beginning of the story. " + "Then the pirate repaired the ship. " * 100,
        spoken="Beginning of the story.",
        interrupted=True,
        delivery="voice",
    )
    store.save(event)
    audience = Audience(frozenset({100}), 1)
    original = Memory(store, AccessPolicy(yes), 24000)
    reduced = Memory(store, AccessPolicy(yes), 24000, voice_max_chars=8000)
    assert await original.paused_reply(event.id, audience) == await reduced.paused_reply(event.id, audience)
    assert reduced.context_budget(voice=True) == 8000
    assert reduced.context_budget() == 24000


async def test_paused_overlap_selects_only_the_latest_heard_sentence(store):
    heard = "The ship left port. Mara pulled the red lever, opening"
    event = Event(
        SOURCE,
        999,
        "Bentebot",
        heard,
        kind="assistant",
        spoken=heard,
        generated=heard + " the airlock. A drone rolled out.",
        interrupted=True,
        delivery="voice",
    )
    store.save(event)
    memory = Memory(store, AccessPolicy(yes), 24000, voice_max_chars=8000)
    message, sources = await memory.paused_reply(event.id, Audience(frozenset({100}), 1))
    background = json.loads(message["content"])["paused_reply_background"]
    assert background["resume_overlap_excerpt"] == "Mara pulled the red lever, opening"
    assert background["heard_so_far_estimated"] == heard
    assert "airlock" in background["previous_generated_draft_not_all_heard"]
    assert sources == (SOURCE,)


async def test_voice_prompt_has_no_delivery_wrappers_ids_or_timestamps(store):
    store.save(Event(SOURCE, 998877665544332211, "Bentebent", "What is the capital of Sweden?"))
    store.save(
        Event(
            SOURCE,
            999,
            "Bentebot",
            "Stockholm.",
            kind="assistant",
            generated="Stockholm.",
            spoken="Stockholm.",
            delivery="voice",
        )
    )
    memory = Memory(store, AccessPolicy(yes), 10000)
    messages, sources = await memory.context(Audience(frozenset({100}), 1), voice=True)
    content = str(messages[1:])
    assert "Bentebent" in content and "capital of Sweden" in content
    assert messages[-1] == {"role": "assistant", "content": "Stockholm."}
    for forbidden in [
        "112233445566778899",
        "998877665544332211",
        "started_at",
        "ended_at",
        "Spoken aloud",
        "reply in",
    ]:
        assert forbidden not in content
    assert sources == (SOURCE,)


async def test_unheard_intention_is_preserved_as_private_data_not_an_assistant_example(store):
    store.save(
        Event(
            SOURCE,
            999,
            "Bentebot",
            "Hello.",
            kind="assistant",
            generated="Hello. Unheard idea.",
            spoken="Hello.",
            delivery="voice",
            interrupted=True,
        )
    )
    memory = Memory(store, AccessPolicy(yes), 10000)
    messages, _ = await memory.context(Audience(frozenset({100}), 1), voice=True)
    assert messages[-1]["role"] == "user"
    background = json.loads(messages[-1]["content"])["private_assistant_background"]
    assert background["heard"] == "Hello."
    assert background["unheard_intention"] == "Unheard idea."
    assert background["interrupted"]
    assert store.recent(1, frozenset())[0].generated == "Hello. Unheard idea."


async def test_old_bookkeeping_readouts_are_excluded_from_voice_prompt_without_deletion(store):
    store.save(
        Event(
            SOURCE,
            999,
            "Bentebot",
            "Spoken aloud (estimated): A bad old reply.",
            kind="assistant",
            generated="Spoken aloud (estimated): A bad old reply.",
            spoken="Spoken aloud (estimated): A bad old reply.",
            delivery="voice",
        )
    )
    store.save(Event(SOURCE, 100, "Me", "Where is Stockholm?"))
    memory = Memory(store, AccessPolicy(yes), 10000)
    messages, _ = await memory.context(Audience(frozenset({100}), 1), query="reply", voice=True)
    assert "bad old reply" not in str(messages)
    assert store.count() == 2


async def test_voice_context_still_filters_denied_sources_and_preserves_written_code(store):
    private = Source("channel", 222233334444555566, 1)

    async def permitted(user, source):
        return source == SOURCE

    store.save(Event(private, 100, "Me", "PRIVATE FACT"))
    code = "```python\nchannel_id = 123\n```"
    store.save(
        Event(
            SOURCE,
            999,
            "Bentebot",
            "Sent it.",
            kind="assistant",
            generated=code,
            spoken="Sent it.",
            written=code,
            delivery="voice",
            written_channel_id=123123123123123123,
        )
    )
    memory = Memory(store, AccessPolicy(permitted), 10000)
    messages, sources = await memory.context(Audience(frozenset({100}), 1), voice=True)
    assert "PRIVATE FACT" not in str(messages)
    assert code in messages[-1]["content"]
    assert messages[-1]["role"] == "assistant"
    assert "123123123123123123" not in str(messages)
    assert private not in sources
