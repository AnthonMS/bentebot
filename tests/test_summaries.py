from types import SimpleNamespace

import pytest

from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import ProviderError
from bentebot.summaries import SummaryService

PUBLIC = Source("channel", 10, 1)
PRIVATE = Source("channel", 20, 1)
AUDIENCE = Audience(frozenset({100}), 1)


async def allowed(user, source):
    return True


async def test_summary_uses_only_requested_channel_and_inherits_private_provenance(store):
    store.save(Event(PUBLIC, 100, "Me", "We chose project cedar."))
    store.save(Event(PRIVATE, 100, "Me", "Do not include other channel history."))
    store.save(
        Event(
            PUBLIC,
            999,
            "Bot",
            "",
            kind="assistant",
            generated="unspoken private intention",
            delivery="voice",
            provenance=(PRIVATE,),
        )
    )
    store.save(Event(PUBLIC, 999, "Bot", "Do not summarize old summaries recursively.", kind="summary"))
    calls = []

    async def stream(messages, model):
        calls.append((messages, model))
        yield "Project cedar was chosen; another intention remained unspoken."

    memory = Memory(store, AccessPolicy(allowed), 3000)
    service = SummaryService(store, memory, SimpleNamespace(stream=stream))
    before = store.count()
    event = await service.summarize(PUBLIC, AUDIENCE, "test-model", 999)
    assert store.count() == before + 1
    assert event.kind == "summary" and event.content.startswith("Lossy AI summary")
    assert set(event.provenance) == {PUBLIC, PRIVATE}
    assert calls[0][1] == "test-model"
    prompt = str(calls[0][0])
    assert "project cedar" in prompt and "unspoken private intention" in prompt
    assert "Do not include other channel" not in prompt
    assert "Do not summarize old summaries" not in prompt
    assert sum(len(message["content"]) for message in calls[0][0]) <= memory.max_chars
    # The resulting derived summary is still removed when its private source is forgotten.
    store.forget_channel(PRIVATE.channel_id)
    assert all(item.id != event.id for item in store.recent(1, frozenset()))


async def test_denied_history_and_source_do_not_trigger_model_calls(store):
    store.save(Event(PUBLIC, 100, "Me", "public text", provenance=(PRIVATE,)))
    calls = []

    async def permitted(user, source):
        return source == PUBLIC

    async def stream(messages, model):
        calls.append(messages)
        yield "should never happen"

    service = SummaryService(
        store, Memory(store, AccessPolicy(permitted), 3000), SimpleNamespace(stream=stream)
    )
    with pytest.raises(ProviderError):
        await service.summarize(PUBLIC, AUDIENCE, "test", 999)
    with pytest.raises(ProviderError):
        await service.summarize(PRIVATE, AUDIENCE, "test", 999)
    assert calls == [] and store.count() == 1


async def test_source_revocation_during_generation_discards_summary(store):
    permitted = True

    async def check(user, source):
        return permitted

    async def stream(messages, model):
        nonlocal permitted
        yield "private information"
        permitted = False

    store.save(Event(PUBLIC, 100, "Me", "question"))
    service = SummaryService(store, Memory(store, AccessPolicy(check), 3000), SimpleNamespace(stream=stream))
    with pytest.raises(ProviderError, match="discarded"):
        await service.summarize(PUBLIC, AUDIENCE, "test", 999)
    assert store.count() == 1


async def test_output_limit_closes_stream_and_prompt_is_bounded(store):
    closed = False
    observed = []

    async def stream(messages, model):
        nonlocal closed
        observed.extend(messages)
        try:
            for _ in range(100):
                yield "x" * 1000
        finally:
            closed = True

    store.save(Event(PUBLIC, 100, "Me", "history " * 1000))
    service = SummaryService(
        store, Memory(store, AccessPolicy(allowed), 1000), SimpleNamespace(stream=stream)
    )
    result = await service.summarize(PUBLIC, AUDIENCE, "test", 999)
    assert len(result.content) <= 3000 and closed
    assert sum(len(message["content"]) for message in observed) <= 1000
