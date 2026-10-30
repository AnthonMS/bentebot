from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.voice_intent import voice_intent


@pytest.mark.parametrize(
    "utterance,generated,written",
    [
        ("What is the capital of Sweden?", "The capital is Stockholm.", False),
        ("Explain how the code works", "It checks each possible divisor.", False),
        ("Hvad er Sveriges hovedstad?", "Hovedstaden er Stockholm.", False),
        ("Send me the link", "https://example.com/documentation", True),
        ("Post this as text", "Some useful written material.", True),
        ("Vis en tabel", "| A | B |\n|---|---|\n| 1 | 2 |", True),
        ("What should I run?", "```sh\nexample-command\n```", True),
    ],
)
async def test_only_written_material_is_posted(store, wav_bytes, utterance, generated, written):
    source = Source("channel", 10, 1)
    store.save(Event(source, 100, "Me", utterance))

    async def stream(messages, model):
        yield generated

    async def play(pcm, text, progress):
        progress(text)

    output = AsyncMock(return_value=True)
    synthesize = AsyncMock(return_value=wav_bytes)
    notice = AsyncMock()
    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synthesize),
        source,
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        play,
        lambda: None,
        notice,
        text_output=output,
        text_channel_id=20,
    )
    session.latest_request = utterance
    try:
        await session._answer()
        assert session.metrics.provider_errors == 0
        assert output.await_count == int(written)
        event = next(event for event in store.recent(1, frozenset()) if event.kind == "assistant")
        assert bool(event.written) is written
        if voice_intent(utterance).text_requested:
            assert synthesize.await_args_list[0].args[0] == "I've sent it in the linked text channel."
            assert generated not in " ".join(call.args[0] for call in synthesize.await_args_list)
        notice.assert_not_awaited()
    finally:
        await session.close()
