from types import SimpleNamespace

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.voice_intent import voice_intent


@pytest.mark.parametrize(
    "utterance",
    [
        "I'll share it in the text channel.",
        "I'm going to show you my code.",
        "I am about to share my screen to test if you can see it, are you ready for that?",
        "I am about to share code in the text channel, are you ready?",
        "I can send you a link.",
        "Let me show you the Python code.",
        "Jeg vil dele noget kode i tekstkanalen.",
        "Don't write any code; just explain the error.",
    ],
)
def test_speaker_sharing_or_declining_code_does_not_request_written_output(utterance):
    intent = voice_intent(utterance)
    assert not intent.code_requested and not intent.text_requested


@pytest.mark.parametrize(
    "utterance",
    [
        "I'll share my code, can you write a Python function?",
        "I'll share my code and can you write a Python function?",
        "Jeg vil vise koden, kan du skrive en Python funktion?",
    ],
)
def test_speaker_offer_preserves_a_separate_request_to_the_bot(utterance):
    assert voice_intent(utterance).code_requested


@pytest.mark.parametrize(
    "utterance",
    [
        "Write a Python function to add two numbers",
        "Can you give me a Docker Compose file?",
        "Skriv en Python funktion",
        "Send me the SQL query",
        "Show me a bash script",
    ],
)
def test_code_generation_intent(utterance):
    assert voice_intent(utterance).code_requested


def test_explanation_is_opt_in():
    assert not voice_intent("Write a Python function").explanation_requested
    assert voice_intent("Write a function and explain it").explanation_requested
    assert voice_intent("Forklar hvordan koden virker").explanation_requested


async def test_code_prompt_excludes_dm_history_unavailable_to_text_readers(store, wav_bytes):
    from unittest.mock import AsyncMock

    source = Source("channel", 10, 1)
    voice_audience = Audience(frozenset({100}), 1)
    text_audience = Audience(frozenset({100, 200}), 1, 20)
    memory = Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000)
    store.save(Event(Source("dm", 30, owner_id=100), 100, "Me", "PRIVATE_DM_FACT"))
    store.save(Event(source, 100, "Me", "Write a Python function"))
    checked = []

    async def stream(messages, model):
        assert "PRIVATE_DM_FACT" not in str(messages)
        yield "```python\nprint('hello')\n```"

    async def output(text, sources, recipients):
        assert await memory.policy.allows_all(sources, text_audience)
        checked.append(text)
        return True

    session = VoiceConversation(
        Settings(),
        store,
        memory,
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        source,
        999,
        AsyncMock(return_value=voice_audience),
        AsyncMock(),
        lambda: None,
        AsyncMock(),
        text_output=output,
        text_channel_id=20,
        text_audience=AsyncMock(return_value=text_audience),
    )
    session.latest_request = "Write a Python function"
    await session._answer()
    assert len(checked) == 1
    assert session.metrics.provider_errors == 0


async def test_code_is_posted_before_only_acknowledgment_is_synthesized(store, wav_bytes):
    source = Source("channel", 10, 1)
    generated = "Here is a long unsolicited explanation.\n```python\nprint('hello')\n```\nNow let me explain it in detail."
    order = []

    async def permitted(*args):
        return True

    async def audience():
        return Audience(frozenset({100}), 1)

    async def stream(messages, model):
        assert "Discord text answer" in str(messages)
        yield generated

    async def output(text, sources, recipients):
        order.append(("text", text))
        return True

    async def synth(text, voice):
        order.append(("speech", text))
        return wav_bytes

    async def play(pcm, text, progress):
        progress(text)

    async def notify(text):
        raise AssertionError(text)

    memory = Memory(store, AccessPolicy(permitted), 10000)
    store.save(Event(source, 100, "Me", "Write a Python function"))
    session = VoiceConversation(
        Settings(),
        store,
        memory,
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synth),
        source,
        999,
        audience,
        play,
        lambda: None,
        notify,
        text_output=output,
        text_channel_id=20,
    )
    session.latest_request = "Write a Python function"
    await session._answer()
    assert order[0] == ("text", generated)
    assert [text for kind, text in order if kind == "speech"] == [
        "I've sent the code in the linked text channel."
    ]
    event = store.recent(1, frozenset())[-1]
    assert event.generated == generated and event.written == generated
    assert event.written_channel_id == 20 and "print" not in event.spoken
    rendered = memory.render(event)
    assert "Sent as text" in rendered and "Unspoken draft" not in rendered


async def test_failed_text_delivery_never_claims_code_was_sent(store, wav_bytes):
    source, spoken = Source("channel", 10, 1), []

    async def permitted(*args):
        return True

    async def audience():
        return Audience(frozenset({100}), 1)

    async def stream(messages, model):
        yield "```python\npass\n```"

    async def output(*args):
        return False

    async def synth(text, voice):
        spoken.append(text)
        return wav_bytes

    async def play(pcm, text, progress):
        progress(text)

    async def notify(text):
        raise AssertionError(text)

    store.save(Event(source, 100, "Me", "Write Python code"))
    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(permitted), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synth),
        source,
        999,
        audience,
        play,
        lambda: None,
        notify,
        text_output=output,
    )
    session.latest_request = "Write Python code"
    await session._answer()
    assert "couldn't post" in " ".join(spoken)
    assert "I've sent" not in " ".join(spoken)
