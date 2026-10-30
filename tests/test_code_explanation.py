from types import SimpleNamespace

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.speech import SentenceBuffer
from bentebot.voice_intent import spoken_explanation


@pytest.mark.parametrize(
    "opening,body,closing",
    [
        ("````markdown", "```python\nSECRET_EXECUTABLE_PAYLOAD()\n```\n", "````"),
        ("~~~~~markdown", "~~~~python\nSECRET_EXECUTABLE_PAYLOAD()\n~~~~\n", "~~~~~~"),
        ("```python", 'print("~~~")\nSECRET_EXECUTABLE_PAYLOAD()\n', "```"),
    ],
)
def test_complete_explanation_parser_never_narrates_nested_source(opening, body, closing):
    text = f"First explanation.\n{opening}\n{body}{closing}\nFinal explanation."
    assert spoken_explanation(text) == "First explanation.\nFinal explanation."


@pytest.mark.parametrize("marker", ["````", "~~~~"])
def test_streaming_split_closing_fence_does_not_release_code_to_speech(marker):
    splitter = SentenceBuffer()
    spoken = []
    for chunk in (
        "First explanation.\n" + marker + "markdown\n",
        marker[0] * 3 + "python\nSECRET_EXECUTABLE_PAYLOAD()\n" + marker[0] * 3 + "\n",
        marker[:2],
        marker[2:] + "\nFinal explanation. ",
    ):
        spoken.extend(splitter.feed(chunk))
    spoken.extend(splitter.feed("", final=True))
    assert spoken == ["First explanation.", "Final explanation."]


@pytest.mark.parametrize(
    "utterance", ["Write a Python function and explain it.", "Explain how this function works."]
)
async def test_explicit_explanation_never_synthesizes_nested_code(store, wav_bytes, utterance):
    source = Source("channel", 10, 1)
    generated = "First explanation.\n````markdown\n```python\nSECRET_EXECUTABLE_PAYLOAD()\n```\n````\nFinal explanation."
    spoken = []
    posted = []

    async def permitted(*args):
        return True

    async def audience():
        return Audience(frozenset({100}), 1)

    async def stream(messages, model):
        # Split inside the outer close as a real model token stream can do.
        split = generated.rfind("````") + 2
        yield generated[:split]
        yield generated[split:]

    async def output(text, sources, recipients):
        posted.append(text)
        return True

    async def synthesize(text, voice):
        spoken.append(text)
        return wav_bytes

    async def play(pcm, text, progress):
        progress(text)

    async def notify(text):
        raise AssertionError(text)

    store.save(Event(source, 100, "Me", utterance))
    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(permitted), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synthesize),
        source,
        999,
        audience,
        play,
        lambda: None,
        notify,
        text_output=output,
    )
    session.latest_request = utterance
    try:
        await session._answer()
        assert posted == [generated]
        assert "SECRET_EXECUTABLE_PAYLOAD" not in " ".join(spoken)
        assert spoken == ["First explanation.", "Final explanation."]
    finally:
        await session.close()


async def test_persisted_screen_provenance_is_hashable_and_enforces_capture_audience(store):
    public = Source("channel", 10, 1)
    screen = Source("screen", 10, 1, owner_id=100, session_id="capture", audience_ids=(100,))
    store.save(
        Event(public, 999, "Bentebot", "private screen observation", kind="assistant", provenance=(screen,))
    )
    decoded = store.recent(1, frozenset())[0]
    assert decoded.provenance[0] == screen
    assert isinstance(decoded.provenance[0].audience_ids, tuple)
    assert len(set(decoded.provenance)) == 1

    async def permitted(*args):
        return True

    memory = Memory(store, AccessPolicy(permitted), 10000)
    allowed, origins = await memory.context(Audience(frozenset({100}), 1))
    denied, _ = await memory.context(Audience(frozenset({200}), 1))
    assert "private screen observation" in str(allowed) and screen in origins
    assert "private screen observation" not in str(denied)
