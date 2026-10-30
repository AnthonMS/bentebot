from types import SimpleNamespace
from unittest.mock import AsyncMock

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy


async def test_recent_project_context_reaches_voice_model_alongside_screen_and_latest_remark(
    store, wav_bytes
):
    channel = Source("channel", 10, 1)
    frame = Source("screen", 10, 1, owner_id=100, session_id="current", audience_ids=(100,))
    goal = "We are getting screen sharing working in Bentebot so we can discuss my projects."
    latest = "Well, the build obviously worked. Hehe."
    store.save(Event(channel, 100, "Me", goal, kind="voice"))
    store.save(
        Event(channel, 999, "Bentebot", "Let's test the receiver.", kind="assistant", delivery="voice")
    )
    store.save(Event(channel, 100, "Me", latest, kind="voice"))
    audience = Audience(frozenset({100}), 1)
    seen = []

    async def extra(recipients):
        assert recipients == audience
        return (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "fixture"}}]}],
            (frame,),
            "vision",
        )

    async def stream(messages, model):
        assert model == "vision"
        assert goal in str(messages)
        assert "Let's test the receiver." in str(messages)
        assert any(isinstance(message["content"], list) for message in messages)
        assert messages[-1] == {"role": "user", "content": latest}
        seen.append(messages)
        yield "Fair point."

    async def play(pcm, text, progress):
        progress(text)

    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        channel,
        999,
        AsyncMock(return_value=audience),
        play,
        lambda: None,
        AsyncMock(),
        extra_context=extra,
        extra_guard=AsyncMock(return_value=True),
    )
    session.latest_request = latest
    await session._answer()
    assert len(seen) == 1 and session.metrics.provider_errors == 0
