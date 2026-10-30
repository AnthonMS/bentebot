import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source, Transcript
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat
from bentebot.tools import WEB_LOOKUP_NOTICE


def call(name, ident):
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": ident,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(
                                            {"query": "release"}
                                            if name == "web_search"
                                            else {"url": "https://example.com/release"}
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\ndata: [DONE]\n\n"
    )


@pytest.mark.parametrize("cancel", [False, True])
async def test_search_ack_precedes_tool_runs_in_parallel_and_is_cancelled_with_turn(store, wav_bytes, cancel):
    source = Source("channel", 10, 1)
    store.save(Event(source, 100, "Me", "Look up the release", kind="voice"))
    ack_started, ack_release, tool_started, tool_cancelled, final_played = (asyncio.Event() for _ in range(5))
    streams = [
        call("web_search", "search"),
        call("read_webpage", "read"),
        'data: {"choices":[{"delta":{"content":"Verified answer."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
    ]

    async def execute(name, arguments):
        assert ack_started.is_set()
        tool_started.set()
        if cancel:
            try:
                await asyncio.Event().wait()
            finally:
                tool_cancelled.set()
        return json.dumps(
            {
                "untrusted_evidence": {
                    "kind": "page_text",
                    "url": "https://example.com/release",
                    "text": "Verified",
                }
            }
        )

    async def play(pcm, text, progress):
        if text == "Let me check that.":
            ack_started.set()
            await ack_release.wait()
        else:
            assert ack_release.is_set()
            final_played.set()
        progress(text)

    async def feedback(action, request):
        assert action == "web_search"
        assert CHAT_OBSERVATION.get() is None  # A short acknowledgement cannot overwrite turn diagnostics.
        return "Let me check that."

    settings = Settings(
        web_tools_enabled=True, web_search_url="http://search", web_tool_models="test", chat_model="test"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=streams.pop(0)))
    ) as client:
        chat = OpenAIChat(settings, client)
        chat.web_tools.execute = AsyncMock(side_effect=execute)
        feedback_mock = AsyncMock(side_effect=feedback)
        session = VoiceConversation(
            settings,
            store,
            Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
            chat,
            SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
            source,
            999,
            AsyncMock(return_value=Audience(frozenset({100}), 1)),
            play,
            lambda: None,
            AsyncMock(),
            control_feedback=feedback_mock,
        )
        session.latest_request = "Look up the release"
        session.task = asyncio.create_task(session._answer())
        try:
            async with asyncio.timeout(3):
                await tool_started.wait()
            assert not final_played.is_set()  # The lookup can proceed while the notice is playing.
            if cancel:
                session.speech_activity(100)
                await session.transcript(100, "Me", Transcript("Actually tell me something else", False))
                assert not tool_cancelled.is_set()
                assert session.research.status()["research_running"] == 1
                assert session.metrics.last_outcome == "interrupted"
                assert session.metrics.last_chat["web_search_ack"] == "cancelled"
                assert not final_played.is_set()
            else:
                ack_release.set()
                async with asyncio.timeout(3):
                    await session.task
                assert session.metrics.last_chat["web_search_ack"] == "spoken"
                assert final_played.is_set()
                saved = next(e for e in store.recent(1, frozenset({100})) if e.kind == "assistant")
                assert saved.generated == saved.spoken == "Verified answer."
                feedback_mock.assert_awaited_once()
        finally:
            if not session.task.done():
                session.task.cancel()
            await asyncio.gather(session.task, return_exceptions=True)
            await session.close()
            if cancel:
                assert tool_cancelled.is_set()
            await chat.close()


@pytest.mark.parametrize("unavailable", [True, False])
async def test_ack_failure_or_unavailable_does_not_block_answer(store, wav_bytes, unavailable):
    source = Source("channel", 10, 1)
    store.save(Event(source, 100, "Me", "Look up the release", kind="voice"))
    played = []

    async def stream(messages, model):
        await WEB_LOOKUP_NOTICE.get()("")
        yield "Verified answer."

    async def synthesize(text, voice):
        if text == "Let me check that.":
            raise RuntimeError("Acknowledgement TTS unavailable")
        return wav_bytes

    async def play(pcm, text, progress):
        played.append(text)
        progress(text)

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
        AsyncMock(),
        control_feedback=AsyncMock(return_value=None if unavailable else "Let me check that."),
    )
    session.latest_request = "Look up the release"
    async with asyncio.timeout(3):
        await session._answer()
    assert session.metrics.last_outcome == "completed"
    assert session.metrics.last_chat["web_search_ack"] == ("unavailable" if unavailable else "failed")
    assert played == ["Verified answer."]
