import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.conversation import VerifiedSourceCard, VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import OpenAIChat
from bentebot.tools import WEB_RESEARCH_CONTEXT
from bentebot.voice_intent import source_link_request, voice_intent

VOICE = Source("channel", 10, 1)
DM = Source("dm", 30, owner_id=100)
SCREEN = Source("screen", 10, 1, owner_id=100, session_id="private", audience_ids=(100,))
CALLERS = Audience(frozenset({100}), 1)
READERS = Audience(frozenset({100, 200}), 1, 20)
URL = "https://example.com/release"


def answer():
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": "Verified release."}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    )


def call(name, argument, ident):
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
                                    "function": {"name": name, "arguments": json.dumps(argument)},
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


@pytest.mark.parametrize("mode", ["research", "ordinary", "destination_changed", "voice_private"])
async def test_only_research_replans_private_context_before_any_tool_executes(store, wav_bytes, mode):
    store.save(Event(DM, 100, "Me", "PRIVATE_DM_CANARY"))
    store.save(Event(SCREEN, 100, "Me", "PRIVATE_SCREEN_CANARY"))
    store.save(Event(VOICE, 100, "Me", "Check the latest release", kind="voice"))
    requests = []
    destination = READERS
    initial = (
        answer() if mode == "ordinary" else call("web_search", {"query": "PRIVATE_DM_CANARY"}, "private_plan")
    )
    streams = [
        initial,
        call("web_search", {"query": "official release"}, "public_plan"),
        call("read_webpage", {"url": URL}, "read"),
        answer(),
    ]

    def handler(request):
        nonlocal destination
        requests.append(json.loads(request.content))
        if len(requests) == 2 and mode == "destination_changed":
            destination = Audience(frozenset({100, 200, 300}), 1, 20)
        return httpx.Response(200, text=streams.pop(0))

    async def can_read(user, source):
        return not (mode == "voice_private" and user == 200 and source == VOICE)

    policy = AccessPolicy(can_read)

    async def text_audience():
        return destination

    async def play(pcm, text, progress):
        progress(text)

    written = []

    async def write(text, origins, audience):
        assert DM not in origins and SCREEN not in origins
        assert await policy.allows_all(origins, destination)
        written.append(text)
        return True

    settings = Settings(
        web_tools_enabled=True,
        web_search_url="http://searxng:8080",
        web_tool_models="test",
        chat_model="test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings, client)
        execute = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "untrusted_evidence": {
                            "kind": "search_results_not_read_articles",
                            "results": [{"url": URL}],
                        }
                    }
                ),
                json.dumps(
                    {"untrusted_evidence": {"kind": "page_text", "url": URL, "text": "Verified release"}}
                ),
            ]
        )
        chat.web_tools.execute = execute
        session = VoiceConversation(
            settings,
            store,
            Memory(store, policy, 10000),
            chat,
            SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
            VOICE,
            999,
            AsyncMock(return_value=CALLERS),
            play,
            lambda: None,
            AsyncMock(),
            text_output=write,
            text_audience=text_audience,
            text_channel_id=20,
        )
        session.latest_request = "Check the latest release"
        try:
            await session._answer()
            first = json.dumps(requests[0]["messages"])
            assert "PRIVATE_DM_CANARY" in first and "PRIVATE_SCREEN_CANARY" in first
            if mode == "research":
                assert execute.await_count == 2 and len(requests) == 4
                for request in requests[1:]:
                    assert "PRIVATE_DM_CANARY" not in json.dumps(request)
                    assert "PRIVATE_SCREEN_CANARY" not in json.dumps(request)
                assert all("PRIVATE" not in args.args[1] for args in execute.await_args_list)
                assert session.metrics.last_chat["web_context_scope"] == "rebuilt_for_voice_and_text"
                assert session.metrics.last_chat["web_source_delivery"] == "delivered"
                assert URL in written[0]
            else:
                execute.assert_not_awaited()
                assert written == []
                if mode == "ordinary":
                    assert session.metrics.provider_errors == 0
                    assert len(requests) == 1
                else:
                    assert session.metrics.provider_errors == 1
            assert WEB_RESEARCH_CONTEXT.get() is None
        finally:
            await chat.close()


@pytest.mark.parametrize("invalid", ["", "expired", "private_origin", "new_topic", "revoked_screen"])
async def test_link_followup_reuses_only_current_permitted_verified_source(store, wav_bytes, invalid):
    request = (
        "Send me the link for a different project" if invalid == "new_topic" else "Can you send me the link?"
    )
    store.save(Event(VOICE, 100, "Me", request, kind="voice"))
    calls = []

    async def stream(messages, model):
        calls.append(messages)
        yield "No reusable source was selected."

    async def play(pcm, text, progress):
        progress(text)

    write, warm, visual = AsyncMock(return_value=True), AsyncMock(), AsyncMock(return_value=([], (), None))
    session = VoiceConversation(
        Settings(),
        store,
        Memory(store, AccessPolicy(AsyncMock(return_value=True)), 10000),
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        VOICE,
        999,
        AsyncMock(return_value=CALLERS),
        play,
        lambda: None,
        AsyncMock(),
        text_output=write,
        text_audience=AsyncMock(return_value=READERS),
        text_channel_id=20,
        before_chat=warm,
        extra_context=visual,
        extra_guard=AsyncMock(return_value=invalid != "revoked_screen"),
    )
    session.last_web_card = VerifiedSourceCard(
        (URL,),
        (DM,) if invalid == "private_origin" else (VOICE,),
        (),
        time.monotonic() - (601 if invalid == "expired" else 0),
    )
    session.latest_request = request
    await session._answer()
    if not invalid:
        assert calls == []
        warm.assert_not_awaited()
        visual.assert_not_awaited()
        assert URL in write.await_args.args[0]
        assert session.metrics.last_chat["web_source_reused"] is True
        assert session.metrics.last_chat["tool_calls"] == 0
        assert session.metrics.last_chat["web_pages_read"] == 0
    else:
        assert calls or invalid == "revoked_screen"
        if write.await_count:
            assert URL not in write.await_args.args[0]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Can you send me the link?", True),
        ("You send me the link.", True),
        ("Send me the sources again", True),
        ("Kan du sende mig linket?", True),
        ("Send me a link about Python", False),
        ("Look up another link", False),
    ],
)
def test_link_followup_does_not_reuse_sources_for_a_new_topic(text, expected):
    assert source_link_request(text) is expected
    if expected:
        assert voice_intent(text).text_requested
