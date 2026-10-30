import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.conversation import TextConversation, VoiceConversation
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import OpenAIChat, ProviderError
from bentebot.summaries import SummaryService
from bentebot.tools import TOOL_GUARD, WEB_TOOLS_ALLOWED

SOURCE = Source("channel", 10, 1)
AUDIENCE = Audience(frozenset({100}), 1)


def tool_response():
    call = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "web_search", "arguments": '{"query":"public news"}'},
    }
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"tool_calls": [call]}}]})
        + "\n\n"
        + 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        + "data: [DONE]\n\n"
    )


@pytest.mark.parametrize("mode", ["voice", "text"])
async def test_permission_revocation_after_tool_plan_prevents_execution_and_delivery(store, mode):
    permitted = True
    requests = []

    async def check(user, source):
        return permitted

    async def audience():
        return AUDIENCE

    def handler(request):
        nonlocal permitted
        requests.append(json.loads(request.content))
        # The real model request is already in flight when source permissions change.
        permitted = False
        return httpx.Response(200, text=tool_response())

    settings = Settings(
        web_tools_enabled=True,
        web_search_url="http://searxng:8080",
        web_tool_models="test",
        chat_model="test",
    )
    memory = Memory(store, AccessPolicy(check), 10000)
    question = Event(SOURCE, 100, "Speaker", "What's in the news today?")
    store.save(question)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings, client)
        chat.web_tools.execute = AsyncMock()
        send, play, synthesize = AsyncMock(), AsyncMock(), AsyncMock()
        original_guard = TOOL_GUARD.get()
        try:
            if mode == "text":
                conversation = TextConversation(store, memory, chat)
                with pytest.raises(ProviderError, match="conversation changed"):
                    await conversation.answer(SOURCE, 999, audience, send, "test", trigger=question)
            else:
                conversation = VoiceConversation(
                    settings,
                    store,
                    memory,
                    chat,
                    SimpleNamespace(synthesize=synthesize),
                    SOURCE,
                    999,
                    audience,
                    play,
                    lambda: None,
                    AsyncMock(),
                )
                conversation.latest_request = question.content
                await conversation._answer()
                assert conversation.metrics.provider_errors == 1
            assert len(requests) == 1
            assert requests[0]["tool_choice"] == "auto"
            chat.web_tools.execute.assert_not_awaited()
            send.assert_not_awaited()
            play.assert_not_awaited()
            synthesize.assert_not_awaited()
            assert TOOL_GUARD.get() is original_guard
            assert store.count() == 1
        finally:
            await chat.close()


@pytest.mark.parametrize("fail", [False, True])
async def test_internal_summary_never_advertises_tools_and_restores_context_after_failure(store, fail):
    requests = []

    async def check(user, source):
        return True

    def handler(request):
        requests.append(json.loads(request.content))
        assert WEB_TOOLS_ALLOWED.get() is False
        if fail:
            return httpx.Response(503, text="PRIVATE provider body")
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"We chose cedar."}}]}\n\ndata: [DONE]\n\n'
        )

    store.save(Event(SOURCE, 100, "Speaker", "We chose cedar."))
    settings = Settings(web_tools_enabled=True, web_search_url="http://searxng:8080", web_tool_models="test")
    token = WEB_TOOLS_ALLOWED.set(True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings, client)
        chat.web_tools.execute = AsyncMock()
        service = SummaryService(store, Memory(store, AccessPolicy(check), 10000), chat)
        try:
            if fail:
                with pytest.raises(ProviderError):
                    await service.summarize(SOURCE, AUDIENCE, "test", 999)
            else:
                summary = await service.summarize(SOURCE, AUDIENCE, "test", 999)
                assert "We chose cedar." in summary.content
            assert "tools" not in requests[0]
            chat.web_tools.execute.assert_not_awaited()
            assert WEB_TOOLS_ALLOWED.get() is True
        finally:
            await chat.close()
            WEB_TOOLS_ALLOWED.reset(token)
