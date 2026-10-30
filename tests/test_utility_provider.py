import json
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.native_web import NativeResult
from bentebot.providers import OpenAIChat
from bentebot.tools import WEB_LOOKUP_NOTICE, WEB_RESEARCH_CONTEXT
from bentebot.utility_tools import TOOL_REQUEST, ToolRequest


def sse(delta, finish):
    return (
        "data: "
        + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})
        + "\n\ndata: [DONE]\n\n"
    )


def call(name, args):
    return sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        },
        "tool_calls",
    )


@pytest.mark.parametrize("native", [False, True])
async def test_utility_calls_work_for_local_and_native_cloud_aliases_without_local_web(native):
    requests = []
    responses = [call("calculate", {"expression": "17*23"}), sse({"content": "391"}, "stop")]

    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=responses.pop(0))

    settings = Settings(
        utility_tools_enabled=True,
        web_tools_enabled=native,
        native_web_anthropic_models="test" if native else "",
    )
    context = TOOL_REQUEST.set(ToolRequest(100))
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            chat = OpenAIChat(settings, client)
            try:
                assert (
                    "".join(
                        [
                            p
                            async for p in chat.stream(
                                [{"role": "user", "content": "Calculate 17*23"}], "test"
                            )
                        ]
                    )
                    == "391"
                )
            finally:
                await chat.close()
        assert json.loads(requests[1]["messages"][-1]["content"])["untrusted_evidence"]["value"] == 391
        assert any(s["function"]["name"] == "request_web_search" for s in requests[0]["tools"]) is native
    finally:
        TOOL_REQUEST.reset(context)


async def test_native_web_delegation_keeps_native_backend_and_guard_hooks(monkeypatch):
    context = TOOL_REQUEST.set(ToolRequest(100))
    preparation, notice = AsyncMock(return_value=None), AsyncMock()
    prepare_token, notice_token = WEB_RESEARCH_CONTEXT.set(preparation), WEB_LOOKUP_NOTICE.set(notice)
    native = AsyncMock(
        return_value=NativeResult(text="Verified result.", searches=1, sources=["https://example.com"])
    )
    monkeypatch.setattr("bentebot.providers.native_round", native)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=call("request_web_search", {"query": "latest release"}))
        )
    ) as client:
        chat = OpenAIChat(
            Settings(utility_tools_enabled=True, web_tools_enabled=True, native_web_anthropic_models="test"),
            client,
        )
        try:
            answer = "".join(
                [p async for p in chat.stream([{"role": "user", "content": "Find latest release"}], "test")]
            )
            assert "Verified result." in answer
            native.assert_awaited_once()
            assert native.await_args.kwargs["search"] is True
            preparation.assert_awaited_once()
            notice.assert_awaited_once()
        finally:
            await chat.close()
            WEB_RESEARCH_CONTEXT.reset(prepare_token)
            WEB_LOOKUP_NOTICE.reset(notice_token)
            TOOL_REQUEST.reset(context)
