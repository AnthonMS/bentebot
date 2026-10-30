import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.domain import ResponseRecord
from bentebot.model_warmth import VoiceModelWarmth
from bentebot.native_web import NativeWebError, native_round
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat, ProviderError
from bentebot.tools import WEB_LOOKUP_NOTICE, WEB_RESEARCH_CONTEXT, WEB_TOOLS_ALLOWED

URL = "https://example.com/release"


def sse(*events):
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events)


def fixture(provider, *, plan=False, ordinary=False, url=URL):
    if provider == "openai":
        output = (
            [
                {
                    "type": "function_call",
                    "name": "request_web_search",
                    "arguments": '{"query":"current release"}',
                }
            ]
            if plan
            else []
            if ordinary
            else [{"type": "web_search_call", "id": "search_1", "status": "completed"}]
        )
        if not plan:
            output.append(
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Verified answer.",
                            "annotations": [] if ordinary else [{"type": "url_citation", "url": url}],
                        }
                    ],
                }
            )
        events = [] if plan else [{"type": "response.output_text.delta", "delta": "Verified answer."}]
        return sse(
            *events,
            {
                "type": "response.completed",
                "response": {"status": "completed", "model": "test", "output": output},
            },
        )
    if plan:
        return sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "plan",
                    "name": "request_web_search",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":"current release"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        )
    events = (
        []
        if ordinary
        else [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "search_1",
                    "name": "web_search",
                    "input": {},
                },
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "web_search_tool_result", "tool_use_id": "search_1", "content": []},
            },
        ]
    )
    events += [
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Verified answer."},
        },
    ]
    if not ordinary:
        events.append(
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "citations_delta",
                    "citation": {"type": "web_search_result_location", "url": url},
                },
            }
        )
    return sse(
        *events, {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, {"type": "message_stop"}
    )


def settings(provider):
    return Settings(web_tools_enabled=True, **{f"native_web_{provider}_models": "test"})


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_native_search_replans_before_server_tools_and_uses_typed_citations(provider):
    requests = []
    streams = [fixture(provider, plan=True), fixture(provider)]

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, text=streams.pop(0), headers={"x-litellm-model-id": "native-deployment"})

    prepare = AsyncMock(return_value=[{"role": "user", "content": "SHAREABLE_REQUEST"}])
    notice = AsyncMock()
    prep_token = WEB_RESEARCH_CONTEXT.set(prepare)
    notice_token = WEB_LOOKUP_NOTICE.set(notice)
    record = ResponseRecord()
    observation = CHAT_OBSERVATION.set(record)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings(provider), client)
        try:
            output = "".join(
                [p async for p in chat.stream([{"role": "user", "content": "PRIVATE_CONTEXT"}], "test")]
            )
            assert output == "Verified answer."
            assert "PRIVATE_CONTEXT" in json.dumps(requests[0])
            assert "PRIVATE_CONTEXT" not in json.dumps(requests[1])
            assert "SHAREABLE_REQUEST" in json.dumps(requests[1])
            assert requests[0][1]["tools"][0].get("name") == "request_web_search"
            assert requests[1][1]["tools"][0]["type"].startswith("web_search")
            assert all(
                "Application runtime metadata for THIS request" in json.dumps(body) for _, body in requests
            )
            assert record.chat_metadata["gateway_deployment_id"] == "native-deployment"
            assert record.web_sources == [URL]
            assert record.chat_metadata["web_source_kind"] == "provider_citations"
            assert record.chat_metadata["native_web_searches"] == 1
            assert record.chat_metadata["web_pages_read"] == 0
            prepare.assert_awaited_once()
            notice.assert_awaited_once()
            assert chat.web_tools is None  # No local SearXNG client is required.
        finally:
            await chat.close()
            CHAT_OBSERVATION.reset(observation)
            WEB_LOOKUP_NOTICE.reset(notice_token)
            WEB_RESEARCH_CONTEXT.reset(prep_token)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_native_ordinary_chat_retains_context_without_search_or_ack(provider):
    prepare, notice = AsyncMock(), AsyncMock()
    pt, nt = WEB_RESEARCH_CONTEXT.set(prepare), WEB_LOOKUP_NOTICE.set(notice)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=fixture(provider, ordinary=True))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings(provider), client)
        try:
            assert (
                "".join(
                    [p async for p in chat.stream([{"role": "user", "content": "PRIVATE_MEMORY"}], "test")]
                )
                == "Verified answer."
            )
            assert "PRIVATE_MEMORY" in json.dumps(requests[0]) and len(requests) == 1
            prepare.assert_not_awaited()
            notice.assert_not_awaited()
        finally:
            await chat.close()
            WEB_RESEARCH_CONTEXT.reset(pt)
            WEB_LOOKUP_NOTICE.reset(nt)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_native_error_and_incomplete_stream_never_deliver_partial_answer(provider):
    text = fixture(provider)
    if provider == "openai":
        text = text[: text.index('data: {"type": "response.completed"')]
    else:
        text = text[: text.index('data: {"type": "message_stop"')]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=text))
    ) as client:
        with pytest.raises(NativeWebError, match="before completion"):
            await native_round(
                client,
                "http://gateway/v1",
                "",
                [],
                "test",
                provider,
                search=True,
                max_tokens=100,
                max_searches=2,
                max_chars=1000,
            )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_native_citations_exclude_private_urls(provider):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=fixture(provider, url="http://127.0.0.1/private"))
        )
    ) as client:
        result = await native_round(
            client,
            "http://gateway/v1",
            "",
            [],
            "test",
            provider,
            search=True,
            max_tokens=100,
            max_searches=2,
            max_chars=1000,
        )
        assert result.sources == []


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_denied_native_research_never_sends_server_tools(provider):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=fixture(provider, plan=True))

    token = WEB_RESEARCH_CONTEXT.set(AsyncMock(side_effect=ProviderError("Denied")))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings(provider), client)
        try:
            with pytest.raises(ProviderError):
                _ = [p async for p in chat.stream([], "test")]
            assert len(calls) == 1
        finally:
            await chat.close()
            WEB_RESEARCH_CONTEXT.reset(token)


def test_native_and_local_aliases_cannot_overlap():
    with pytest.raises(ValueError, match="exactly one"):
        Settings(
            web_tools_enabled=True,
            web_search_url="http://search",
            web_tool_models="test",
            native_web_openai_models="test",
        ).validate(login=False)


async def test_feedback_context_disables_native_tools_without_inventing_cloud_suffixes():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"Checking."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        )

    token = WEB_TOOLS_ALLOWED.set(False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings("openai"), client)
        try:
            assert "".join([p async for p in chat.stream([], "test")]) == "Checking."
            assert calls[0].url.path.endswith("/chat/completions")
            assert "tools" not in json.loads(calls[0].content)
        finally:
            await chat.close()
            WEB_TOOLS_ALLOWED.reset(token)


@pytest.mark.parametrize("alias", ["claude-sonnet-5-5", "gpt-6.1-sol"])
async def test_enabling_native_cloud_search_never_enables_cloud_warming(alias):
    requests = []

    def handler(request):
        requests.append(request)
        raise AssertionError("No native/cloud request is permitted from a warmer")

    config = Settings(
        web_tools_enabled=True,
        native_web_openai_models="gpt-6.1-sol",
        native_web_anthropic_models="claude-sonnet-5-5",
        voice_warm_models="qwen3.8-uncensored:fast",
    )
    config.validate(login=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(config, client)
        manager = VoiceModelWarmth(config, chat)
        try:
            await manager.prepare(1, alias)
            assert not manager.models
            with pytest.raises(ProviderError, match="explicitly configured local"):
                await chat.warm(alias)
            assert requests == []
        finally:
            await manager.close()
            await chat.close()


@pytest.mark.parametrize(
    "provider,stream,reason",
    [
        ("openai", sse({"type": "response.failed"}), "reported an error"),
        (
            "anthropic",
            sse({"type": "message_delta", "delta": {"stop_reason": "pause_turn"}}, {"type": "message_stop"}),
            "paused",
        ),
        (
            "anthropic",
            sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "server_tool_use", "name": "unknown"},
                }
            ),
            "unsupported server",
        ),
        ("openai", fixture("openai", ordinary=True), "skipped"),
        ("openai", fixture("openai"), "budget"),
    ],
)
async def test_native_failure_and_budget_states_fail_closed(provider, stream, reason):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=stream))
    ) as client:
        with pytest.raises(NativeWebError, match=reason):
            await native_round(
                client,
                "http://gateway/v1",
                "",
                [],
                "test",
                provider,
                search=True,
                max_tokens=100,
                max_searches=0 if reason == "budget" else 2,
                max_chars=1000,
            )


@pytest.mark.parametrize("cancel", [False, True])
async def test_native_pings_do_not_defeat_timeout_and_cancellation_closes_stream(cancel):
    entered, closed = asyncio.Event(), asyncio.Event()

    class Pings(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            yield b'event: ping\ndata: {"type":"ping"}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Pings()))
    ) as client:
        task = asyncio.create_task(
            native_round(
                client,
                "http://gateway/v1",
                "",
                [],
                "test",
                "anthropic",
                search=True,
                max_tokens=100,
                max_searches=2,
                max_chars=1000,
                first_timeout=None if cancel else 0.02,
            )
        )
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert closed.is_set()


@pytest.mark.parametrize("voice", [True, False])
async def test_native_citation_markers_are_not_prose_and_text_replies_keep_links(voice):
    streams = [
        fixture("openai", plan=True),
        fixture("openai").replace(
            "Verified answer.", "Verified answer. \\ue200cite\\ue202turn0search0\\ue201"
        ),
    ]
    token = CHAT_OBSERVATION.set(ResponseRecord() if voice else None)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=streams.pop(0)))
    ) as client:
        chat = OpenAIChat(settings("openai"), client)
        try:
            output = "".join([part async for part in chat.stream([], "test")])
            assert "Verified answer." in output
            assert "\ue200" not in output and "turn0search0" not in output
            assert (URL in output) == (not voice)
        finally:
            CHAT_OBSERVATION.reset(token)
            await chat.close()
