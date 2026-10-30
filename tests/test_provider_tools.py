import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.domain import ResponseRecord
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat, ProviderError
from bentebot.tools import TOOL_GUARD, WEB_TOOLS_ALLOWED


def event(delta=None, finish=None):
    return "data: " + json.dumps({"choices": [{"delta": delta or {}, "finish_reason": finish}]}) + "\n\n"


def call_event(name="web_search", arguments='{"query":"latest news"}', call_id="call_1", index=0):
    return event(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ]
        }
    )


def finished_calls(prefix):
    return prefix + event(finish="tool_calls") + "data: [DONE]\n\n"


def enabled_settings(**kwargs):
    return Settings(
        web_tools_enabled=True, web_search_url="http://searxng:8080", web_tool_models="test", **kwargs
    )


async def test_fragmented_automatic_search_and_page_read_before_grounded_answer():
    requests = []
    streams = [
        finished_calls(
            event({"content": "I will look this up."})
            + event(
                {
                    "tool_calls": [
                        {"index": 0, "id": "call_1", "function": {"name": "web_", "arguments": '{"query":'}}
                    ]
                }
            )
            + event(
                {"tool_calls": [{"index": 0, "function": {"name": "search", "arguments": '"latest news"}'}}]}
            )
        ),
        finished_calls(call_event("read_webpage", '{"url":"https://example.com/article"}', "call_2")),
        event({"content": "The event happened today. "})
        + event({"content": "[Source](https://example.com/article)"}, "stop")
        + "data: [DONE]\n\n",
    ]

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=streams.pop(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = AsyncMock(
            side_effect=[
                '{"untrusted_evidence":{"results":[{"url":"https://example.com/article"}]}}',
                '{"untrusted_evidence":{"text":"The event happened today."}}',
            ]
        )
        record = ResponseRecord()
        token = CHAT_OBSERVATION.set(record)
        try:
            output = "".join(
                [
                    piece
                    async for piece in provider.stream(
                        [{"role": "user", "content": "What happened in the news today?"}], "test"
                    )
                ]
            )
            assert output == "The event happened today. [Source](https://example.com/article)"
            assert "look this up" not in output
            assert provider.web_tools.execute.await_count == 2
            assert requests[0]["tool_choice"] == "auto"
            assert not requests[0]["parallel_tool_calls"]
            assert requests[1]["messages"][-1]["role"] == "tool"
            final_evidence = requests[2]["messages"][-1]
            assert final_evidence["role"] == "user"
            assert "The event happened today." in final_evidence["content"]
            assert "https://example.com/article" in final_evidence["content"]
            assert not any(
                message.get("tool_calls") or message["role"] == "tool" for message in requests[2]["messages"]
            )
            assert record.chat_metadata["tool_calls"] == 2
            assert record.web_sources == []  # Neither result is a successful typed page read.
        finally:
            CHAT_OBSERVATION.reset(token)
            await provider.close()


@pytest.mark.parametrize(
    "tool_name,result,expected",
    [
        (
            "read_webpage",
            '{"untrusted_evidence":{"kind":"page_text","url":"https://example.com/article","text":"Actual article"}}',
            ["https://example.com/article"],
        ),
        (
            "web_search",
            '{"untrusted_evidence":{"kind":"page_text","url":"https://example.com/search-hit","text":"Search excerpt"}}',
            [],
        ),
        (
            "read_webpage",
            '{"untrusted_evidence":{"url":"https://example.com/article","text":"Missing kind"}}',
            [],
        ),
        (
            "read_webpage",
            '{"untrusted_evidence":{"kind":"page_text","url":"http://127.0.0.1/","text":"Private address"}}',
            [],
        ),
        (
            "read_webpage",
            '{"error":"failed","untrusted_evidence":{"kind":"page_text","url":"https://example.com/article","text":"Unverified"}}',
            [],
        ),
        (
            "read_webpage",
            '{"untrusted_evidence":{"kind":"page_text","url":"https://example.com/article","text":""}}',
            [],
        ),
        ("read_webpage", '{"untrusted_evidence":null}', []),
        ("read_webpage", "not json", []),
    ],
)
async def test_only_successful_typed_page_reads_produce_written_sources(tool_name, result, expected):
    streams = [
        finished_calls(call_event(tool_name, '{"url":"https://example.com/article"}', "read_1")),
        event({"content": "A final answer with a fabricated https://invented.example.com/link"}, "stop")
        + "data: [DONE]\n\n",
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=streams.pop(0)))
    ) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = AsyncMock(return_value=result)
        record = ResponseRecord()
        token = CHAT_OBSERVATION.set(record)
        try:
            _ = [part async for part in provider.stream([], "test")]
            assert record.web_sources == expected
            assert record.chat_metadata["web_pages_read"] == len(expected)
        finally:
            CHAT_OBSERVATION.reset(token)
            await provider.close()


@pytest.mark.parametrize("initial_count", [0, 3, 4])
async def test_successful_page_sources_are_unique_and_capped_per_turn(initial_count):
    streams = [
        finished_calls(
            call_event("read_webpage", '{"url":"https://example.com/a"}', "read_1")
            + call_event("read_webpage", '{"url":"https://example.com/a"}', "read_2", index=1)
        ),
        event({"content": "An answer."}, "stop") + "data: [DONE]\n\n",
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=streams.pop(0)))
    ) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = AsyncMock(
            return_value='{"untrusted_evidence":{"kind":"page_text","url":"https://example.com/a","text":"Page read"}}'
        )
        record = ResponseRecord()
        initial = [f"https://example.com/initial-{index}" for index in range(initial_count)]
        record.web_sources.extend(initial)
        token = CHAT_OBSERVATION.set(record)
        try:
            _ = [part async for part in provider.stream([], "test")]
            assert record.web_sources == initial + (["https://example.com/a"] if initial_count < 4 else [])
        finally:
            CHAT_OBSERVATION.reset(token)
            await provider.close()


@pytest.mark.parametrize(
    "stream",
    [
        call_event() + "data: [DONE]\n\n",
        call_event() + event(finish="tool_calls"),
        finished_calls(call_event(arguments="not json")),
        finished_calls(call_event(arguments="[]")),
        finished_calls(call_event(call_id="same", index=0) + call_event(call_id="same", index=1)),
        finished_calls(call_event(index=4)),
        finished_calls(call_event(arguments='{"query":"' + "x" * 5000 + '"}')),
        finished_calls(event({"tool_calls": [{"index": 0, "function": "invalid"}]})),
    ],
)
async def test_incomplete_invalid_and_duplicate_calls_execute_nothing(stream):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=stream))
    ) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = AsyncMock()
        try:
            with pytest.raises(ProviderError):
                _ = [piece async for piece in provider.stream([], "test")]
            provider.web_tools.execute.assert_not_awaited()
        finally:
            await provider.close()


async def test_tool_budget_and_repeated_call_id_are_enforced():
    streams = [finished_calls(call_event()), finished_calls(call_event())]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=streams.pop(0)))
    ) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = AsyncMock(return_value='{"error":"unavailable"}')
        try:
            with pytest.raises(ProviderError, match="repeated"):
                _ = [piece async for piece in provider.stream([], "test")]
            assert provider.web_tools.execute.await_count == 1
        finally:
            await provider.close()


@pytest.mark.parametrize("ignore_disabled", [False, True])
async def test_last_round_disables_further_tools_after_search_and_read(ignore_disabled):
    requests = []
    execute = AsyncMock(
        side_effect=[
            '{"untrusted_evidence":{"results":[{"url":"https://example.com/article"}]}}',
            '{"untrusted_evidence":{"kind":"page_text","url":"https://example.com/article","text":"Verified release."}}',
        ]
    )

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            stream = finished_calls(call_event())
        elif len(requests) == 2:
            stream = finished_calls(
                call_event("read_webpage", '{"url":"https://example.com/article"}', "read")
            )
        elif body["tool_choice"] == "none" and not ignore_disabled:
            stream = event({"content": "Verified release."}, "stop") + "data: [DONE]\n\n"
        else:
            stream = finished_calls(call_event(call_id="extra"))
        return httpx.Response(200, text=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = execute
        record = ResponseRecord()
        token = CHAT_OBSERVATION.set(record)
        try:
            if ignore_disabled:
                with pytest.raises(ProviderError, match="turn limit"):
                    _ = [part async for part in provider.stream([], "test")]
            else:
                assert "".join([part async for part in provider.stream([], "test")]) == "Verified release."
                assert record.web_sources == ["https://example.com/article"]
            assert [r["tool_choice"] for r in requests] == ["auto", "auto", "none"]
            assert execute.await_count == 2
        finally:
            CHAT_OBSERVATION.reset(token)
            await provider.close()


@pytest.mark.parametrize("progress,short_generation", [(True, False), (False, False), (True, True)])
async def test_structured_progress_satisfies_voice_first_output_but_not_generation_deadline(
    progress, short_generation
):
    class DelayedCall(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (call_event() if progress else event({"role": "assistant"})).encode()
            await asyncio.sleep(0.08)
            yield (event(finish="tool_calls") + "data: [DONE]\n\n").encode()

    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=DelayedCall())
        return httpx.Response(200, text=event({"content": "Result."}, "stop") + "data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(
            enabled_settings(
                voice_first_content_timeout=0.02, generation_timeout=0.04 if short_generation else 2
            ),
            client,
        )
        provider.web_tools.execute = AsyncMock(return_value='{"error":"no fixture result"}')
        token = CHAT_OBSERVATION.set(ResponseRecord())
        try:
            if progress and not short_generation:
                assert "".join([part async for part in provider.stream([], "test")]) == "Result."
                assert provider.web_tools.execute.await_count == 1
            else:
                with pytest.raises(ProviderError):
                    _ = [part async for part in provider.stream([], "test")]
                provider.web_tools.execute.assert_not_awaited()
        finally:
            CHAT_OBSERVATION.reset(token)
            await provider.close()


async def test_nullable_tool_fields_from_compatible_gateway_are_accepted():
    streams = [
        finished_calls(
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{"},
                        }
                    ]
                }
            )
            + event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "type": None,
                            "function": {"name": None, "arguments": '"query":"public news"}'},
                        }
                    ]
                }
            )
        ),
        event({"content": "A grounded answer.", "tool_calls": None}, "stop") + "data: [DONE]\n\n",
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=streams.pop(0)))
    ) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = AsyncMock(return_value='{"untrusted_evidence":{}}')
        try:
            assert "".join([part async for part in provider.stream([], "test")]) == "A grounded answer."
            assert provider.web_tools.execute.await_count == 1
        finally:
            await provider.close()


async def test_tool_plans_never_spoken_on_cancel_and_guard_blocks_execution():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=finished_calls(call_event())))
    ) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = AsyncMock()
        token = TOOL_GUARD.set(AsyncMock(side_effect=[True, False]))
        try:
            with pytest.raises(ProviderError, match="conversation changed"):
                _ = [piece async for piece in provider.stream([], "test")]
            provider.web_tools.execute.assert_not_awaited()
        finally:
            TOOL_GUARD.reset(token)
            await provider.close()


async def test_cancelling_inflight_tool_cancels_runtime_and_does_not_requery_model():
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    requests = []

    async def execute(name, args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=finished_calls(call_event()))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(enabled_settings(), client)
        provider.web_tools.execute = execute

        async def collect():
            return [piece async for piece in provider.stream([], "test")]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert cancelled.is_set() and len(requests) == 1
            assert not provider._active_models["test"]
        finally:
            await provider.close()


@pytest.mark.parametrize("alias,disable_context", [("not-allowlisted", False), ("test", True)])
async def test_disabled_aliases_and_vision_context_preserve_original_streaming(alias, disable_context):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=event({"content": "Fast sentence."}) + "data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(enabled_settings(), client)
        token = WEB_TOOLS_ALLOWED.set(not disable_context)
        try:
            assert "".join([part async for part in provider.stream([], alias)]) == "Fast sentence."
            assert "tools" not in requests[0]
        finally:
            WEB_TOOLS_ALLOWED.reset(token)
            await provider.close()
