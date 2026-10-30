import asyncio
import json
from copy import deepcopy

import httpx

from bentebot.config import Settings
from bentebot.domain import ResponseRecord
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat, compatible_messages


def test_system_instructions_merge_without_changing_authority_order_or_history():
    messages = [
        {"role": "system", "content": "Personality"},
        {"role": "system", "content": "Voice delivery"},
        {"role": "system", "content": "Explain when asked"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    original = deepcopy(messages)
    normalized = compatible_messages(messages)
    assert normalized == [
        {"role": "system", "content": "Personality\n\nVoice delivery\n\nExplain when asked"},
        *original[3:],
    ]
    assert messages == original
    structured = [{"role": "system", "content": [{"type": "text", "text": "Keep structured"}]}]
    assert compatible_messages(structured) == structured


async def test_actual_chat_payload_works_with_single_system_template():
    def handler(request):
        messages = json.loads(request.content)["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[0]["content"].startswith("Personality\n\nVoice rules\n\n")
        assert 'selected LiteLLM model alias is "local"' in messages[0]["content"]
        return httpx.Response(200, text=event({"content": "Stockholm."}) + "data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)
        messages = [
            {"role": "system", "content": "Personality"},
            {"role": "system", "content": "Voice rules"},
            {"role": "user", "content": "Capital?"},
        ]
        assert [piece async for piece in chat.stream(messages, "local")] == ["Stockholm."]


def event(delta):
    return "data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n"


async def test_chat_trace_records_route_and_reasoning_counts_without_private_content():
    private = "PRIVATE THOUGHTS AND USER CONTENT"
    stream = event({"reasoning_content": private}) + event({"content": "Stockholm."}) + "data: [DONE]\n\n"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=stream,
                headers={"x-litellm-model-api-base": "http://user:SECRET@192.168.50.10:9292/v1?key=SECRET"},
            )
        )
    ) as client:
        chat = OpenAIChat(Settings(), client)
        record = ResponseRecord()
        token = CHAT_OBSERVATION.set(record)
        try:
            assert [text async for text in chat.stream([{"role": "user", "content": private}], "local")] == [
                "Stockholm."
            ]
        finally:
            CHAT_OBSERVATION.reset(token)
        expected = {
            "requested_model": "local",
            "upstream_host": "192.168.50.10",
            "reasoning_chunks": 1,
            "reasoning_chars": len(private),
        }
        assert {key: record.chat_metadata[key] for key in expected} == expected
        assert record.chat_metadata["prompt_messages"] == 2
        assert record.chat_metadata["prompt_chars"] > len(private)
        assert record.chat_metadata["gateway_deployment_id"] == ""
        assert record.timings["chat_headers_ms"] <= record.timings["chat_first_reasoning_ms"]
        assert "chat_first_event_ms" in record.timings
        assert private not in str(record.chat_metadata)
        assert "SECRET" not in str(record.chat_metadata)


async def test_concurrent_requests_keep_observations_separate():
    async def handler(request):
        body = json.loads(request.content)
        await asyncio.sleep(0)
        model = body["model"]
        return httpx.Response(
            200,
            text=event({"reasoning": model}) + event({"content": "Done."}) + "data: [DONE]\n\n",
            headers={"x-litellm-model-api-base": f"http://{model}.local/v1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)

        async def run(model):
            record = ResponseRecord()
            token = CHAT_OBSERVATION.set(record)
            try:
                async for _ in chat.stream([], model):
                    pass
            finally:
                CHAT_OBSERVATION.reset(token)
            return record

        first, second = await asyncio.gather(run("pc"), run("nas"))
        assert first.chat_metadata["upstream_host"] == "pc.local"
        assert second.chat_metadata["upstream_host"] == "nas.local"
        assert first.chat_metadata["reasoning_chars"] == 2
        assert second.chat_metadata["reasoning_chars"] == 3
        assert CHAT_OBSERVATION.get() is None


async def test_prewarm_reports_its_host_without_overwriting_real_turn_trace():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}}]},
                headers={"x-litellm-model-api-base": "http://192.168.50.20:9292/v1"},
            )
        )
    ) as client:
        chat = OpenAIChat(Settings(voice_warm_models="local"), client)
        record = ResponseRecord(chat_metadata={"requested_model": "actual-turn"})
        token = CHAT_OBSERVATION.set(record)
        try:
            assert await chat.warm("local") == "192.168.50.20"
            assert record.chat_metadata == {"requested_model": "actual-turn"}
            assert not record.timings
        finally:
            CHAT_OBSERVATION.reset(token)
