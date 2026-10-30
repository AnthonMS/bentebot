import json
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.domain import ResponseRecord
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat, ProviderError
from bentebot.speech import SentenceBuffer, VoiceLinkFilter, speak_response


def prose(text):
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    )


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


def settings():
    return Settings(web_tools_enabled=True, web_search_url="http://searxng:8080", web_tool_models="test")


@pytest.mark.parametrize("ignored", ["", "repeat_search", "plain_answer"])
async def test_real_search_reserves_next_round_for_page_read(ignored):
    requests = []
    second = (
        prose("Unverified answer.")
        if ignored == "plain_answer"
        else call("web_search" if ignored else "read_webpage", "read_2")
    )
    streams = [call("web_search", "search_1"), second, prose("The release is verified.")]

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=streams.pop(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(settings(), client)
        provider.web_tools.execute = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "untrusted_evidence": {
                            "kind": "search_results_not_read_articles",
                            "results": [{"url": "https://example.com/release"}],
                        }
                    }
                ),
                json.dumps(
                    {
                        "untrusted_evidence": {
                            "kind": "page_text",
                            "url": "https://example.com/release",
                            "text": "Official release",
                        }
                    }
                ),
            ]
        )
        record = ResponseRecord()
        token = CHAT_OBSERVATION.set(record)
        try:
            if ignored:
                with pytest.raises(ProviderError, match="required page"):
                    _ = [piece async for piece in provider.stream([], "test")]
                assert provider.web_tools.execute.await_count == 1
            else:
                assert (
                    "".join([piece async for piece in provider.stream([], "test")])
                    == "The release is verified."
                )
                assert record.web_sources == ["https://example.com/release"]
                assert requests[2]["tool_choice"] == "none"
                assert "tools" not in requests[2] and "parallel_tool_calls" not in requests[2]
                assert "Official release" in requests[2]["messages"][-1]["content"]
                assert "Untrusted web evidence" in requests[2]["messages"][-1]["content"]
                assert not any(
                    message.get("tool_calls") or message["role"] == "tool"
                    for message in requests[2]["messages"]
                )
            assert requests[1]["tool_choice"] == {"type": "function", "function": {"name": "read_webpage"}}
            assert [schema["function"]["name"] for schema in requests[1]["tools"]] == ["read_webpage"]
        finally:
            CHAT_OBSERVATION.reset(token)
            await provider.close()


@pytest.mark.parametrize(
    "answer",
    [
        "<tool_call>\n<function=read_webpage>\n<parameter=url>https://example.com</parameter>\n</function>\n</tool_call>",
        'The result is: <tool_call>{"name":"web_search"}</tool_call>',
        '{"name":"read_webpage","arguments":{"url":"https://example.com"}}',
        '```json\n{"tool_calls":[{"function":{"name":"read_webpage"}}]}\n```',
    ],
)
async def test_plaintext_tool_protocol_is_never_yielded_or_executed(answer):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=prose(answer)))
    ) as client:
        provider = OpenAIChat(settings(), client)
        provider.web_tools.execute = AsyncMock()
        output = []
        try:
            with pytest.raises(ProviderError, match="tool request instead of an answer"):
                async for piece in provider.stream([], "test"):
                    output.append(piece)
            assert output == []
            provider.web_tools.execute.assert_not_awaited()
        finally:
            await provider.close()


@pytest.mark.parametrize("size", [1, 2, 7, 10000])
def test_spoken_link_filter_handles_split_and_long_urls_without_losing_prose(size):
    text = (
        "Release 16.2.1 is out. [Official page](https://example.com/"
        + "a" * 2000
        + ") Read it. <https://example.org/release> www.example.net/path Done."
    )
    links, splitter = VoiceLinkFilter(), SentenceBuffer(max_chars=80)
    spoken = []
    for offset in range(0, len(text), size):
        spoken.extend(splitter.feed(links.feed(text[offset : offset + size])))
        assert len(links.pending) < 8
    spoken.extend(splitter.feed(links.feed("", final=True), final=True))
    answer = " ".join(spoken)
    assert "Release 16.2.1 is out." in answer
    assert "Official page" in answer and "Read it." in answer and "Done." in answer
    assert "https" not in answer and "example" not in answer and "aaa" not in answer
    assert "www" not in answer and "<>" not in answer


async def test_url_filter_changes_only_audio_not_generated_text(wav_bytes):
    raw = "Verified release. https://example.com/release"

    async def tokens():
        for char in raw:
            yield char

    spoken = []

    async def synth(text):
        spoken.append(text)
        return wav_bytes

    async def play(pcm, text, progress):
        progress(text)

    async def allowed():
        return True

    record = ResponseRecord()
    await speak_response(tokens(), synth, play, allowed, record, suppress_urls=True)
    assert spoken == ["Verified release."]
    assert record.generated == raw and record.spoken == "Verified release."
