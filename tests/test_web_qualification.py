import json
import runpy
from pathlib import Path

import httpx
import pytest

from bentebot.config import Settings

PROBE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "deploy/web-tools/qualify-stream.py"))


def test_deployment_selection_requires_one_exact_alias_and_host():
    def entry(host, alias="test", ident="nas-deployment"):
        return {
            "model_name": alias,
            "litellm_params": {"api_base": f"http://{host}:9292/v1"},
            "model_info": {"id": ident},
        }

    select = PROBE["deployment_for_host"]
    assert select({"data": [entry("pc"), entry("nas")]}, "test", "nas") == "nas-deployment"
    for data in [[], [entry("nas", alias="other")], [entry("nas"), entry("nas")]]:
        with pytest.raises(ValueError):
            select({"data": data}, "test", "nas")


@pytest.mark.parametrize("wrong", [False, True])
async def test_pinning_preserves_gateway_and_template_flags_and_checks_response(wrong):
    def handler(request):
        assert request.url == httpx.URL("http://gateway.test/v1/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == "nas-deployment"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert int(request.headers["content-length"]) == len(request.content)
        return httpx.Response(
            200,
            headers={
                "x-litellm-model-api-base": "http://nas:9292/v1",
                "x-litellm-model-id": "other" if wrong else "nas-deployment",
            },
        )

    transport = PROBE["DeploymentTransport"]("test", "nas-deployment", "nas", httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        if wrong:
            with pytest.raises(ValueError):
                await client.post(
                    "http://gateway.test/v1/chat/completions",
                    json={"model": "test", "chat_template_kwargs": {"enable_thinking": False}},
                )
        else:
            response = await client.post(
                "http://gateway.test/v1/chat/completions",
                json={"model": "test", "chat_template_kwargs": {"enable_thinking": False}},
            )
            assert response.status_code == 200


def event(delta, finish=None):
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n"


def call(name, args, ident):
    return (
        event(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": ident,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ]
            },
            "tool_calls",
        )
        + "data: [DONE]\n\n"
    )


@pytest.mark.parametrize("failure", [None, "truncated_call", "truncated_answer", "wrong_page", "ungrounded"])
async def test_streamed_probe_uses_only_synthetic_evidence_and_requires_complete_rounds(failure):
    page = "https://example.com/wrong" if failure == "wrong_page" else PROBE["PAGE"]
    streams = [
        call("web_search", '{"query":"qualification bulletin"}', "first"),
        call("read_webpage", json.dumps({"url": page}), "second"),
        event({"content": "wrong" if failure == "ungrounded" else PROBE["MARKER"]}, "stop")
        + "data: [DONE]\n\n",
    ]
    if failure == "truncated_call":
        streams[0] = streams[0].replace("data: [DONE]\n\n", "")
    if failure == "truncated_answer":
        streams[2] = streams[2].replace("data: [DONE]\n\n", "")
    requests = []

    def handler(request):
        # A fetch of a search/page URL would fail here, even if the probe regressed.
        assert request.url == httpx.URL("http://gateway.test/v1/chat/completions")
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["tool_choice"] == ("none" if len(requests) == 2 else "auto")
        requests.append(body)
        return httpx.Response(200, text=streams.pop(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await PROBE["qualify"](Settings(chat_base_url="http://gateway.test/v1"), "test", client)
    assert result["streamed_tool_contract"] == ("pass" if failure is None else "fail")
    assert "BENTE-Q7" not in json.dumps(result)
    if failure is None:
        assert len(requests) == 3
        assert result["tool_sequence"] == ["web_search", "read_webpage"]
        assert "page_text" in requests[-1]["messages"][-1]["content"]


async def test_latency_comparison_really_disables_tools_and_never_fetches_live_pages():
    streams = [
        call("web_search", '{"query":"bulletin"}', "first"),
        call("read_webpage", json.dumps({"url": PROBE["PAGE"]}), "second"),
        event({"content": PROBE["MARKER"]}, "stop") + "data: [DONE]\n\n",
        event({"content": "Hello."}, "stop") + "data: [DONE]\n\n",
        event({"content": "Hello."}, "stop") + "data: [DONE]\n\n",
    ]
    requests = []

    def handler(request):
        assert request.url.host == "gateway.test"
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=streams.pop(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await PROBE["qualify"](
            Settings(chat_base_url="http://gateway.test/v1"), "test", client, latency=True
        )
    assert result["streamed_tool_contract"] == "pass"
    assert "tools" not in requests[3]
    assert requests[4]["tool_choice"] == "auto"
    assert [s["tools_enabled"] for s in result["ordinary_reply_samples"]] == [False, True]
