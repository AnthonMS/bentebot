import asyncio
import json
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.tools import TOOL_GUARD, ToolError, WebTools, public_url, resolve_public


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "https://example.com:8000/",
        "https://user:pass@example.com/",
        "http://host.local/",
        "https://example.com/#fragment",
        "https://example.com\\@127.0.0.1/",
        "http://224.0.0.1/",
        "http://[64:ff9b::7f00:1]/",
        "http://[2002:7f00:1::]/",
        "http://[fe80::1%25eth0]/",
    ],
)
def test_fetch_rejects_nonpublic_and_ambiguous_urls(url):
    with pytest.raises(ToolError):
        public_url(url)


async def test_dns_rebinding_mixed_results_rejected(monkeypatch):
    resolver = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]
    )
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolver)
    with pytest.raises(ToolError):
        await resolve_public("example.com", 443)


async def test_search_reads_json_without_auth_and_pins_page_dns():
    requests = []

    async def search(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Recent event",
                        "content": "A snippet",
                        "publishedDate": "2026-10-28",
                    },
                    {"url": "http://127.0.0.1/secret", "title": "Private"},
                ]
            },
        )

    async def fetch(request):
        requests.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["Host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        assert "authorization" not in request.headers and not request.headers.get("cookie")
        return httpx.Response(
            200,
            text="<html><head><title>Hidden</title></head><script>secret()</script><p>Actual article.</p></html>",
            headers={"content-type": "text/html"},
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(search)) as search_client,
        httpx.AsyncClient(transport=httpx.MockTransport(fetch)) as fetch_client,
    ):
        tools = WebTools(
            "http://searxng:8080",
            search_client=search_client,
            fetch_client=fetch_client,
            resolver=AsyncMock(return_value="93.184.216.34"),
        )
        result = json.loads(
            await tools.execute("web_search", '{"query":"recent public event","time_range":"day"}')
        )["untrusted_evidence"]
        assert len(result["results"]) == 1
        assert "not_read_articles" in result["kind"]
        result = json.loads(await tools.execute("read_webpage", '{"url":"https://example.com/article"}'))[
            "untrusted_evidence"
        ]
        assert "Actual article." in result["text"]
        assert "secret" not in result["text"] and "Hidden" not in result["text"]
        assert requests[0].url.host == "searxng"
        assert requests[0].url.params["format"] == "json"
        assert tools.status()["read_webpage_successes"] == 1


async def test_redirect_to_private_network_never_connected():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = WebTools(
            "http://searx",
            search_client=client,
            fetch_client=client,
            resolver=AsyncMock(return_value="93.184.216.34"),
        )
        result = json.loads(await tools.execute("read_webpage", '{"url":"https://example.com/"}'))
        assert "error" in result and len(requests) == 1


async def test_each_public_redirect_is_resolved_and_pinned_separately():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": "https://other.example.com/article"})
        return httpx.Response(200, text="Read this", headers={"content-type": "text/plain"})

    resolver = AsyncMock(side_effect=["93.184.216.34", "93.184.216.35"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = WebTools("http://searx", search_client=client, fetch_client=client, resolver=resolver)
        result = json.loads(await tools.execute("read_webpage", '{"url":"https://example.com/"}'))
        assert result["untrusted_evidence"]["url"] == "https://other.example.com/article"
        assert [request.url.host for request in requests] == ["93.184.216.34", "93.184.216.35"]
        assert [request.headers["host"] for request in requests] == ["example.com", "other.example.com"]
        assert resolver.await_count == 2


async def test_guard_revocation_during_lookup_discards_results():
    guard = AsyncMock(side_effect=[True, False])
    token = TOOL_GUARD.set(guard)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": []}))
    ) as client:
        tools = WebTools("http://searx", search_client=client, fetch_client=client)
        try:
            result = json.loads(await tools.execute("web_search", '{"query":"public topic"}'))
            assert "error" in result and "conversation changed" in result["error"].lower()
        finally:
            TOOL_GUARD.reset(token)


async def test_tool_errors_are_sanitized_and_logs_exclude_arguments(caplog):
    caplog.set_level("INFO", logger="bentebot.tools")

    def handler(request):
        return httpx.Response(429, text="SECRET response body")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = WebTools("http://searx", search_client=client, fetch_client=client)
        result = await tools.execute("web_search", '{"query":"PRIVATE_QUERY"}')
        assert "SECRET" not in result and "PRIVATE_QUERY" not in result
        assert "SECRET" not in caplog.text and "PRIVATE_QUERY" not in caplog.text
        assert tools.status()["last_error_type"] == "HTTPStatusError"


async def test_hourly_limit_and_unknown_tools_do_not_execute():
    now = [1000.0]
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = WebTools(
            "http://searx", search_client=client, fetch_client=client, hourly_limit=1, clock=lambda: now[0]
        )
        assert "untrusted_evidence" in json.loads(await tools.execute("web_search", '{"query":"hi"}'))
        assert "error" in json.loads(await tools.execute("web_search", '{"query":"hi again"}'))
        assert len(requests) == 1
        now[0] += 3601
        assert "error" in json.loads(await tools.execute("run_shell", '{"code":"delete everything"}'))
        assert len(requests) == 1


async def test_bounded_decompressed_body_and_binary_rejection():
    responses = [
        httpx.Response(200, content=b"x" * 500_001, headers={"content-type": "text/plain"}),
        httpx.Response(200, content=b"PDF", headers={"content-type": "application/pdf"}),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: responses.pop(0))) as client:
        tools = WebTools(
            "http://searx",
            search_client=client,
            fetch_client=client,
            resolver=AsyncMock(return_value="93.184.216.34"),
        )
        for _ in range(2):
            assert "error" in json.loads(
                await tools.execute("read_webpage", '{"url":"https://example.com/"}')
            )


async def test_large_html_keeps_bounded_explicitly_partial_evidence_and_stops_reading():
    class LargePage(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"<html><h1>Official release announcement</h1><script>" + b"x" * 490_000
            yield b"x" * 20_000
            raise AssertionError("The reader must close the response after its bounded prefix")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/html"}, stream=LargePage())
        )
    ) as client:
        tools = WebTools(
            "http://searx",
            search_client=client,
            fetch_client=client,
            resolver=AsyncMock(return_value="93.184.216.34"),
        )
        result = json.loads(await tools.execute("read_webpage", '{"url":"https://example.com/release"}'))
        evidence = result["untrusted_evidence"]
        assert evidence["text"] == "Official release announcement"
        assert evidence["truncated"] and evidence["download_truncated"]
        assert tools.status()["read_webpage_successes"] == 1
