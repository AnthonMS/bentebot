"""Probe the real streamed tool loop using synthetic evidence, without web access.

Uses existing gateway environment credentials. Reports only safe contract/timing
metadata. Does not enable tools in production or change provider configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import replace
from urllib.parse import urlsplit

import httpx

from bentebot.config import Settings
from bentebot.providers import OpenAIChat, ProviderError

PAGE = "https://example.com/bentebot-qualification"
MARKER = "BENTE-Q7"


def deployment_for_host(payload: dict, alias: str, host: str) -> str:
    matches = [
        entry.get("model_info", {}).get("id")
        for entry in payload.get("data", [])
        if entry.get("model_name") == alias
        and urlsplit(entry.get("litellm_params", {}).get("api_base", "")).hostname == host
    ]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise ValueError("Expected exactly one deployment for the alias and host")
    return matches[0]


class DeploymentTransport(httpx.AsyncBaseTransport):
    """Pin only these diagnostic requests; retain alias-specific template settings."""

    def __init__(self, alias, deployment, host, inner=None):
        self.alias, self.deployment, self.host = alias, deployment, host
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport(trust_env=False)

    async def handle_async_request(self, request):
        body = json.loads(request.content)
        if body.get("model") != self.alias:
            raise ValueError("Unexpected model in qualification request")
        body["model"] = self.deployment
        headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
        pinned = httpx.Request(
            request.method,
            request.url,
            headers=headers,
            content=json.dumps(body).encode(),
            extensions=request.extensions,
        )
        response = await self.inner.handle_async_request(pinned)
        if response.status_code < 400 and (
            urlsplit(response.headers.get("x-litellm-model-api-base", "")).hostname != self.host
            or response.headers.get("x-litellm-model-id") != self.deployment
        ):
            await response.aclose()
            raise ValueError("Gateway did not confirm the requested deployment")
        return response

    async def aclose(self):
        await self.inner.aclose()


class SyntheticTools:
    """All tool operations terminate here. Never fetches a URL or runs a search."""

    def __init__(self):
        self.calls: list[str] = []

    async def execute(self, name: str, arguments: str) -> str:
        args = json.loads(arguments)
        if (
            not self.calls
            and name == "web_search"
            and isinstance(args.get("query"), str)
            and args["query"].strip()
        ):
            result = {"results": [{"title": "Qualification bulletin", "url": PAGE}]}
        elif self.calls == ["web_search"] and name == "read_webpage" and args.get("url") == PAGE:
            result = {"kind": "page_text", "url": PAGE, "text": f"The current release code is {MARKER}."}
        else:
            raise ValueError("Unexpected synthetic tool sequence")
        self.calls.append(name)
        return json.dumps({"untrusted_evidence": result})


class CapturedChat(OpenAIChat):
    def __init__(self, settings, client):
        super().__init__(settings, client)
        self.rounds = []
        self.first_content_at = None

    async def _stream(self, messages, model, tool_state=None):
        if tool_state is not None:
            self.rounds.append(tool_state)
        async for piece in super()._stream(messages, model, tool_state):
            if self.first_content_at is None:
                self.first_content_at = time.monotonic()
            yield piece


async def qualify(settings: Settings, model: str, client: httpx.AsyncClient, *, latency=False) -> dict:
    settings = replace(
        settings,
        web_tools_enabled=True,
        web_search_url="http://synthetic.invalid",
        web_tool_models=model,
        web_max_tool_rounds=2,
        chat_max_tokens=300,
        generation_timeout=90.0,
    )
    result = {"model": model, "streamed_tool_contract": "fail", "evidence": "synthetic_only"}
    provider = CapturedChat(settings, client)
    tools = SyntheticTools()
    provider.web_tools = tools
    started = time.monotonic()
    try:
        async with asyncio.timeout(95):
            parts = [
                piece
                async for piece in provider.stream(
                    [
                        {
                            "role": "system",
                            "content": "This is a synthetic qualification test. Use the supplied evidence to answer. "
                            "Search for the bulletin, read its page, then give only its release code.",
                        },
                        {
                            "role": "user",
                            "content": "What is the qualification bulletin's current release code?",
                        },
                    ],
                    model,
                )
            ]
        if tools.calls != ["web_search", "read_webpage"] or MARKER not in "".join(parts):
            raise ValueError("Missing search/read/grounded answer")
        if len(provider.rounds) != 3 or any(not state.saw_done for state in provider.rounds):
            raise ValueError("Incomplete streamed rounds")
        if [state.finish_reason for state in provider.rounds] != ["tool_calls", "tool_calls", "stop"]:
            raise ValueError("Invalid round completion")
        result.update(streamed_tool_contract="pass", tool_sequence=tools.calls, rounds=3)
        if latency:
            result["ordinary_reply_samples"] = []
            for enabled in (False, True):
                sample = CapturedChat(replace(settings, web_tools_enabled=enabled), client)
                # The completed fixture rejects further calls before any network I/O.
                if enabled:
                    sample.web_tools = tools
                sample.first_content_at = None
                began = time.monotonic()
                delivered = None
                async with asyncio.timeout(95):
                    async for _ in sample.stream(
                        [{"role": "user", "content": "Say hello in one short sentence. No lookup needed."}],
                        model,
                    ):
                        if delivered is None:
                            delivered = time.monotonic()
                if (
                    delivered is None
                    or sample.first_content_at is None
                    or any(s.calls for s in sample.rounds)
                ):
                    raise ValueError("Ordinary reply produced no text or requested a tool")
                result["ordinary_reply_samples"].append(
                    {
                        "tools_enabled": enabled,
                        "first_model_text_ms": round((sample.first_content_at - began) * 1000, 1),
                        "first_delivered_text_ms": round((delivered - began) * 1000, 1),
                        "buffer_delay_ms": round((delivered - sample.first_content_at) * 1000, 1),
                        "total_ms": round((time.monotonic() - began) * 1000, 1),
                    }
                )
    except (
        httpx.HTTPError,
        ProviderError,
        TimeoutError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        result.update(streamed_tool_contract="fail", error_type=type(exc).__name__)
        if isinstance(exc, ProviderError):
            # ProviderError is explicitly body-free and safe for public diagnostics.
            result["error_reason"] = str(exc)
    result["round_states"] = [
        {
            "finish_reason": state.finish_reason
            if state.finish_reason in {"", "stop", "tool_calls", "length", "content_filter"}
            else "other",
            "done": state.saw_done,
            "call_count": len(state.calls),
        }
        for state in provider.rounds
    ]
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="")
    parser.add_argument("--latency", action="store_true", help="Add two ordinary-reply timing samples")
    parser.add_argument(
        "--deployment-host", default="", help="Pin this probe to an existing gateway deployment host"
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    model = args.model or settings.chat_model
    hosts = set()
    statuses = []

    async def record_response(response):
        statuses.append(response.status_code)
        hosts.add(urlsplit(response.headers.get("x-litellm-model-api-base", "")).hostname or "unreported")

    transport = None
    try:
        if args.deployment_host:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as discovery:
                response = await discovery.get(
                    settings.chat_base_url.rstrip("/").removesuffix("/v1") + "/model/info",
                    headers={"Authorization": f"Bearer {settings.chat_api_key}"}
                    if settings.chat_api_key
                    else {},
                )
                response.raise_for_status()
                deployment = deployment_for_host(response.json(), model, args.deployment_host)
            transport = DeploymentTransport(model, deployment, args.deployment_host)
        async with httpx.AsyncClient(
            timeout=30, trust_env=False, transport=transport, event_hooks={"response": [record_response]}
        ) as client:
            result = await qualify(settings, model, client, latency=args.latency)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result = {"model": model, "streamed_tool_contract": "fail", "error_type": type(exc).__name__}
    if args.deployment_host:
        result["requested_deployment_host"] = args.deployment_host
    result.update(http_statuses=statuses, upstream_hosts=sorted(hosts))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["streamed_tool_contract"] == "pass" else 1)


if __name__ == "__main__":
    asyncio.run(main())
