"""Read configured gateway credentials without displaying them. No web calls are made."""

import argparse
import asyncio
import json
from urllib.parse import urlsplit

import httpx

from bentebot.config import Settings
from bentebot.tools import WEB_SCHEMAS


async def main():
    parser = argparse.ArgumentParser(description="Qualify one LiteLLM alias for Bentebot web-tool calls")
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    settings = Settings.from_env()
    selected = args.model or settings.chat_model
    result = {"model": selected, "tool_contract": "fail", "upstream_host": "unreported"}
    headers = {"Authorization": f"Bearer {settings.chat_api_key}"} if settings.chat_api_key else {}
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.post(
                settings.chat_base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json={
                    "model": selected,
                    "messages": [
                        {
                            "role": "user",
                            "content": "This is an offline contract probe. Call web_search with query 'Bentebot tool contract test'.",
                        }
                    ],
                    "tools": WEB_SCHEMAS,
                    "tool_choice": {"type": "function", "function": {"name": "web_search"}},
                    "parallel_tool_calls": False,
                    "max_tokens": 200,
                    "stream": False,
                },
            )
            result["http_status"] = response.status_code
            result["upstream_host"] = (
                urlsplit(response.headers.get("x-litellm-model-api-base", "")).hostname or "unreported"
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            calls = choice["message"].get("tool_calls", [])
            if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
                raise ValueError("Missing tool call")
            call = calls[0]
            arguments = json.loads(call["function"]["arguments"])
            if (
                not call.get("id")
                or call["function"]["name"] != "web_search"
                or not isinstance(arguments.get("query"), str)
            ):
                raise ValueError("Invalid tool call")
            result["tool_contract"] = "pass"
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["error_type"] = type(exc).__name__
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["tool_contract"] == "pass" else 1)


if __name__ == "__main__":
    asyncio.run(main())
