import json

import httpx
import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.model_aliases import CLAUDE_MODELS, GPT_COMPAT_ALIASES, OPENAI_MODELS, canonical_model
from bentebot.model_warmth import VoiceModelWarmth
from bentebot.providers import OpenAIChat, ProviderError


async def test_model_inventory_hides_only_the_four_duplicate_gpt_aliases():
    names = [*OPENAI_MODELS, *CLAUDE_MODELS, *GPT_COMPAT_ALIASES, "qwen3.8-uncensored:fast"]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"id": n} for n in names]})
        )
    ) as client:
        chat = OpenAIChat(Settings(), client)
        try:
            assert set(await chat.models()) == set(names) - GPT_COMPAT_ALIASES.keys()
            assert set(await chat.models()) >= set(CLAUDE_MODELS)
        finally:
            await chat.close()


@pytest.mark.parametrize("alias,canonical", GPT_COMPAT_ALIASES.items())
async def test_old_saved_model_selections_resolve_to_canonical_gpt(store, alias, canonical):
    bot = BenteBot(Settings(health_port=0, chat_model=alias), store)
    try:
        assert bot.model_selection(10, 1) == (canonical, "deployment default")
        store.set("guild:1", "model", alias)
        assert bot.model_selection(10, 1) == (canonical, "server default")
        store.set("channel:10", "model", alias)
        assert bot.model_selection(10, 1) == (canonical, "channel override")
    finally:
        await bot.close()


@pytest.mark.parametrize("model", [*OPENAI_MODELS, *CLAUDE_MODELS, *GPT_COMPAT_ALIASES])
async def test_all_requested_cloud_aliases_use_explicit_provider_and_never_warm(model):
    calls = []
    settings = Settings(
        web_tools_enabled=True,
        native_web_openai_models=",".join(OPENAI_MODELS),
        native_web_anthropic_models=",".join(CLAUDE_MODELS),
        voice_warm_models="local-qwen",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    ) as client:
        chat = OpenAIChat(settings, client)
        manager = VoiceModelWarmth(settings, chat)
        try:
            expected = "openai" if canonical_model(model) in OPENAI_MODELS else "anthropic"
            assert chat.native_web_provider(model) == expected
            assert await manager.prepare(1, model) is None
            with pytest.raises(ProviderError, match="explicitly configured local"):
                await chat.warm(model)
            assert calls == []
        finally:
            await manager.close()
            await chat.close()


async def test_compat_alias_stream_uses_gpt_identity_and_canonical_request():
    requests = []

    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"Hello."},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        chat = OpenAIChat(Settings(), client)
        try:
            assert "".join([p async for p in chat.stream([], "claude-sol-6.1")]) == "Hello."
            assert requests[0]["model"] == "gpt-6.1-sol"
            assert "gpt-6.1-sol" in str(requests[0]["messages"])
        finally:
            await chat.close()
