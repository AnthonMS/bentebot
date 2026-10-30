import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.model_warmth import VoiceModelWarmth
from bentebot.providers import OpenAIChat, ProviderError


def warmer(*, interval=120, warm=None, recent=False):
    chat = SimpleNamespace(warm=warm or AsyncMock(), recently_used=lambda *_: recent)
    return VoiceModelWarmth(
        Settings(voice_warm_models="local, other-local", voice_warm_interval=interval), chat
    )


async def test_shared_alias_loads_once_and_last_leave_cancels_refreshes():
    manager = warmer()
    try:
        first = manager.select(1, "local")
        assert manager.select(2, "local") is first
        await manager.prepare(1, "local")
        assert manager.chat.warm.await_count == 1
        assert "ready" in manager.describe(2)
        await manager.release(1)
        assert not first.task.done()
        await manager.release(2)
        assert first.task.cancelled()
        assert not manager.models and not manager.owners
    finally:
        await manager.close()


async def test_silent_call_refreshes_and_recent_chat_suppresses_refresh():
    refreshed = asyncio.Event()

    async def warm(model):
        if mock.await_count == 2:
            refreshed.set()

    mock = AsyncMock(side_effect=warm)
    manager = warmer(interval=0.01, warm=mock)
    try:
        await manager.prepare(1, "local")
        async with asyncio.timeout(1):
            await refreshed.wait()
        manager.chat.recently_used = lambda *_: True
        calls = mock.await_count
        await asyncio.sleep(0.04)
        assert mock.await_count == calls
    finally:
        await manager.close()


async def test_periodic_refresh_does_not_block_real_response():
    refreshing = asyncio.Event()

    async def warm(model):
        if mock.await_count > 1:
            refreshing.set()
            await asyncio.Event().wait()

    mock = AsyncMock(side_effect=warm)
    manager = warmer(interval=0.01, warm=mock)
    try:
        await manager.prepare(1, "local")
        await asyncio.wait_for(refreshing.wait(), 1)
        await asyncio.wait_for(manager.prepare(1, "local"), 0.1)
        assert manager.models["local"].status == "warming"
    finally:
        await manager.close()


async def test_cloud_alias_never_warms_and_switch_releases_previous_model():
    manager = warmer()
    try:
        await manager.prepare(1, "local")
        previous = manager.models["local"].task
        await manager.prepare(1, "other-local")
        assert previous.cancelled()
        await manager.prepare(1, "cloud")
        await manager.release(1)
        assert manager.chat.warm.await_count == 2
        assert not manager.models
        assert "disabled" in manager.describe(1)
    finally:
        await manager.close()


@pytest.mark.parametrize(
    "alias", ["claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-5-5", "gpt-6-luna", "gpt-6.1-sol"]
)
async def test_real_cloud_names_without_suffix_never_start_or_refresh_warmups(alias):
    manager = warmer(interval=0.01)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    try:
        await manager.prepare(1, "local")
        previous = manager.models["local"].task
        await manager.prepare(1, alias)
        await asyncio.sleep(0.04)
        assert previous.cancelled()
        assert not manager.owners and not manager.models
        assert "disabled" in manager.describe(1)
        manager.chat.warm.assert_awaited_once_with("local")
        # Also test the provider's independent guard, without the manager.
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            chat = OpenAIChat(Settings(voice_warm_models="local"), client)
            with pytest.raises(ProviderError, match="explicitly configured local"):
                await chat.warm(alias)
        assert requests == []
    finally:
        await manager.close()


async def test_selecting_cloud_chat_does_not_keep_warming_default_local_model(store):
    bot = BenteBot(Settings(health_port=0, chat_model="local", voice_warm_models="local"), store)
    bot.chat.warm = AsyncMock()
    try:
        await bot.prepare_voice_model(1, 10, "local")
        previous = bot.model_warmth.models["local"].task
        store.set("channel:10", "model", "claude-haiku-5-5")
        await bot.prepare_voice_model(1, 10, "claude-haiku-5-5")
        await asyncio.sleep(0)
        assert previous.cancelled()
        assert not bot.model_warmth.owners and not bot.model_warmth.models
        bot.chat.warm.assert_awaited_once_with("local")
    finally:
        await bot.close()


async def test_interrupted_answer_does_not_cancel_shared_loading_and_shutdown_does():
    started = asyncio.Event()

    async def warm(model):
        started.set()
        await asyncio.Event().wait()

    manager = warmer(warm=AsyncMock(side_effect=warm))
    answer = asyncio.create_task(manager.prepare(1, "local"))
    await started.wait()
    task = manager.models["local"].task
    answer.cancel()
    await asyncio.gather(answer, return_exceptions=True)
    assert not task.done()
    await manager.close()
    assert task.cancelled()
    assert not manager.models
    assert manager.select(2, "local") is None


async def test_failed_warm_releases_answer_gate_and_retries_without_logging_body(caplog):
    recovered = asyncio.Event()

    async def warm(model):
        if mock.await_count == 1:
            raise ProviderError("PRIVATE provider body")
        recovered.set()

    mock = AsyncMock(side_effect=warm)
    manager = warmer(interval=0.01, warm=mock)
    try:
        async with asyncio.timeout(1):
            await manager.prepare(1, "local")
            assert manager.models["local"].status == "failed"
            await recovered.wait()
        assert "PRIVATE" not in caplog.text
        assert "ProviderError" in caplog.text
    finally:
        await manager.close()


async def test_gateway_warm_is_uncached_bounded_and_disallows_unlisted_models():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(voice_warm_models="local"), client)
        await chat.warm("local")
        payload = json.loads(requests[0].content)
        assert payload["max_tokens"] == 1 and payload["stream"] is False
        assert payload["cache"] == {"no-cache": True, "no-store": True}
        assert len(payload["messages"]) == 1
        with pytest.raises(ProviderError, match="explicitly configured local"):
            await chat.warm("cloud")
        assert len(requests) == 1


@pytest.mark.parametrize("payload", [{"error": "PRIVATE"}, {}, [], {"choices": []}])
async def test_gateway_warm_rejects_malformed_success(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        chat = OpenAIChat(Settings(voice_warm_models="local"), client)
        with pytest.raises(ProviderError, match="rejected"):
            await chat.warm("local")


async def test_transient_vision_override_keeps_chat_model_selected(store):
    bot = BenteBot(Settings(health_port=0, chat_model="local", voice_warm_models="local"), store)
    bot.chat.warm = AsyncMock()
    try:
        await bot.prepare_voice_model(1, 10, "vision-cloud")
        await bot.model_warmth.prepare(1, "local")
        assert bot.model_warmth.owners == {1: "local"}
        bot.chat.warm.assert_awaited_once_with("local")
    finally:
        await bot.close()


@pytest.mark.parametrize("field,value", [("voice_warm_interval", 0), ("voice_warm_timeout", float("nan"))])
def test_warm_configuration_rejects_invalid_bounds(field, value):
    with pytest.raises(ValueError):
        Settings(**{field: value}).validate(login=False)
