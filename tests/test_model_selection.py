import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import httpx
import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.domain import Metrics, ResponseRecord, Source
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat

HAIKU = "claude-haiku-4-5"


def channel(identifier, name, *, voice=False, viewers=(100, 999)):
    result = MagicMock(spec=discord.VoiceChannel if voice else discord.TextChannel)
    result.id, result.name, result.category = identifier, name, None
    result.permissions_for.side_effect = lambda member: SimpleNamespace(view_channel=member.id in viewers)
    return result


def setup(bot):
    voice, text = channel(10, "dev-audio", voice=True), channel(20, "bot-stuff")
    guild = SimpleNamespace(id=1, me=SimpleNamespace(id=999), channels=[voice, text], voice_client=None)
    user = SimpleNamespace(id=100, voice=SimpleNamespace(channel=voice))
    done = [False]

    async def defer(**kwargs):
        done[0] = True

    interaction = SimpleNamespace(
        user=user,
        guild_id=1,
        guild=guild,
        channel_id=20,
        response=SimpleNamespace(
            defer=AsyncMock(side_effect=defer), is_done=lambda: done[0], send_message=AsyncMock()
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1),
        task=None,
        research=None,
        metrics=Metrics(),
        paused=False,
        latest_speaker_id=None,
        partial={},
    )
    bot.voice_sessions[1] = (conversation, SimpleNamespace(speakers={}, last_error=""), None)
    bot.chat.models = AsyncMock(return_value=["qwen", HAIKU])
    bot.is_admin = lambda _: True
    return interaction, conversation, voice, text


async def test_selection_from_text_defaults_to_call_and_marks_current_model(store):
    bot = BenteBot(Settings(health_port=0, chat_model="qwen"), store)
    interaction, _, _, _ = setup(bot)
    group = bot.tree.get_command("model")
    try:
        await group.get_command("set").callback(interaction, HAIKU)
        assert bot.model(10, 1) == HAIKU and bot.model(20, 1) == "qwen"
        assert "Active voice <#10>: " + HAIKU in interaction.followup.send.call_args.args[0]
        await group.get_command("list").callback(interaction)
        text = interaction.followup.send.call_args.args[0]
        assert r"\* " + HAIKU + " [voice]" in text
        assert "Command text channel <#20>: qwen" in text
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_explicit_channel_and_server_default_preserve_overrides(store):
    bot = BenteBot(Settings(health_port=0, chat_model="qwen"), store)
    interaction, _, _, _ = setup(bot)
    command = bot.tree.get_command("model").get_command("set")
    try:
        await command.callback(interaction, "qwen", channel="10")
        await command.callback(interaction, HAIKU, scope="server")
        assert bot.model_selection(10, 1) == ("qwen", "channel override")
        assert bot.model_selection(20, 1) == (HAIKU, "server default")
        await command.callback(interaction, HAIKU, channel="bot-stuff")
        assert bot.model_selection(20, 1) == (HAIKU, "channel override")
        with pytest.raises(ValueError, match="no channel"):
            await command.callback(interaction, "qwen", channel="10", scope="server")
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_gated_channels_are_searchable_and_prioritized_even_with_many_open_channels(store):
    bot = BenteBot(Settings(health_port=0), store)
    interaction, _, voice, text = setup(bot)
    # Gated channels only grant access to this member and bot, not @everyone.
    interaction.guild.channels = [channel(1000 + i, f"open-{i}") for i in range(40)] + [
        voice,
        text,
        channel(90, "private-other", viewers=(999,)),
        channel(91, "bot-denied", viewers=(100,)),
    ]
    autocomplete = bot.tree.get_command("model").get_command("set")._params["channel"].autocomplete
    try:
        choices = await autocomplete(interaction, "")
        assert len(choices) == 25 and [c.value for c in choices[:2]] == ["10", "20"]
        assert "active call" in choices[0].name and "command channel" in choices[1].name
        assert [c.value for c in await autocomplete(interaction, "bot stuff")] == ["20"]
        assert [c.value for c in await autocomplete(interaction, "open-39")] == ["1039"]
        assert not await autocomplete(interaction, "private-other")
        assert not await autocomplete(interaction, "bot-denied")
        text.permissions_for.return_value = SimpleNamespace(view_channel=False)
        text.permissions_for.side_effect = None
        with pytest.raises(ValueError, match="both you and the bot"):
            await bot.tree.get_command("model").get_command("set").callback(interaction, HAIKU, channel="20")
        assert store.get("channel:20", "model") == ""
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_voice_status_distinguishes_new_selection_from_old_request_and_defers_lookup(store):
    bot = BenteBot(Settings(health_port=0, chat_model="qwen"), store)
    interaction, conversation, _, _ = setup(bot)
    store.set("channel:10", "model", HAIKU)
    conversation.metrics.last_chat.update(
        requested_model="qwen", reported_model="qwen", gateway_deployment_id="deployment-nas"
    )
    bot.chat.route_metadata = AsyncMock(return_value={"gateway_backend_model": "openai/qwen3.5-4b:fast"})
    try:
        await bot.tree.get_command("voice").get_command("status").callback(interaction)
        text = "\n".join(c.args[0] for c in interaction.followup.send.await_args_list)
        assert "voice_model_selected: " + HAIKU in text
        assert "requested_model: qwen" in text and "gateway_backend_model: openai/qwen3.5-4b:fast" in text
        assert "may predate a model switch" in text
        interaction.response.defer.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_runtime_identity_is_current_and_does_not_mutate_history_or_enable_warming():
    history = [
        {"role": "system", "content": "Personality"},
        {"role": "assistant", "content": "I am old Qwen"},
        {"role": "user", "content": "Which model are you?"},
    ]
    original = deepcopy(history)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert HAIKU in body["messages"][0]["content"]
        assert "Prefer this metadata" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            headers={
                "x-litellm-model-id": "deployment-haiku",
                "x-litellm-model-api-base": "http://proxy:8317/v1",
            },
            text='data: {"model":"claude-haiku-4-5","choices":[{"delta":{"content":"Using Haiku."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        )

    token = CHAT_OBSERVATION.set(record := ResponseRecord())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(voice_warm_models="local"), client)
        try:
            assert "".join([p async for p in chat.stream(history, HAIKU)]) == "Using Haiku."
            assert history == original and len(requests) == 1
            assert record.chat_metadata["gateway_deployment_id"] == "deployment-haiku"
            assert record.chat_metadata["reported_model"] == HAIKU
            assert record.chat_metadata["upstream_host"] == "proxy"
        finally:
            CHAT_OBSERVATION.reset(token)
            await chat.close()


@pytest.mark.parametrize("forbidden", [True, False])
async def test_deployment_lookup_is_read_only_cached_and_does_not_guess_alias_or_expose_keys(forbidden):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET" and request.url.path == "/model/info"
        return httpx.Response(
            403 if forbidden else 200,
            json={
                "data": [
                    {
                        "model_name": "fast",
                        "model_info": {"id": "nas-id"},
                        "litellm_params": {"model": "openai/qwen3.5-4b:fast", "api_key": "SECRET"},
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(chat_base_url="http://gateway:4000/v1"), client)
        try:
            assert await chat.route_metadata({"requested_model": "fast"}) == {}
            result = await chat.route_metadata({"gateway_deployment_id": "nas-id"})
            assert result == ({} if forbidden else {"gateway_backend_model": "openai/qwen3.5-4b:fast"})
            assert await chat.route_metadata({"gateway_deployment_id": "unknown"}) == {}
            assert "SECRET" not in json.dumps(result) and len(requests) == 1
        finally:
            await chat.close()
