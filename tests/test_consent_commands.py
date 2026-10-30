import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from discord import app_commands

from bentebot.config import Settings
from bentebot.consent import VoiceConsent
from bentebot.discord_app import BenteBot

VOICE_ID = "voice_" + "a" * 32


def interaction(user_id=100, guild_id=1):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id),
        guild_id=guild_id,
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock(), edit_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
        message=SimpleNamespace(edit=AsyncMock()),
    )


def button(view, label):
    return next(item for item in view.children if item.label == label)


async def test_only_selected_speaker_can_answer_voice_consent():
    start = AsyncMock()
    view = VoiceConsent(100, start)
    other = interaction(200)
    owner = interaction(100)
    try:
        assert not await view.interaction_check(other)
        other.response.send_message.assert_awaited_once()
        assert other.response.send_message.call_args.kwargs["ephemeral"]
        assert await view.interaction_check(owner)
        start.assert_not_awaited()
    finally:
        view.stop()


async def test_concurrent_allow_callbacks_consume_consent_once_before_network_wait():
    start = AsyncMock()
    view = VoiceConsent(100, start)
    first, duplicate = interaction(), interaction()
    entered, release = asyncio.Event(), asyncio.Event()

    async def defer(**kwargs):
        entered.set()
        await release.wait()

    first.response.defer.side_effect = defer
    first_callback = asyncio.create_task(button(view, "Allow voice sample").callback(first))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        # Discord can dispatch another button callback before its first HTTP response
        # completes. Disabling the visible controls does not cancel that task.
        await button(view, "Allow voice sample").callback(duplicate)
        release.set()
        await asyncio.wait_for(first_callback, 1)
        start.assert_awaited_once()
        assert view.is_finished()
    finally:
        release.set()
        await asyncio.gather(first_callback, return_exceptions=True)
        view.stop()


async def test_already_dispatched_allow_cannot_override_declined_consent():
    start = AsyncMock()
    view = VoiceConsent(100, start)
    decline, late_allow = interaction(), interaction()
    entered, release = asyncio.Event(), asyncio.Event()

    async def edit_message(**kwargs):
        entered.set()
        await release.wait()

    decline.response.edit_message.side_effect = edit_message
    decline_callback = asyncio.create_task(button(view, "Decline").callback(decline))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await button(view, "Allow voice sample").callback(late_allow)
        release.set()
        await asyncio.wait_for(decline_callback, 1)
        start.assert_not_awaited()
        assert view.is_finished()
    finally:
        release.set()
        await asyncio.gather(decline_callback, return_exceptions=True)
        view.stop()


@pytest.mark.parametrize("actor,guild_id,is_admin", [(100, 1, False), (200, 1, False), (200, 2, True)])
async def test_profile_delete_requires_admin_and_matching_server(store, actor, guild_id, is_admin):
    bot = BenteBot(Settings(health_port=0), store)
    bot.profiles = SimpleNamespace(delete=AsyncMock(), close=AsyncMock())
    bot.is_admin = lambda user: is_admin
    store.set("voice_profiles", VOICE_ID, json.dumps({"owner_id": 100, "guild_id": 1}))
    command = bot.tree.get_command("speech").get_command("delete_voice")
    try:
        with pytest.raises(app_commands.CheckFailure):
            await command.callback(interaction(actor, guild_id), voice_id=VOICE_ID)
        bot.profiles.delete.assert_not_awaited()
        assert bot.profile_info(VOICE_ID) is not None
    finally:
        await bot.close()


async def test_admin_can_delete_other_speakers_profile_after_provider_confirmation(store):
    bot = BenteBot(Settings(health_port=0), store)
    bot.profiles = SimpleNamespace(delete=AsyncMock(), close=AsyncMock())
    bot.quiesce = AsyncMock()
    bot.is_admin = lambda user: True
    store.set("voice_profiles", VOICE_ID, json.dumps({"owner_id": 100, "guild_id": 1}))
    store.set("guild:1", "voice", VOICE_ID)
    command = bot.tree.get_command("speech").get_command("delete_voice")
    try:
        await command.callback(interaction(200), voice_id=VOICE_ID)
        bot.profiles.delete.assert_awaited_once_with(VOICE_ID)
        bot.quiesce.assert_awaited_once_with(1)
        assert bot.profile_info(VOICE_ID) is None
        assert store.get("guild:1", "voice", "missing") == "missing"
    finally:
        await bot.close()


@pytest.mark.parametrize("name", [VOICE_ID, f" {VOICE_ID}", f"{VOICE_ID} ", f"\t{VOICE_ID}\n"])
async def test_custom_voice_selection_validates_normalized_id_and_server_ownership(store, name):
    bot = BenteBot(Settings(health_port=0), store)
    bot.is_admin = lambda user: True
    store.set("voice_profiles", VOICE_ID, json.dumps({"owner_id": 100, "guild_id": 2}))
    command = bot.tree.get_command("speech").get_command("voice")
    try:
        with pytest.raises(ValueError, match="consenting owner"):
            await command.callback(interaction(), name=name)
        assert store.get("guild:1", "voice", "missing") == "missing"
    finally:
        await bot.close()


async def test_fresh_install_voice_autocomplete_works_without_saved_guild_voice(store):
    bot = BenteBot(Settings(health_port=0, tts_voice="builtin", tts_voices="builtin,second"), store)
    command = bot.tree.get_command("speech").get_command("voice")
    try:
        autocomplete = command._params["name"].autocomplete
        choices = await autocomplete(interaction(), "")
        assert {choice.value for choice in choices} == {"builtin", "second"}
    finally:
        await bot.close()
