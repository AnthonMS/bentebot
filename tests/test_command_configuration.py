from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from discord import app_commands

from bentebot.config import Settings
from bentebot.discord_app import BenteBot, ephemeral_status
from bentebot.domain import Source


async def test_long_requested_diagnostics_keep_all_content_in_ephemeral_chunks():
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock()), followup=SimpleNamespace(send=AsyncMock())
    )
    await ephemeral_status(interaction, "metric: evidence\n" * 400)
    calls = [*interaction.response.send_message.await_args_list, *interaction.followup.send.await_args_list]
    assert len(calls) > 1 and all(len(call.args[0]) <= 1900 for call in calls)
    assert all(call.kwargs["ephemeral"] for call in calls)
    assert sum(call.args[0].count("metric: evidence") for call in calls) == 400


async def test_clone_autocomplete_only_current_humans_and_execution_rechecks_membership(store):
    bot = BenteBot(Settings(health_port=0, native_viewer_id=123), store)
    bot.is_admin = lambda _: False
    channel = SimpleNamespace(id=10, members=[])
    guild = SimpleNamespace(
        id=1,
        get_channel=lambda _: channel,
        get_member=lambda user: next((item for item in channel.members if item.id == user), None),
    )
    owner = SimpleNamespace(
        id=100,
        name="bentebent",
        display_name="Bentebent",
        bot=False,
        guild=guild,
        voice=SimpleNamespace(channel=channel),
    )
    viewer = SimpleNamespace(id=123, bot=False, name="viewer", display_name="Viewer")
    otherbot = SimpleNamespace(id=999, bot=True, name="bot", display_name="Bot")
    channel.members = [owner, viewer, otherbot]
    bot.voice_sessions[1] = (SimpleNamespace(source=Source("channel", 10, 1), paused=False), None, None)
    interaction = SimpleNamespace(user=owner, guild_id=1, guild=guild)
    command = bot.tree.get_command("voice").get_command("clone")
    try:
        choices = await command._params["user"].autocomplete(interaction, "@bent")
        assert [choice.value for choice in choices] == ["100"]
        owner.voice.channel = SimpleNamespace(id=20)
        with pytest.raises(app_commands.CheckFailure):
            await command.callback(interaction, "100")
        assert bot.enrollments.active_count == 0
    finally:
        bot.voice_sessions.clear()
        await bot.close()


@pytest.mark.parametrize(
    "settings",
    [
        Settings(vision_mode="continuous"),
        Settings(vision_timeout=0),
        Settings(vision_requests_per_hour=0),
        Settings(web_tools_enabled=True),
        Settings(web_search_url="http://user:password@searxng:8080"),
        Settings(web_max_tool_rounds=10),
        Settings(voice_clone_speech_seconds=10),
        Settings(voice_clone_wait_seconds=1000),
    ],
)
def test_new_settings_reject_unbounded_or_incomplete_configuration(settings):
    with pytest.raises(ValueError):
        settings.validate(login=False)
