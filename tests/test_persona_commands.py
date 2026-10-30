from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from discord import app_commands

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.persona import PersonaStore


def interaction():
    return SimpleNamespace(
        user=SimpleNamespace(id=100),
        guild_id=1,
        guild=SimpleNamespace(id=1),
        channel_id=20,
        response=SimpleNamespace(
            is_done=Mock(return_value=False), send_message=AsyncMock(), send_modal=AsyncMock()
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )


async def test_persona_commands_require_admin_and_modal_rechecks_after_opening(store):
    bot = BenteBot(Settings(health_port=0), store)
    permission = [False]
    bot.is_admin = lambda _: permission[0]
    bot.resolve_model_channel = lambda _, value: None if value is None else SimpleNamespace(id=10)
    group = bot.tree.get_command("persona")
    it = interaction()
    try:
        with pytest.raises(app_commands.CheckFailure):
            await group.get_command("add").callback(it, "Unauthorized rule")
        assert not store.get("guild:1", PersonaStore.key)
        permission[0] = True
        await group.get_command("add").callback(it, "Use subtle dry humour.")
        before = store.get("guild:1", PersonaStore.key)
        assert before
        await group.get_command("edit").callback(it, 1)
        editor = it.response.send_modal.await_args.args[0]
        assert editor.text.default
        permission[0] = False
        with pytest.raises(app_commands.CheckFailure):
            await editor.on_submit(interaction())
        assert store.get("guild:1", PersonaStore.key) == before
    finally:
        await bot.close()


async def test_persona_scope_conflicts_are_explicit(store):
    bot = BenteBot(Settings(health_port=0), store)
    bot.is_admin = lambda _: True
    bot.resolve_model_channel = lambda *_: SimpleNamespace(id=10)
    try:
        with pytest.raises(ValueError, match="channel scope"):
            await (
                bot.tree.get_command("persona")
                .get_command("add")
                .callback(interaction(), "A rule", "server", "10")
            )
        assert not store.get("guild:1", PersonaStore.key)
    finally:
        await bot.close()
