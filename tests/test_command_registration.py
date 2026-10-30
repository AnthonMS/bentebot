from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.command_registration import command_paths, verify_registered_commands
from bentebot.config import Settings
from bentebot.discord_app import BenteBot


def test_command_paths_checks_subcommands_without_treating_options_as_commands():
    commands = [
        {
            "name": "tools",
            "options": [{"type": 1, "name": "python", "options": [{"type": 3, "name": "code"}]}],
        },
        {"name": "persona", "options": [{"type": 1, "name": "show"}]},
    ]
    assert command_paths(commands) == {"tools", "tools python", "persona", "persona show"}


@pytest.mark.parametrize("old", [True, False])
async def test_readback_detects_old_registration_and_performs_only_authenticated_gets(old):
    seen = []

    def transport(request):
        seen.append(request)
        assert request.method == "GET" and request.headers["Authorization"] == "Bot private-test-token"
        return httpx.Response(
            200,
            json={"id": "123"}
            if request.url.path.endswith("@me")
            else [{"name": "model"}]
            if old
            else [{"name": "persona"}, {"name": "tools"}],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        if old:
            with pytest.raises(ValueError, match="persona, tools"):
                await verify_registered_commands(
                    Settings(discord_token="private-test-token"), {"tools", "persona"}, client=client
                )
        else:
            result = await verify_registered_commands(
                Settings(discord_token="private-test-token"), {"tools", "persona"}, client=client
            )
            assert result["verified_entries"] == 2
        assert len(seen) == 2


async def test_startup_rejects_missing_current_commands_before_becoming_ready(store):
    bot = BenteBot(Settings(health_port=0), store)
    bot.tree.sync = AsyncMock(return_value=[SimpleNamespace(name="voice")])
    try:
        await bot._async_setup_hook()
        with pytest.raises(RuntimeError, match="current slash commands"):
            await bot.setup_hook()
        assert not bot.accepting and bot.background is None
    finally:
        await bot.close()
