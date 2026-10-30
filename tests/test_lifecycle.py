import asyncio
from unittest.mock import AsyncMock

import httpx

from bentebot.config import Settings
from bentebot.discord_app import BenteBot


async def test_setup_and_health_endpoints_without_discord_login(store, unused_tcp_port):
    bot = BenteBot(Settings(guild_ids=frozenset({123}), health_port=unused_tcp_port), store)
    bot.tree.sync = AsyncMock(return_value=bot.tree.get_commands())
    try:
        await bot._async_setup_hook()
        await bot.setup_hook()
        assert bot.tree.sync.await_count == 2
        assert bot.tree.sync.await_args_list[0].kwargs == {}
        assert bot.tree.sync.call_args.kwargs["guild"].id == 123
        assert bot.tree.get_commands(guild=bot.tree.sync.call_args.kwargs["guild"]) == []
        assert bot.tree.get_command("voice") is not None
        async with httpx.AsyncClient() as client:
            url = f"http://127.0.0.1:{unused_tcp_port}"
            assert (await client.get(url + "/health/live")).status_code == 200
            assert (await client.get(url + "/health/ready")).status_code == 503
            bot._ready.set()
            bot.accepting = True
            assert (await client.get(url + "/health/ready")).status_code == 200
        assert bot.background and not bot.background.done()
    finally:
        await bot.close()
    assert bot.background.cancelled()
    assert bot.health is not None
    await asyncio.sleep(0)
