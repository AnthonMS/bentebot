import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.native_stream import NativeStreamProbe


class Gateway:
    def __init__(self, packets=()):
        self.listeners = {}
        self.packets = packets
        self.ws = SimpleNamespace(send_as_json=AsyncMock(side_effect=self.send))

    def add_listener(self, callback, name):
        self.listeners[name] = callback

    def remove_listener(self, callback, name):
        assert self.listeners.pop(name) is callback

    async def send(self, payload):
        if payload["op"] == 20:
            for packet in self.packets:
                await self.listeners["on_socket_raw_receive"](packet)


def dispatch(kind, **data):
    return json.dumps({"op": 0, "t": kind, "d": {"stream_key": "guild:1:2:3", **data}})


@pytest.mark.parametrize("reverse", [True, False])
async def test_gateway_allocation_both_orders_sanitizes_credentials_and_unsubscribes(reverse):
    packets = [
        dispatch("STREAM_CREATE", rtc_server_id="44"),
        dispatch("STREAM_SERVER_UPDATE", endpoint="secret.discord.media", token="SECRET_TOKEN"),
    ]
    gateway = Gateway(packets[::-1] if reverse else packets)
    probe = NativeStreamProbe(gateway)
    result = await probe.run(1, 2, 3)
    assert result.outcome == "gateway_access_observed"
    assert result.stream_created and result.server_allocated
    assert "SECRET" not in repr(result) + result.describe()
    assert "secret.discord.media" not in repr(result)
    assert not gateway.listeners
    assert gateway.ws.send_as_json.call_args_list[-1].args[0] == {
        "op": 19,
        "d": {"stream_key": "guild:1:2:3"},
    }
    with pytest.raises(ValueError, match="cooling down"):
        await probe.run(1, 2, 3)


@pytest.mark.parametrize(
    "reason,outcome", [("unauthorized", "denied"), ("stream_ended", "ended"), ("SECRET", "ended")]
)
async def test_rejection_is_bounded_and_unknown_reasons_are_not_echoed(reason, outcome):
    gateway = Gateway([dispatch("STREAM_DELETE", reason=reason)])
    result = await NativeStreamProbe(gateway).run(1, 2, 3)
    assert result.outcome == outcome
    assert "SECRET" not in result.describe()
    assert not gateway.listeners


async def test_missing_null_or_unrelated_dispatches_are_inconclusive():
    gateway = Gateway(
        [
            "not json",
            "[]",
            "null",
            dispatch("STREAM_CREATE", rtc_server_id="44"),
            dispatch("STREAM_SERVER_UPDATE", endpoint=None, token="SECRET"),
            dispatch("STREAM_SERVER_UPDATE", stream_key="guild:9:8:7", endpoint="host", token="SECRET"),
        ]
    )
    result = await NativeStreamProbe(gateway, timeout=0.01).run(1, 2, 3)
    assert result.outcome == "inconclusive" and result.reason == "timeout"
    assert result.stream_created and not result.server_allocated
    assert not gateway.listeners


async def test_cancellation_removes_listeners_and_stops_watching():
    gateway = Gateway()
    probe = NativeStreamProbe(gateway)
    task = asyncio.create_task(probe.run(1, 2, 3))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not gateway.listeners and not probe._busy
    assert gateway.ws.send_as_json.call_args.args[0]["op"] == 19


async def test_gateway_disconnect_is_inconclusive():
    gateway = Gateway()
    probe = NativeStreamProbe(gateway, cooldown=0)
    task = asyncio.create_task(probe.run(1, 2, 3))
    await asyncio.sleep(0)
    await gateway.listeners["on_disconnect"]()
    result = await task
    assert result.reason == "gateway_disconnected"
    assert not gateway.listeners
    assert [call.args[0]["op"] for call in gateway.ws.send_as_json.call_args_list] == [20]
    with pytest.raises(ValueError, match="disabled after a gateway failure"):
        await probe.run(1, 2, 3)
    assert gateway.ws.send_as_json.await_count == 1


@pytest.mark.parametrize("resumable", [False, True])
async def test_invalid_session_stops_probe_without_cleanup_or_retry(resumable):
    gateway = Gateway([json.dumps({"op": 9, "d": resumable})])
    probe = NativeStreamProbe(gateway, cooldown=0)
    result = await probe.run(1, 2, 3)
    assert result.outcome == "inconclusive"
    assert result.reason == "gateway_invalid_session"
    assert not result.stream_created and not result.server_allocated
    assert not gateway.listeners and not probe._busy
    assert [call.args[0]["op"] for call in gateway.ws.send_as_json.call_args_list] == [20]
    with pytest.raises(ValueError, match="disabled after a gateway failure"):
        await probe.run(1, 2, 3)
    assert gateway.ws.send_as_json.await_count == 1


@pytest.mark.parametrize(
    "admin,enabled,streaming,in_call",
    [
        (False, True, True, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, False),
        (True, True, True, True),
    ],
)
async def test_command_is_opt_in_admin_and_current_call_only(store, admin, enabled, streaming, in_call):
    bot = BenteBot(
        Settings(
            health_port=0,
            owner_ids=frozenset({100}) if admin else frozenset(),
            native_stream_probe_enabled=enabled,
        ),
        store,
    )
    user = SimpleNamespace(
        id=100,
        bot=False,
        guild_permissions=SimpleNamespace(manage_guild=False),
        voice=SimpleNamespace(channel=SimpleNamespace(id=10 if in_call else 20), self_stream=streaming),
    )
    interaction = SimpleNamespace(
        user=user,
        guild_id=1,
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    bot.voice_sessions[1] = (SimpleNamespace(source=SimpleNamespace(channel_id=10)), None, None)
    bot.native_stream_probe.run = AsyncMock(return_value=SimpleNamespace(describe=lambda: "safe result"))
    try:
        command = bot.tree.get_command("screen").get_command("probe")
        if admin and enabled and streaming and in_call:
            await command.callback(interaction)
            bot.native_stream_probe.run.assert_awaited_once_with(1, 10, 100)
            interaction.followup.send.assert_awaited_once_with("safe result", ephemeral=True)
        else:
            with pytest.raises((ValueError, discord.app_commands.CheckFailure)):
                await command.callback(interaction)
            bot.native_stream_probe.run.assert_not_awaited()
    finally:
        bot.voice_sessions.clear()
        await bot.close()
