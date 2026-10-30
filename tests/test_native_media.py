import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.native_stream import VoiceMediaAudit


def voice_socket():
    listeners = []
    connection = SimpleNamespace(
        add_socket_listener=listeners.append, remove_socket_listener=listeners.remove
    )
    return SimpleNamespace(is_connected=lambda: True, _connection=connection), listeners


async def test_passive_headers_preserve_normal_reader_and_do_not_retain_payloads():
    voice, listeners = voice_socket()
    normal_packets = []
    listeners.append(normal_packets.append)
    secret = b"PRIVATE_MEDIA_PAYLOAD"
    packets = [
        b"\x80\xf8" + bytes(10) + secret,  # Opus with RTP marker bit.
        b"\x80\x78" + bytes(10) + secret,
        b"\x80\xe5" + bytes(10) + secret,  # Non-audio candidate; not proof of video.
        b"\x80\xc8" + bytes(6) + secret,  # RTCP sender report header.
        b"\x83\x78" + bytes(10),  # Truncated CSRC list.
        b"\x80\xc8",  # Truncated RTCP header.
        b"\x00\x02" + bytes(72),  # UDP discovery, not RTP.
        b"",
    ]
    audit = VoiceMediaAudit(seconds=0.001)
    task = asyncio.create_task(audit.collect(voice))
    await asyncio.sleep(0)

    def receive():
        for packet in packets:
            for listener in list(listeners):
                listener(packet)

    thread = threading.Thread(target=receive)
    thread.start()
    thread.join()
    result = await task
    assert result.rtp_types == ((101, 1), (120, 2))
    assert result.rtcp_types == ((200, 1),)
    assert result.other_headers == 4
    assert normal_packets == packets and listeners == [normal_packets.append]
    assert secret.decode() not in repr(result) + result.describe()
    assert "do not prove screen access" in result.describe()
    assert not audit._busy


async def test_cancellation_removes_only_observer_and_releases_concurrency_guard():
    voice, listeners = voice_socket()

    def original(data):
        return None

    listeners.append(original)
    audit = VoiceMediaAudit()
    task = asyncio.create_task(audit.collect(voice))
    await asyncio.sleep(0)
    with pytest.raises(ValueError, match="already running"):
        await audit.collect(voice)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert listeners == [original] and not audit._busy


@pytest.mark.parametrize("seconds", [0, -1, 31])
def test_audit_is_bounded(seconds):
    with pytest.raises(ValueError):
        VoiceMediaAudit(seconds=seconds)


@pytest.mark.parametrize("connected", [False, True])
async def test_missing_or_disconnected_socket_is_rejected(connected):
    audit = VoiceMediaAudit()
    with pytest.raises(ValueError, match="not connected|does not support"):
        await audit.collect(SimpleNamespace(is_connected=lambda: connected))
    assert not audit._busy


@pytest.mark.parametrize("admin,in_call", [(False, True), (True, False), (True, True)])
async def test_command_is_admin_current_call_only_and_does_not_need_active_probe(store, admin, in_call):
    bot = BenteBot(Settings(health_port=0, owner_ids=frozenset({100}) if admin else frozenset()), store)
    voice = object()
    user = SimpleNamespace(
        id=100,
        guild_permissions=SimpleNamespace(manage_guild=False),
        voice=SimpleNamespace(channel=SimpleNamespace(id=10 if in_call else 20)),
    )
    interaction = SimpleNamespace(
        user=user,
        guild_id=1,
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    bot.voice_sessions[1] = (
        SimpleNamespace(source=SimpleNamespace(channel_id=10)),
        None,
        SimpleNamespace(client=voice),
    )
    bot.native_media_audit.collect = AsyncMock(return_value=SimpleNamespace(describe=lambda: "headers only"))
    try:
        command = bot.tree.get_command("screen").get_command("inspect")
        if admin and in_call:
            assert not bot.settings.native_stream_probe_enabled
            await command.callback(interaction)
            bot.native_media_audit.collect.assert_awaited_once_with(voice)
            interaction.followup.send.assert_awaited_once_with("headers only", ephemeral=True)
            assert not bot.voice_jobs
        else:
            with pytest.raises((ValueError, discord.app_commands.CheckFailure)):
                await command.callback(interaction)
            bot.native_media_audit.collect.assert_not_awaited()
    finally:
        bot.voice_sessions.clear()
        await bot.close()
