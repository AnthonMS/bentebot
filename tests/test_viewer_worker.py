import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot_viewer.lease import Lease, LeaseState
from bentebot_viewer.receiver import DiscordReceiver


def payload(**changes):
    return {
        "active": True,
        "lease_id": "a" * 32,
        "guild_id": "1",
        "channel_id": "10",
        "owner_id": "100",
        "bot_user_id": "999",
        "viewer_user_id": "900",
        "frame_token": "SECRET_FRAME_TOKEN" * 3,
        "lease_seconds": 3,
        **changes,
    }


def test_lease_expiry_renewal_and_job_change_revoke_previous_watch_authority():
    now = {"value": 100}
    state = LeaseState(lambda: now["value"])
    first = Lease.parse(payload(), viewer_id=900, now=100)
    assert not state.permits(first)
    state.current = first
    assert state.permits(first) and "SECRET" not in repr(first)
    now["value"] = 103
    assert not state.permits(first)
    state.current = Lease.parse(payload(), viewer_id=900, now=103)
    assert state.permits(first)
    state.current = replace(state.current, owner_id=200)
    assert not state.permits(first)
    state.current = None
    assert not state.permits(first)


@pytest.mark.parametrize(
    "changes",
    [
        {"viewer_user_id": "100"},
        {"lease_seconds": 60},
        {"lease_seconds": float("nan")},
        {"frame_token": "short"},
        {"bot_user_id": "100"},
        {"owner_id": "-1"},
    ],
)
def test_untrusted_lease_fields_fail_closed(changes):
    with pytest.raises(ValueError, match="Invalid viewer lease"):
        Lease.parse(payload(**changes), viewer_id=900, now=100)


async def test_bot_departure_during_voice_connect_prevents_stream_watch_and_disconnects(monkeypatch):
    state = LeaseState(lambda: 100)
    state.current = job = Lease.parse(payload(), viewer_id=900, now=100)
    channel = SimpleNamespace()
    bot = SimpleNamespace(voice=SimpleNamespace(channel=channel))
    owner = SimpleNamespace(voice=SimpleNamespace(channel=channel, self_stream=True))
    guild = SimpleNamespace(
        get_channel=lambda _: channel, get_member={999: bot, 100: owner, 900: object()}.get
    )
    channel.permissions_for = lambda _: SimpleNamespace(view_channel=True, connect=True)
    voice = SimpleNamespace(disconnect=AsyncMock())
    client = SimpleNamespace(is_ready=lambda: True, get_guild=lambda _: guild, voice_clients=[])

    async def connect(**kwargs):
        assert kwargs["self_mute"] and kwargs["self_deaf"]
        client.voice_clients.append(voice)
        bot.voice.channel = None
        return voice

    channel.connect = AsyncMock(side_effect=connect)
    native_voice = SimpleNamespace(get_voice_codecs=lambda: [], with_config=lambda **kwargs: object)
    monkeypatch.setitem(
        sys.modules,
        "discord.ext.native_voice",
        SimpleNamespace(AsyncQueueSink=object, StreamClient=object, VoiceClient=native_voice),
    )
    receiver = DiscordReceiver(client, state, AsyncMock(), "idle")
    await receiver.watch(job)
    channel.connect.assert_awaited_once()
    voice.disconnect.assert_awaited_once_with(force=True)


async def test_missing_bot_or_expired_lease_never_joins(monkeypatch):
    state = LeaseState(lambda: 104)
    job = Lease.parse(payload(), viewer_id=900, now=100)
    state.current = job
    client = SimpleNamespace(is_ready=lambda: True, voice_clients=[], get_guild=lambda _: None)
    monkeypatch.setitem(
        sys.modules,
        "discord.ext.native_voice",
        SimpleNamespace(AsyncQueueSink=object, StreamClient=object, VoiceClient=object),
    )
    receiver = DiscordReceiver(client, state, AsyncMock(), "idle")
    assert not receiver.permitted(job)
    await receiver.watch(job)
    assert not client.voice_clients


async def test_uncached_stream_is_requested_directly_instead_of_waiting_for_cache(monkeypatch):
    state = LeaseState(lambda: 100)
    state.current = job = Lease.parse(payload(), viewer_id=900, now=100)
    channel = SimpleNamespace(permissions_for=lambda _: SimpleNamespace(view_channel=True, connect=True))
    members = {
        999: SimpleNamespace(voice=SimpleNamespace(channel=channel)),
        100: SimpleNamespace(voice=SimpleNamespace(channel=channel, self_stream=True)),
        900: object(),
    }
    guild = SimpleNamespace(get_channel=lambda _: channel, get_member=members.get)
    stream = SimpleNamespace(
        listen=Mock(),
        enable_video_receive=AsyncMock(),
        ws=SimpleNamespace(media_sink_wants=AsyncMock()),
        is_connected=lambda: False,
        disconnect=AsyncMock(),
    )
    # There is deliberately no .streams cache attribute on this connection.
    voice = SimpleNamespace(watch_stream=AsyncMock(return_value=stream), disconnect=AsyncMock())
    client = SimpleNamespace(is_ready=lambda: True, get_guild=lambda _: guild, voice_clients=[voice])
    channel.connect = AsyncMock(return_value=voice)
    native_voice = SimpleNamespace(get_voice_codecs=lambda: [], with_config=lambda **kwargs: object)
    monkeypatch.setitem(
        sys.modules,
        "discord.ext.native_voice",
        SimpleNamespace(
            AsyncQueueSink=lambda **kwargs: SimpleNamespace(**kwargs),
            StreamClient=object,
            VoiceClient=native_voice,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "discord.stream",
        SimpleNamespace(StreamKey=SimpleNamespace(from_guild=lambda **kwargs: SimpleNamespace(**kwargs))),
    )
    receiver = DiscordReceiver(client, state, AsyncMock(), "idle")
    await receiver.watch(job)
    voice.watch_stream.assert_awaited_once()
    key = voice.watch_stream.call_args.args[0]
    assert (key.guild_id, key.channel_id, key.owner_id) == (1, 10, 100)
    assert receiver.stage == "receiving_video"
    stream.disconnect.assert_awaited_once_with(force=True)
    voice.disconnect.assert_awaited_once_with(force=True)


async def test_lease_revoked_during_codec_probe_prevents_join(monkeypatch):
    state = LeaseState(lambda: 100)
    state.current = job = Lease.parse(payload(), viewer_id=900, now=100)
    channel = SimpleNamespace(
        permissions_for=lambda _: SimpleNamespace(view_channel=True, connect=True), connect=AsyncMock()
    )
    members = {
        999: SimpleNamespace(voice=SimpleNamespace(channel=channel)),
        100: SimpleNamespace(voice=SimpleNamespace(channel=channel, self_stream=True)),
        900: object(),
    }
    guild = SimpleNamespace(get_channel=lambda _: channel, get_member=members.get)
    client = SimpleNamespace(is_ready=lambda: True, get_guild=lambda _: guild, voice_clients=[])

    def probe():
        state.current = None
        return []

    native_voice = SimpleNamespace(get_voice_codecs=probe, with_config=lambda **kwargs: object)
    monkeypatch.setitem(
        sys.modules,
        "discord.ext.native_voice",
        SimpleNamespace(AsyncQueueSink=object, StreamClient=object, VoiceClient=native_voice),
    )
    await DiscordReceiver(client, state, AsyncMock(), "idle").watch(job)
    channel.connect.assert_not_awaited()
