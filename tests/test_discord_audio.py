import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from discord.ext.voice_recv import gateway

from bentebot.discord_app import voice_recv_client
from bentebot.discord_audio import AudioReceiveClient, audio_gateway_hook
from bentebot.voice import VoiceReceiver


def socket():
    client = SimpleNamespace(
        _add_ssrc=Mock(), guild=SimpleNamespace(get_member=lambda _: None), dispatch=Mock()
    )
    return SimpleNamespace(
        _connection=SimpleNamespace(voice_client=client), READY=2, SESSION_DESCRIPTION=4, SPEAKING=5
    ), client


@pytest.mark.parametrize("legacy_ssrc", [False, True])
async def test_optional_video_metadata_crashes_upstream_but_audio_adapter_keeps_mapping(legacy_ssrc):
    ws, client = socket()
    data = {
        "user_id": "123",
        "audio_ssrc": 37622,
        "streams": [
            {"type": "video", "rid": "100", "quality": 100, "active": False, "ssrc": 77, "rtx_ssrc": 78}
        ],
    }
    if legacy_ssrc:
        data["video_ssrc"] = 77
    event = {"op": 12, "d": data}
    # Reproduce the actual pinned dependency's missing video_ssrc/max_framerate
    # failure. No crypto, frame payload, or Discord account is involved.
    with pytest.raises(KeyError):
        await gateway.hook(ws, event)
    client._add_ssrc.reset_mock()
    await audio_gateway_hook(ws, event)
    client._add_ssrc.assert_called_once_with(123, 37622)


async def test_zero_audio_ssrc_does_not_replace_valid_speaker_mapping():
    ws, client = socket()
    await audio_gateway_hook(ws, {"op": 12, "d": {"user_id": "123", "audio_ssrc": 0, "streams": []}})
    client._add_ssrc.assert_not_called()


@pytest.mark.parametrize("data", [None, {}, {"user_id": "123", "audio_ssrc": -1}])
async def test_malformed_optional_peer_metadata_does_not_kill_voice_poller(data):
    ws, client = socket()
    await audio_gateway_hook(ws, {"op": 12, "d": data})
    client._add_ssrc.assert_not_called()


async def test_other_gateway_events_keep_original_handler(monkeypatch):
    ws, _ = socket()
    original = AsyncMock()
    monkeypatch.setattr(gateway, "hook", original)
    for opcode in (2, 4, 5, 6, 11, 13, 21, 22, 24):
        event = {"op": opcode, "d": {}}
        await audio_gateway_hook(ws, event)
        original.assert_awaited_with(ws, event)
    assert voice_recv_client() is AudioReceiveClient


async def test_failed_control_reader_is_visible_without_logging_exception_contents(caplog):
    async def fail():
        raise RuntimeError("PRIVATE_SESSION_DATA")

    runner = asyncio.create_task(fail())
    client = SimpleNamespace(
        listen=Mock(), stop_listening=Mock(), _connection=SimpleNamespace(_runner=runner)
    )
    receiver = VoiceReceiver(client, SimpleNamespace(metrics=Mock()), None, AsyncMock())
    receiver.start()
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert "RuntimeError" in receiver.last_error
        assert "PRIVATE_SESSION_DATA" not in receiver.last_error + caplog.text
    finally:
        await receiver.close()
