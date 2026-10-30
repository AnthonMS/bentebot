import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import discord
import httpx
import pytest
from aiohttp import web
from PIL import Image

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience, Source
from bentebot.native_viewer import NativeViewerBroker, ViewerTarget, control_key
from bentebot.screen import ScreenBinding, ScreenService

KEY = "test-control-key-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + KEY, "X-Bentebot-Viewer-Id": "900"}


@pytest.fixture
async def broker(unused_tcp_port):
    clock = {"now": 100.0}
    state = {"target": None, "allowed": True}

    async def desired():
        return state["target"]

    async def validate(binding):
        return state["allowed"] and state["target"] is not None

    service = ScreenService(validate, clock=lambda: clock["now"])
    control = NativeViewerBroker(service, desired, viewer_id=900, key=KEY, clock=lambda: clock["now"])
    app = web.Application()
    service.install(app)
    control.install(app)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{unused_tcp_port}") as client:
        yield control, service, client, state, clock
    await runner.cleanup()


def target(audience=frozenset({100})):
    return ViewerTarget(1, 10, 100, 999, audience)


async def lease(client):
    response = await client.get("/native-viewer/lease", headers=HEADERS)
    assert response.status_code == 200
    return response.json()


async def test_control_requires_secret_and_exact_viewer_identity_and_waits_when_bot_absent(broker):
    _, service, client, _, _ = broker
    assert (await client.get("/native-viewer/lease")).status_code == 401
    wrong = {**HEADERS, "X-Bentebot-Viewer-Id": "100"}
    assert (await client.get("/native-viewer/lease", headers=wrong)).status_code == 403
    assert (await lease(client)) == {"active": False}
    assert not service.sessions


async def test_native_frame_uses_bound_audience_and_is_revoked_when_bot_stops(broker):
    _, service, client, state, _ = broker
    state["target"] = target()
    job = await lease(client)
    assert job["active"] and job["owner_id"] == "100" and job["viewer_user_id"] == "900"
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, format="JPEG")
    upload_headers = {
        "Authorization": "Bearer " + job["frame_token"],
        "Content-Type": "image/jpeg",
        "X-Bentebot-Sequence": "1",
    }
    response = await client.post("/screen/frame", content=image.getvalue(), headers=upload_headers)
    assert response.status_code == 200
    assert await service.snapshots(10, Audience(frozenset({100}), 1))
    assert not await service.snapshots(10, Audience(frozenset({100, 200}), 1))
    assert job["frame_token"] not in repr(service.sessions)
    state["target"] = None
    assert (await lease(client)) == {"active": False}
    assert not service.sessions
    assert (
        await client.post("/screen/frame", content=image.getvalue(), headers=upload_headers)
    ).status_code == 401


async def test_renewal_keeps_stream_identity_but_audience_changes_and_expiry_replace_it(broker):
    _, service, client, state, clock = broker
    state["target"] = target()
    first = await lease(client)
    clock["now"] += 2
    assert (await lease(client))["lease_id"] == first["lease_id"]
    state["target"] = target(frozenset({100, 200}))
    second = await lease(client)
    assert second["lease_id"] != first["lease_id"]
    assert not service.renew_native(first["frame_token"])
    clock["now"] += 6
    third = await lease(client)
    assert third["lease_id"] != second["lease_id"]


async def test_worker_reports_are_bounded_and_do_not_echo_arbitrary_content(broker):
    control, _, client, state, _ = broker
    state["target"] = target()
    job = await lease(client)
    report = {"state": "watching", "lease_id": job["lease_id"], "decoded_frames": 5}
    assert (await client.post("/native-viewer/report", json=report, headers=HEADERS)).status_code == 200
    assert control.describe()["viewer_decoded_frames"] == 5
    bad = {**report, "state": "SECRET ARBITRARY CONTENT"}
    response = await client.post("/native-viewer/report", json=bad, headers=HEADERS)
    assert response.status_code == 400 and "SECRET" not in response.text
    control.close()
    assert not (await lease(client))["active"]


async def test_report_distinguishes_stream_and_sample_dimensions_and_clears_on_revoke(broker):
    control, _, client, state, _ = broker
    state["target"] = target()
    job = await lease(client)
    report = {
        "state": "watching",
        "lease_id": job["lease_id"],
        "source_size": [2560, 720],
        "sample_size": [1600, 450],
    }
    assert (await client.post("/native-viewer/report", json=report, headers=HEADERS)).status_code == 200
    assert control.describe()["viewer_stream_size"] == "2560x720"
    assert control.describe()["viewer_sample_size"] == "1600x450"
    state["target"] = None
    await lease(client)
    assert control.describe()["viewer_stream_size"] is None
    assert (await client.post("/native-viewer/report", json=report, headers=HEADERS)).status_code == 400


@pytest.mark.parametrize("size", [[0, 720], [9000, 720], [True, 720], [4096, 4096], [720], "2560x720"])
async def test_report_rejects_invalid_dimensions(broker, size):
    _, _, client, state, _ = broker
    state["target"] = target()
    job = await lease(client)
    report = {"state": "watching", "lease_id": job["lease_id"], "source_size": size}
    assert (await client.post("/native-viewer/report", json=report, headers=HEADERS)).status_code == 400


def setup_bot(store):
    bot = BenteBot(
        Settings(
            health_port=0,
            guild_ids=frozenset({1}),
            screen_enabled=True,
            native_viewer_enabled=True,
            native_viewer_id=900,
        ),
        store,
    )
    bot._connection.user = SimpleNamespace(id=999)
    channel = MagicMock(spec=discord.VoiceChannel)
    channel.id = 10
    members = {
        id_: SimpleNamespace(
            id=id_, bot=id_ == 999, voice=SimpleNamespace(channel=channel, self_stream=id_ == 100)
        )
        for id_ in (100, 200, 900, 999)
    }
    guild = SimpleNamespace(
        id=1,
        members=list(members.values()),
        chunked=True,
        get_member=members.get,
        get_channel=lambda id_: channel if id_ == 10 else None,
        get_channel_or_thread=lambda id_: channel if id_ == 10 else None,
    )
    channel.members, channel.guild = list(members.values()), guild
    channel.permissions_for.side_effect = lambda member: SimpleNamespace(
        view_channel=True, connect=True, read_message_history=member.id != 900
    )
    bot.get_guild = lambda id_: guild if id_ == 1 else None
    bot.is_ready = lambda: True
    bot.accepting = True
    conversation = SimpleNamespace(paused=False, source=Source("channel", 10, 1))
    voice = SimpleNamespace(channel=channel, is_connected=lambda: True)
    bot.voice_sessions[1] = (conversation, None, SimpleNamespace(client=voice))
    return bot, channel, members, conversation


async def test_service_account_is_excluded_from_audience_and_does_not_invalidate_human_context(store):
    bot, channel, members, _ = setup_bot(store)
    try:
        assert (await bot.audience_for(channel, voice=True)).user_ids == frozenset({100, 200})
        assert (await bot.audience_for(channel)).user_ids == frozenset({100, 200})
        bot.invalidate = Mock()
        await bot.on_voice_state_update(
            members[900], SimpleNamespace(channel=None), SimpleNamespace(channel=channel)
        )
        bot.invalidate.assert_not_called()
        result = await bot._desired_native_view()
        assert result and result.channel_id == 10 and result.owner_id == 100
        binding = ScreenBinding(100, 1, 10, result.audience_ids, "session", "native")
        assert await bot._screen_binding_valid(binding)
        members[100].voice.self_stream = False
        assert not await bot._screen_binding_valid(binding)
    finally:
        bot.voice_sessions.clear()
        await bot.close()


@pytest.mark.parametrize(
    "change",
    ["paused", "bot_left", "no_stream", "two_streams", "no_session", "viewer_denied", "owner_paused"],
)
async def test_no_viewer_lease_without_active_bot_and_exactly_one_stream(store, change):
    bot, channel, members, conversation = setup_bot(store)
    try:
        if change == "paused":
            conversation.paused = True
        elif change == "bot_left":
            channel.members = [member for member in channel.members if member.id != 999]
        elif change == "no_stream":
            members[100].voice.self_stream = False
        elif change == "two_streams":
            members[200].voice.self_stream = True
        elif change == "viewer_denied":
            channel.permissions_for.side_effect = lambda member: SimpleNamespace(
                view_channel=member.id != 900, connect=True, read_message_history=True
            )
        elif change == "owner_paused":
            bot.native_paused_owners.add((1, 100))
        else:
            bot.voice_sessions.clear()
        assert await bot._desired_native_view() is None
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_native_stop_blocks_reenrollment_and_start_resumes_without_companion_or_vision_alias(store):
    bot, channel, members, conversation = setup_bot(store)
    conversation.invalidate = Mock()
    interaction = SimpleNamespace(
        guild_id=1, user=members[100], response=SimpleNamespace(send_message=AsyncMock())
    )
    try:
        bot.screens.open_native(100, 1, 10, frozenset({100, 200}))
        await bot.tree.get_command("screen").get_command("stop").callback(interaction)
        assert not bot.screens.sessions and await bot._desired_native_view() is None
        await bot.tree.get_command("screen").get_command("start").callback(interaction)
        assert await bot._desired_native_view() is not None
        assert not bot.screens.pending and not bot.settings.vision_model
        assert "no capture helper" in interaction.response.send_message.call_args.args[0]
    finally:
        bot.voice_sessions.clear()
        await bot.close()


def test_control_key_errors_never_echo_secret(tmp_path):
    file = tmp_path / "key"
    file.write_text("SHORT_SECRET", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        control_key(str(file))
    assert "SHORT_SECRET" not in str(error.value)
    file.write_text(KEY, encoding="utf-8")
    assert control_key(str(file)) == KEY
