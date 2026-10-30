import asyncio
import base64
import io
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest
from aiohttp import web
from PIL import Image

from bentebot.domain import Audience
from bentebot.screen import ScreenService
from scripts.screen_companion import make_handler, validate_target


def jpeg(size=(640, 360), colour="blue"):
    output = io.BytesIO()
    Image.new("RGB", size, colour).save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
async def screens(unused_tcp_port):
    allowed = {"value": True}
    clock = {"now": 100.0}

    async def validate(binding):
        return allowed["value"]

    service = ScreenService(validate, clock=lambda: clock["now"])
    app = web.Application()
    service.install(app)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{unused_tcp_port}") as client:
        yield service, client, allowed, clock
    await runner.cleanup()


async def pair(service, client, audience=frozenset({100})):
    grant = service.issue_pair(100, 1, 10, audience)
    response = await client.post("/screen/pair", json={"code": grant.code})
    assert response.status_code == 200
    return response.json()["token"], grant.code


async def upload(client, token, data=None, sequence=1):
    return await client.post(
        "/screen/frame",
        content=data or jpeg(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "image/jpeg",
            "X-Bentebot-Sequence": str(sequence),
        },
    )


async def test_one_time_pair_authentication_and_no_tokens_in_repr(screens):
    service, client, _, _ = screens
    token, code = await pair(service, client)
    assert (await client.post("/screen/pair", json={"code": code})).status_code == 401
    assert (await upload(client, "wrong-token")).status_code == 401
    assert code not in repr(service.pending) and token not in repr(service.sessions)
    assert (await upload(client, token)).status_code == 200


async def test_network_chunked_requests_are_fully_read(screens):
    service, client, _, _ = screens
    grant = service.issue_pair(100, 1, 10, frozenset({100}))

    async def chunks(data):
        for index in range(0, len(data), 7):
            yield data[index : index + 7]
            await asyncio.sleep(0.001)

    import json

    response = await client.post(
        "/screen/pair",
        content=chunks(json.dumps({"code": grant.code}).encode()),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    token = response.json()["token"]
    response = await client.post(
        "/screen/frame",
        content=chunks(jpeg()),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "image/jpeg",
            "X-Bentebot-Sequence": "1",
        },
    )
    assert response.status_code == 200
    assert service.describe(100, 10)["fresh"]


async def test_fresh_image_is_normalised_and_carries_owner_and_audience(screens):
    service, client, _, _ = screens
    token, _ = await pair(service, client)
    assert (await upload(client, token, jpeg((2400, 1600)))).status_code == 200
    snapshots = await service.snapshots(10, Audience(frozenset({100}), 1))
    assert len(snapshots) == 1
    frame = snapshots[0]
    assert frame.width <= 1600 and frame.height <= 1000
    assert service.describe(100, 10)["retained_frame_size"] == f"{frame.width}x{frame.height}"
    assert frame.owner_id == 100 and frame.channel_id == 10
    assert frame.source.kind == "screen" and frame.source.audience_ids == (100,)
    assert frame.session_id == frame.source.session_id
    assert "data:image/jpeg;base64," not in repr(frame)
    with Image.open(io.BytesIO(base64.b64decode(frame.data_url.split(",", 1)[1]))) as image:
        assert image.format == "JPEG" and image.getexif() == {}


async def test_frame_expiry_wipes_image_but_keeps_session_until_stopped(screens):
    service, client, _, clock = screens
    token, _ = await pair(service, client)
    await upload(client, token)
    clock["now"] += 11
    assert await service.snapshots(10, Audience(frozenset({100}), 1)) == ()
    assert next(iter(service.sessions.values())).snapshot is None
    assert (
        await client.get("/screen/status", headers={"Authorization": "Bearer " + token})
    ).status_code == 200


async def test_public_status_contains_only_metadata(screens):
    service, client, _, _ = screens
    token, code = await pair(service, client)
    await upload(client, token)
    status = service.describe(100, 10)
    assert status["paired"] and status["fresh"] and status["frames_received"] == 1
    assert token not in str(status) and code not in str(status) and "data:image" not in str(status)
    assert not service.describe(200, 10)["paired"]


async def test_changed_call_audience_clears_frames_and_session(screens):
    service, client, allowed, _ = screens
    token, _ = await pair(service, client)
    await upload(client, token)
    allowed["value"] = False  # Discord validator sees the actual membership change.
    assert await service.snapshots(10, Audience(frozenset({100, 200}), 1)) == ()
    assert not service.sessions
    assert (await upload(client, token, sequence=2)).status_code == 401


async def test_broader_text_audience_does_not_stop_a_still_valid_call_capture(screens):
    service, client, _, _ = screens
    token, _ = await pair(service, client)
    await upload(client, token)
    assert await service.snapshots(10, Audience(frozenset({100, 200}), 1, 10)) == ()
    assert service.describe(100, 10)["fresh"]
    assert len(await service.snapshots(10, Audience(frozenset({100}), 1))) == 1


async def test_callback_revocation_blocks_upload_and_snapshot(screens):
    service, client, allowed, clock = screens
    token, _ = await pair(service, client)
    await upload(client, token)
    clock["now"] += 1
    allowed["value"] = False
    assert (await upload(client, token, sequence=2)).status_code == 403
    assert not service.sessions


async def test_stop_and_invalidate_wipe_raw_frames(screens):
    service, client, _, _ = screens
    token, _ = await pair(service, client)
    await upload(client, token)
    assert (
        await client.post("/screen/stop", headers={"Authorization": "Bearer " + token})
    ).status_code == 200
    assert not service.sessions
    token, _ = await pair(service, client)
    await upload(client, token)
    assert service.invalidate(channel_id=10) == 1
    assert not service.sessions


async def test_session_and_pair_expiry(screens):
    service, client, _, clock = screens
    grant = service.issue_pair(100, 1, 10, frozenset({100}))
    clock["now"] += 121
    assert (await client.post("/screen/pair", json={"code": grant.code})).status_code == 401
    token, _ = await pair(service, client)
    clock["now"] += 3601
    assert (await upload(client, token)).status_code == 401


async def test_replay_rate_and_cross_origin_are_rejected(screens):
    service, client, _, clock = screens
    token, _ = await pair(service, client)
    assert (await upload(client, token)).status_code == 200
    assert (await upload(client, token, sequence=1)).status_code == 409
    assert (await upload(client, token, sequence=2)).status_code == 429
    clock["now"] += 1
    assert (await upload(client, token, sequence=2)).status_code == 200
    assert (
        await client.post("/screen/pair", json={"code": "wrong"}, headers={"Origin": "https://evil.example"})
    ).status_code == 403


async def test_corrupt_oversized_and_huge_pixel_images_are_rejected(screens):
    service, client, _, clock = screens
    token, _ = await pair(service, client)
    assert (await upload(client, token, b"not an image")).status_code == 400
    clock["now"] += 1
    assert (await upload(client, token, b"x" * 1_000_001)).status_code == 413
    assert (await upload(client, token, jpeg((3000, 2000)))).status_code == 400
    assert not next(iter(service.sessions.values())).snapshot


async def test_ui_has_no_external_scripts_and_no_capture_without_picker(screens):
    _, client, _, _ = screens
    response = await client.get("/screen")
    assert (
        response.status_code == 200
        and "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    )
    script = await client.get("/screen/app.js")
    assert "getDisplayMedia" in script.text and "audio: false" in script.text
    assert "localStorage" not in script.text


async def test_invalidation_while_pair_permission_check_runs_cannot_resurrect(screens):
    service, client, _, _ = screens
    started, finish = asyncio.Event(), asyncio.Event()

    async def validate(binding):
        started.set()
        await finish.wait()
        return True

    service.validate = validate
    grant = service.issue_pair(100, 1, 10, frozenset({100}))
    task = asyncio.create_task(client.post("/screen/pair", json={"code": grant.code}))
    await started.wait()
    service.invalidate(owner_id=100)
    finish.set()
    assert (await task).status_code == 401
    assert not service.sessions


async def test_invalidation_during_image_decode_cannot_resurrect_frames(screens, monkeypatch):
    from bentebot import screen

    service, client, _, _ = screens
    token, _ = await pair(service, client)
    started, finish = threading.Event(), threading.Event()
    original = screen._normalise_image

    def delayed(data, *, max_pixels):
        started.set()
        assert finish.wait(2)
        return original(data, max_pixels=max_pixels)

    monkeypatch.setattr(screen, "_normalise_image", delayed)
    task = asyncio.create_task(upload(client, token))
    assert await asyncio.to_thread(started.wait, 2)
    service.invalidate(owner_id=100)
    finish.set()
    assert (await task).status_code == 403
    assert not service.sessions


async def test_localhost_relay_pairs_and_transfers_frames_without_discord_tokens(screens):
    service, remote, _, _ = screens
    relay = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(remote.base_url).rstrip("/")))
    thread = threading.Thread(target=relay.serve_forever, daemon=True)
    thread.start()
    local_url = f"http://127.0.0.1:{relay.server_port}"
    try:
        async with httpx.AsyncClient(base_url=local_url) as client:
            page = await client.get("/screen")
            assert page.status_code == 200
            grant = service.issue_pair(100, 1, 10, frozenset({100}))
            response = await client.post(
                "/screen/pair", json={"code": grant.code}, headers={"Origin": local_url}
            )
            assert response.status_code == 200
            token = response.json()["token"]
            assert (await upload(client, token)).status_code == 200
            assert len(await service.snapshots(10, Audience(frozenset({100}), 1))) == 1
            assert (await client.get("/screen", headers={"Host": "evil.example"})).status_code == 403
            assert (
                await client.get("/screen/status", headers={"Origin": "https://evil.example"})
            ).status_code == 403
            assert (await client.get("/arbitrary-upstream-path")).status_code == 404
    finally:
        await asyncio.to_thread(relay.shutdown)
        relay.server_close()


@pytest.mark.parametrize(
    "target",
    [
        "file:///etc/passwd",
        "http://public.example",
        "http://user:pass@localhost",
        "http://localhost/path",
        "http://localhost?token=secret",
    ],
)
def test_companion_rejects_invalid_targets(target):
    with pytest.raises(ValueError):
        validate_target(target)


def test_companion_accepts_lan_and_https_origins():
    assert validate_target("http://192.168.50.10:8081/") == "http://192.168.50.10:8081"
    assert validate_target("https://bentebot.example") == "https://bentebot.example"
