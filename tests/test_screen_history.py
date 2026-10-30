"""Screen history follows real ingress, call permissions, retention and RAM limits."""

import asyncio
import io

import httpx
import pytest
from aiohttp import web
from PIL import Image

from bentebot.domain import Audience
from bentebot.screen import ScreenService


def jpeg(colour):
    output = io.BytesIO()
    Image.new("RGB", (96, 64), colour).save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
async def history_service(unused_tcp_port):
    state = {"now": 100.0, "allowed": True}

    async def validate(binding):
        return state["allowed"]

    service = ScreenService(validate, clock=lambda: state["now"])
    app = web.Application()
    service.install(app)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{unused_tcp_port}") as client:
        yield service, client, state
    await runner.cleanup()


async def pair(service, client, owner=100, audience=frozenset({100})):
    grant = service.issue_pair(owner, 1, 10, audience)
    response = await client.post("/screen/pair", json={"code": grant.code})
    assert response.status_code == 200
    return response.json()["token"]


async def upload(client, token, sequence=1, colour="blue"):
    response = await client.post(
        "/screen/frame",
        content=jpeg(colour),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "image/jpeg",
            "X-Bentebot-Sequence": str(sequence),
        },
    )
    assert response.status_code == 200
    return response.json()


async def test_history_preserves_pre_reaction_time_and_samples_interval(history_service):
    service, client, state = history_service
    token = await pair(service, client)
    for index, colour in enumerate(("red", "green", "blue", "yellow", "black")):
        state["now"] = 100.0 + index
        await upload(client, token, index + 1, colour)
    audience = Audience(frozenset({100}), 1)
    assert [frame.received_at for frame in await service.history(10, audience)] == [100, 104]
    before = await service.history(10, audience, before=102.5, lookback=2, limit=10)
    assert [frame.received_at for frame in before] == [101, 102]
    assert [frame.received_at for frame in await service.history(10, audience, limit=1)] == [104]
    # Latest snapshots retain their old per-owner behavior, independent of history.
    assert [frame.received_at for frame in await service.snapshots(10, audience)] == [104]


async def test_repeated_static_frame_has_stable_original_sample_times(history_service):
    service, client, state = history_service
    token = await pair(service, client)
    assert (await upload(client, token))["changed"]
    first = (await service.history(10, Audience(frozenset({100}), 1)))[0]
    state["now"] += 1
    assert not (await upload(client, token, 2))["changed"]
    frames = await service.history(10, Audience(frozenset({100}), 1), limit=10)
    assert [frame.received_at for frame in frames] == [100, 101]
    assert first.received_at == 100 and frames[0] is first
    assert frames[0].fingerprint == frames[1].fingerprint
    assert frames[0].data_url is frames[1].data_url
    assert frames[0].created_at <= frames[1].created_at


async def test_history_never_crosses_guild_call_audience_or_owner(history_service):
    service, client, _ = history_service
    token = await pair(service, client)
    await upload(client, token)
    assert await service.history(10, Audience(frozenset({100, 200}), 1)) == ()
    assert await service.history(10, Audience(frozenset({100}), 2)) == ()
    assert await service.history(20, Audience(frozenset({100}), 1)) == ()
    assert await service.history(10, Audience(frozenset({100}), 1), owner_id=200) == ()
    assert service.describe(100, 10)["fresh"]


async def test_history_expires_without_stopping_a_longer_session(history_service):
    service, client, state = history_service
    token = await pair(service, client)
    await upload(client, token)
    session = next(iter(service.sessions.values()))
    state["now"] += 11
    assert await service.history(10, Audience(frozenset({100}), 1), limit=10) == ()
    assert session.snapshot is None and not session.history
    assert service.buffered_bytes == 0 and service.sessions


async def test_shorter_history_does_not_shorten_existing_snapshot_ttl(history_service):
    service, client, state = history_service
    service.history_seconds = 2
    token = await pair(service, client)
    await upload(client, token)
    state["now"] += 3
    audience = Audience(frozenset({100}), 1)
    assert await service.history(10, audience) == ()
    assert len(await service.snapshots(10, audience)) == 1
    assert service.buffered_bytes > 0
    state["now"] += 8
    assert await service.snapshots(10, audience) == ()
    assert service.buffered_bytes == 0


async def test_frame_and_byte_budgets_keep_recent_samples_and_latest(history_service):
    service, client, state = history_service
    service.max_history_frames = 3
    token = await pair(service, client)
    for index in range(6):
        state["now"] = 100.0 + index
        await upload(client, token, index + 1)
    audience = Audience(frozenset({100}), 1)
    assert [frame.received_at for frame in await service.history(10, audience, limit=10)] == [103, 104, 105]
    session = next(iter(service.sessions.values()))
    encoded_bytes = len(session.snapshot.data_url)
    service.max_history_bytes = encoded_bytes * 2 + 1
    state["now"] += 1
    await upload(client, token, 7)
    assert len(session.history) == 2 and session.snapshot.received_at == 106
    assert service.buffered_bytes <= service.max_history_bytes
    status = service.describe(100, 10)
    assert status["history_frames"] == 2 and status["history_bytes"] == service.buffered_bytes
    assert status["history_oldest_age_seconds"] == 1
    assert "data:image" not in repr(status) and token not in repr(status)


async def test_global_byte_budget_evicts_oldest_across_sessions(history_service):
    service, client, state = history_service
    first = await pair(service, client)
    await upload(client, first)
    first_session = next(iter(service.sessions.values()))
    frame_bytes = service.buffered_bytes
    service.max_total_history_bytes = frame_bytes + 8
    state["now"] += 1
    second = await pair(service, client, owner=200, audience=frozenset({200}))
    await upload(client, second)
    assert service.buffered_bytes <= service.max_total_history_bytes
    assert first_session.snapshot is None and not first_session.history
    assert await service.snapshots(10, Audience(frozenset({100}), 1)) == ()
    assert len(await service.snapshots(10, Audience(frozenset({200}), 1))) == 1


@pytest.mark.parametrize("revoke", ["invalidate", "native", "close", "expiry", "permission", "http_stop"])
async def test_revocation_clears_owned_frame_references(history_service, revoke):
    service, client, state = history_service
    if revoke == "native":
        token, _ = service.open_native(100, 1, 10, frozenset({100}))
    else:
        token = await pair(service, client)
    await upload(client, token)
    session = next(iter(service.sessions.values()))
    assert session.history and session.snapshot
    if revoke == "invalidate":
        service.invalidate(owner_id=100)
    elif revoke == "native":
        service.revoke_native(token)
    elif revoke == "close":
        await service.close()
    elif revoke == "expiry":
        state["now"] += 3601
        service.expire()
    elif revoke == "permission":
        state["allowed"] = False
        assert await service.history(10, Audience(frozenset({100}), 1)) == ()
    else:
        response = await client.post("/screen/stop", headers={"Authorization": "Bearer " + token})
        assert response.status_code == 200
    assert not service.sessions and not session.history and session.snapshot is None
    assert service.buffered_bytes == 0


async def test_revocation_while_history_validator_yields_cannot_return_pixels(history_service):
    service, client, _ = history_service
    token = await pair(service, client)
    await upload(client, token)
    started, finish = asyncio.Event(), asyncio.Event()

    async def validate(binding):
        started.set()
        await finish.wait()
        return True

    service.validate = validate
    task = asyncio.create_task(service.history(10, Audience(frozenset({100}), 1)))
    await started.wait()
    service.invalidate(owner_id=100)
    finish.set()
    assert await task == ()


async def test_native_lease_expiry_during_history_validation_clears_pixels(history_service):
    service, client, state = history_service
    token, _ = service.open_native(100, 1, 10, frozenset({100}))
    await upload(client, token)
    session = next(iter(service.sessions.values()))

    async def validate(binding):
        state["now"] += 6
        await asyncio.sleep(0)
        return True

    service.validate = validate
    assert await service.history(10, Audience(frozenset({100}), 1)) == ()
    assert not session.history and session.snapshot is None and not service.sessions


async def test_frame_larger_than_entire_history_budget_is_rejected(history_service):
    service, client, _ = history_service
    service.max_total_history_bytes = 64
    token = await pair(service, client)
    response = await client.post(
        "/screen/frame",
        content=jpeg("blue"),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "image/jpeg",
            "X-Bentebot-Sequence": "1",
        },
    )
    assert response.status_code == 413
    session = next(iter(service.sessions.values()))
    assert not session.history and session.snapshot is None
    assert session.frames_received == 0 and service.buffered_bytes == 0


@pytest.mark.parametrize("method", ["history", "snapshots"])
async def test_later_permission_await_cannot_resurrect_an_earlier_frame(history_service, method):
    service, client, _ = history_service
    audience = Audience(frozenset({100, 200}), 1)
    first = await pair(service, client, audience=audience.user_ids)
    await upload(client, first)
    second = await pair(service, client, owner=200, audience=audience.user_ids)
    await upload(client, second)

    async def validate(binding):
        if binding.owner_id == 200:
            service.invalidate(owner_id=100)
            await asyncio.sleep(0)
        return True

    service.validate = validate
    frames = await getattr(service, method)(10, audience)
    assert frames and {frame.owner_id for frame in frames} == {200}


async def test_sources_rechecked_after_later_source_validation_revokes_first(history_service):
    service, client, _ = history_service
    audience = Audience(frozenset({100, 200}), 1)
    first = await pair(service, client, audience=audience.user_ids)
    await upload(client, first)
    second = await pair(service, client, owner=200, audience=audience.user_ids)
    await upload(client, second)
    frames = await service.snapshots(10, audience)
    sources = tuple(frame.source for frame in sorted(frames, key=lambda frame: frame.owner_id))

    async def validate(binding):
        if binding.owner_id == 200:
            service.invalidate(owner_id=100)
            await asyncio.sleep(0)
        return True

    service.validate = validate
    assert not await service.sources_active(sources)


@pytest.mark.parametrize(
    "kwargs",
    [{"lookback": 0}, {"lookback": float("inf")}, {"before": float("nan")}, {"limit": 0}, {"limit": True}],
)
async def test_history_rejects_unbounded_or_invalid_queries(history_service, kwargs):
    service, _, _ = history_service
    with pytest.raises(ValueError):
        await service.history(10, Audience(frozenset({100}), 1), **kwargs)
