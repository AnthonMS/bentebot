import asyncio
import io
import json
from dataclasses import replace

import httpx
import pytest
from aiohttp import web
from PIL import Image

from bentebot.config import Settings
from bentebot.domain import Audience, ResponseRecord
from bentebot.providers import CHAT_OBSERVATION, OpenAIChat
from bentebot.screen import ScreenService, ScreenSnapshot
from bentebot.vision import ScreenAwareness, visual_reason


class Frames:
    def __init__(self):
        self.active = True
        self.queries = 0
        self.latest = self.frame(100, "current")
        self.old = self.frame(98, "before")

    def frame(self, timestamp, suffix):
        return ScreenSnapshot(
            1,
            2,
            3,
            frozenset({1}),
            "session",
            timestamp,
            timestamp,
            "data:image/jpeg;base64," + suffix,
            1280,
            720,
        )

    async def snapshots(self, *_):
        self.queries += 1
        return (self.latest,) if self.active else ()

    async def history(self, *_args, **_kwargs):
        return (self.old,) if self.active else ()

    async def sources_active(self, _):
        return self.active


class Policy:
    def __init__(self):
        self.allowed = True

    async def allows(self, source, audience):
        return self.allowed and source.audience_ids == tuple(sorted(audience.user_ids))

    async def allows_all(self, sources, audience):
        return all([await self.allows(source, audience) for source in sources])


def awareness(**kwargs):
    seen = []
    frames, policy = Frames(), Policy()

    async def observe(messages, model):
        seen.append((messages, model, CHAT_OBSERVATION.get()))
        yield "The terminal shows a Docker swap-limit warning."

    settings = Settings(screen_enabled=True, vision_model="cloud-image", vision_mode="observer", **kwargs)
    return ScreenAwareness(settings, frames, policy, observe, clock=lambda: 100), frames, policy, seen


def assert_unavailable(result):
    messages, sources, model = result
    assert not sources and model is None
    assert "no fresh visual evidence" in messages[0]["content"]
    assert "Earlier descriptions may now be stale" in messages[0]["content"]
    assert "data:image" not in repr(result)


@pytest.mark.parametrize(
    "text", ["oh my god", "hold da kæft", "What does the warning mean?", "The code is broken", "fejl i koden"]
)
def test_visual_relevance_without_explicit_look_command(text):
    assert visual_reason(text)


@pytest.mark.parametrize(
    "text",
    [
        "What do you see?",
        "What can you see now?",
        "Can you see this?",
        "Can you see it clearly?",
        "Could you really see that?",
        "Bentebot, what do you see?",
        "Hey, can you see this, please?",
        "Hvad kan du se?",
        "Hvad kan du se her?",
        "Hvad ser du?",
        "Kan du se det?",
        "Kan du se det her?",
    ],
)
async def test_explicit_visual_questions_work_without_screen_keyword_or_prior_focus(text):
    assert visual_reason(text) == "visual_question"
    service, frames, _, seen = awareness()
    audience = Audience(frozenset({1}), 2)
    result = await service.context(3, audience, audience, text)
    assert result[1] and frames.queries == 1 and len(seen) == 1


@pytest.mark.parametrize(
    "text",
    [
        "I can see your point",
        "Can you see this happening in the future?",
        "What do you see as the main benefit?",
        "Yesterday I asked what do you see",
        "Jeg kan se hvad du mener",
        "Kan du se det fra mit perspektiv?",
    ],
)
def test_general_uses_of_see_do_not_trigger_visual_processing(text):
    assert visual_reason(text) is None


@pytest.mark.parametrize("text", ["Thanks hehe", "What is the capital of Sweden?", "Hello there"])
async def test_unrelated_chat_has_no_frame_access_or_cloud_request(text):
    service, frames, _, seen = awareness()
    assert await service.context(3, Audience(frozenset({1}), 2), Audience(frozenset({1}), 2), text) == (
        [],
        (),
        None,
    )
    assert seen == [] and frames.queries == 0


async def test_observer_keeps_normal_chat_model_and_minimal_context():
    service, _, _, seen = awareness()
    audience = Audience(frozenset({1}), 2)
    record = ResponseRecord()
    token = CHAT_OBSERVATION.set(record)
    try:
        messages, sources, override = await service.context(
            3, audience, audience, "What does this warning mean?"
        )
    finally:
        CHAT_OBSERVATION.reset(token)
    assert override is None and sources[0].session_id == "session"
    assert "swap-limit" in messages[0]["content"]
    assert seen[0][1] == "cloud-image" and seen[0][2] is None
    assert len(seen[0][0]) == 2
    assert record.chat_metadata["vision_frames"] == 1
    assert "vision_analysis_ms" in record.timings


async def test_reaction_includes_pre_speech_frame_not_just_current_image():
    service, _, _, seen = awareness()
    audience = Audience(frozenset({1}), 2)
    await service.context(3, audience, audience, "Oh my god!", at=99)
    content = seen[0][0][-1]["content"]
    images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert images == ["data:image/jpeg;base64,before", "data:image/jpeg;base64,current"]


async def test_private_screen_not_sent_to_broader_destination():
    service, _, _, seen = awareness()
    _, sources, _ = await service.context(
        3, Audience(frozenset({1}), 2), Audience(frozenset({1, 4}), 2), "Read the code"
    )
    assert not sources and not seen


async def test_revoked_screen_result_is_discarded():
    service, frames, _, _ = awareness()

    async def revoke(_messages, _model):
        frames.active = False
        yield "Private screen content"

    service.observe = revoke
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert service.last_reason == "revoked_during_analysis"


async def test_observer_budget_never_changes_to_paid_fallback():
    service, _, _, seen = awareness(vision_requests_per_hour=1)
    audience = Audience(frozenset({1}), 2)
    await service.context(3, audience, audience, "Read the code")
    assert_unavailable(await service.context(3, audience, audience, "Read the warning"))
    assert len(seen) == 1 and service.describe()["vision_limited"] == 1


async def test_observer_failures_do_not_break_conversation_or_log_content(caplog):
    service, _, _, _ = awareness()

    async def fail(_messages, _model):
        raise RuntimeError("SECRET_IMAGE_CONTENT")
        yield  # pragma: no cover

    service.observe = fail
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the warning"))
    assert "RuntimeError" in caplog.text and "SECRET_IMAGE_CONTENT" not in caplog.text


async def test_cancelled_observation_is_not_swallowed():
    service, _, _, _ = awareness()

    async def cancelled(_messages, _model):
        raise asyncio.CancelledError
        yield  # pragma: no cover

    service.observe = cancelled
    audience = Audience(frozenset({1}), 2)
    with pytest.raises(asyncio.CancelledError):
        await service.context(3, audience, audience, "Read the code")


async def test_direct_mode_backwards_compatible_vision_model_override():
    service, _, _, seen = awareness()
    service.settings = Settings(screen_enabled=True, vision_model="local-vision")
    audience = Audience(frozenset({1}), 2)
    messages, sources, model = await service.context(3, audience, audience, "What is on the screen?")
    assert model == "local-vision" and messages[0]["content"][1]["type"] == "text"
    assert sources and not seen


@pytest.mark.parametrize("query", ["How do I fix it?", "Hvad betyder det?", "Why does this look wrong?"])
def test_contextual_screen_followups_do_not_need_another_look_command(query):
    assert visual_reason(query, focused=True) == "followup"
    assert visual_reason(query, focused=False) is None


async def test_followup_does_not_reuse_focus_from_a_previous_stream_session():
    service, frames, _, seen = awareness()
    audience = Audience(frozenset({1}), 2)
    await service.context(3, audience, audience, "Read the code")
    frames.latest = replace(frames.latest, session_id="new-stream")
    assert_unavailable(await service.context(3, audience, audience, "How do I fix it?"))
    assert len(seen) == 1 and service.last_reason == "focus_session_changed"


async def test_clear_discards_an_inflight_analysis_even_when_lease_still_looks_active():
    service, _, _, _ = awareness()

    async def clear(_messages, _model):
        service.clear(2)
        yield "Old private screen content"

    service.observe = clear
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert not service._focus and service.last_reason == "revoked_during_analysis"


async def test_clear_during_final_permission_await_discards_observer_result():
    service, _, policy, _ = awareness()

    async def clear(sources, audience):
        service.clear(2)
        await asyncio.sleep(0)
        return True

    policy.allows_all = clear
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert service.last_reason == "revoked_during_analysis"


async def test_direct_mode_clear_during_source_check_never_returns_pixels():
    service, frames, _, seen = awareness()
    service.settings = replace(service.settings, vision_mode="direct")

    async def clear(sources):
        service.clear(2)
        await asyncio.sleep(0)
        return True

    frames.sources_active = clear
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert seen == [] and not service._calls


async def test_concurrent_observer_turn_is_not_queued_and_reports_uncertainty():
    service, _, _, _ = awareness()
    started, finish = asyncio.Event(), asyncio.Event()

    async def observe(_messages, _model):
        started.set()
        await finish.wait()
        yield "The code has a syntax error."

    service.observe = observe
    audience = Audience(frozenset({1}), 2)
    first_record, second_record = ResponseRecord(), ResponseRecord()
    token = CHAT_OBSERVATION.set(first_record)
    try:
        first = asyncio.create_task(service.context(3, audience, audience, "Read the code"))
    finally:
        CHAT_OBSERVATION.reset(token)
    await started.wait()
    token = CHAT_OBSERVATION.set(second_record)
    try:
        assert_unavailable(await service.context(3, audience, audience, "Explain the warning"))
    finally:
        CHAT_OBSERVATION.reset(token)
    finish.set()
    assert (await first)[1]
    assert first_record.chat_metadata["vision_outcome"] == "observer_ok"
    assert second_record.chat_metadata["vision_outcome"] == "observer_busy"
    assert len(service._calls) == 1


async def test_timeout_cleans_up_stream_and_leaves_a_natural_uncertainty_cue():
    service, _, _, _ = awareness()
    service.settings = replace(service.settings, vision_timeout=0.01)
    closed = asyncio.Event()

    async def stalled(_messages, _model):
        try:
            await asyncio.Event().wait()
            yield "never"
        finally:
            closed.set()

    service.observe = stalled
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert closed.is_set() and not service._lock.locked()
    assert service.last_reason == "observer_timeout"


async def test_direct_mode_obeys_the_same_hourly_budget():
    service, _, _, seen = awareness(vision_requests_per_hour=1)
    service.settings = replace(service.settings, vision_mode="direct")
    audience = Audience(frozenset({1}), 2)
    assert (await service.context(3, audience, audience, "Read the code"))[2] == "cloud-image"
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert service.last_reason == "hourly_limit" and seen == []


async def test_budget_window_uses_request_time_after_permission_awaits():
    service, frames, _, seen = awareness(vision_requests_per_hour=1)
    state = {"now": 100.0}
    service.clock = lambda: state["now"]
    service._calls.append(100)
    original = frames.snapshots

    async def delayed(*args):
        state["now"] = 3700
        frames.latest = replace(frames.latest, received_at=3700)
        return await original(*args)

    frames.snapshots = delayed
    audience = Audience(frozenset({1}), 2)
    assert (await service.context(3, audience, audience, "Read the code"))[1]
    assert list(service._calls) == [3700] and len(seen) == 1


async def test_frame_that_expires_during_observation_is_not_presented_as_current():
    service, _, _, _ = awareness()
    state = {"now": 100.0}
    service.clock = lambda: state["now"]

    async def slow(_messages, _model):
        state["now"] = 111
        yield "Old scene content"

    service.observe = slow
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert service.last_reason == "evidence_expired" and not service._focus


async def test_stale_frame_after_permission_check_does_not_spend_quota():
    service, frames, _, seen = awareness()
    state = {"now": 100.0}
    service.clock = lambda: state["now"]

    async def slow(sources):
        state["now"] = 111
        return True

    frames.sources_active = slow
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Read the code"))
    assert seen == [] and not service._calls


async def test_reaction_with_identical_images_sends_one_pre_reaction_sample():
    service, frames, _, seen = awareness()
    frames.latest = replace(frames.latest, fingerprint="same-content")
    frames.old = replace(frames.old, fingerprint="same-content", data_url=frames.latest.data_url)
    audience = Audience(frozenset({1}), 2)
    await service.context(3, audience, audience, "Oh my god!", at=99)
    content = seen[0][0][1]["content"]
    assert len([part for part in content if part["type"] == "image_url"]) == 1
    assert any("2.0s before" in part.get("text", "") for part in content)


async def test_missing_pre_reaction_sample_is_explicit_in_private_observer_context():
    service, frames, _, seen = awareness()
    frames.old = replace(frames.old, received_at=99.5)
    audience = Audience(frozenset({1}), 2)
    await service.context(3, audience, audience, "Oh my god!", at=99)
    assert "No sample from before the reaction" in repr(seen[0][0])
    assert "before" not in [
        part["image_url"]["url"].rsplit(",", 1)[-1]
        for part in seen[0][0][1]["content"]
        if part["type"] == "image_url"
    ]


@pytest.mark.parametrize("reference", [float("nan"), float("inf")])
async def test_nonfinite_reaction_reference_does_not_access_frames(reference):
    service, frames, _, seen = awareness()
    audience = Audience(frozenset({1}), 2)
    assert_unavailable(await service.context(3, audience, audience, "Oh my god!", at=reference))
    assert frames.queries == 0 and seen == []


async def test_real_ingress_history_and_separate_http_observer(unused_tcp_port):
    state = {"now": 100.0, "allowed": True}
    requests = []

    async def validate(binding):
        return state["allowed"]

    screens = ScreenService(validate, clock=lambda: state["now"])
    app = web.Application()
    screens.install(app)

    async def observe(request):
        body = await request.json()
        requests.append(body)
        assert body["model"] == "image-observer" and body["max_tokens"] == 256
        assert len(body["messages"]) == 2
        images = [item for item in body["messages"][1]["content"] if item["type"] == "image_url"]
        assert len(images) == 2 and all(
            item["image_url"]["url"].startswith("data:image/jpeg;") for item in images
        )
        data = {"choices": [{"delta": {"content": "A red scene preceded the current blue scene."}}]}
        return web.Response(
            text="data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n", content_type="text/event-stream"
        )

    app.router.add_post("/v1/chat/completions", observe)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    settings = Settings(
        screen_enabled=True,
        vision_model="image-observer",
        vision_mode="observer",
        chat_base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
        chat_max_tokens=256,
        web_tools_enabled=False,
    )
    chat = OpenAIChat(settings)
    try:
        service = ScreenAwareness(settings, screens, Policy(), chat.stream, clock=lambda: state["now"])
        audience = Audience(frozenset({1}), 2)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{unused_tcp_port}") as client:
            grant = screens.issue_pair(1, 2, 3, audience.user_ids)
            token = (await client.post("/screen/pair", json={"code": grant.code})).json()["token"]
            for sequence, colour in enumerate(("red", "blue"), 1):
                state["now"] = 99.0 + sequence
                output = io.BytesIO()
                Image.new("RGB", (96, 64), colour).save(output, format="JPEG")
                response = await client.post(
                    "/screen/frame",
                    content=output.getvalue(),
                    headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "image/jpeg",
                        "X-Bentebot-Sequence": str(sequence),
                    },
                )
                assert response.status_code == 200
            record = ResponseRecord()
            observation = CHAT_OBSERVATION.set(record)
            try:
                result = await service.context(3, audience, audience, "Oh my god!", at=100.5)
            finally:
                CHAT_OBSERVATION.reset(observation)
            assert result[2] is None and "red scene" in result[0][0]["content"]
            assert result[1][0].kind == "screen" and record.chat_metadata["vision_frames"] == 2
            assert "upstream_host" not in record.chat_metadata
            # Casual speech never starts the observer, despite an active stream.
            assert await service.context(3, audience, audience, "Haha, thanks") == ([], (), None)
            assert len(requests) == 1
    finally:
        await chat.close()
        await runner.cleanup()
