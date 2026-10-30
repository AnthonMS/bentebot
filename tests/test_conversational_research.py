import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_native_web import fixture as native_fixture

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Event, Source, Transcript
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import OpenAIChat
from bentebot.research import ResearchJob, ResearchJobs
from bentebot.voice_control import spoken_control

VOICE = Source("channel", 10, 1)
CALLERS = Audience(frozenset({100}), 1)
READERS = Audience(frozenset({100, 200}), 1, 20)
URL = "https://example.com/python-release"


def prose(text):
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    )


def tool(name):
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": name,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(
                                            {"query": "Python release"}
                                            if name == "web_search"
                                            else {"url": URL}
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + "\n\ndata: [DONE]\n\n"
    )


def session_for(settings, store, chat, wav_bytes, *, policy=None, audience=None, play=None):
    played, written = [], []

    async def playback(pcm, text, progress):
        played.append(text)
        progress(text)
        if play:
            await play(text)

    async def write(text, origins, audience):
        written.append((text, origins))
        return True

    session = VoiceConversation(
        settings,
        store,
        Memory(store, policy or AccessPolicy(AsyncMock(return_value=True)), 10000),
        chat,
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        VOICE,
        999,
        audience or AsyncMock(return_value=CALLERS),
        playback,
        lambda: None,
        AsyncMock(),
        text_output=write,
        text_audience=AsyncMock(return_value=READERS),
        text_channel_id=20,
    )
    return session, played, written


@pytest.mark.parametrize("blocked", ["web_search", "read_webpage"])
@pytest.mark.parametrize("relevant", [True, False])
async def test_real_local_loop_survives_new_question_and_contextually_integrates(
    store, wav_bytes, blocked, relevant
):
    entered, release = asyncio.Event(), asyncio.Event()
    requests, integrated = [], []
    steps = [tool("web_search"), tool("read_webpage"), prose("OLD DRAFT must never replay.")]

    def handler(request):
        assert request.url.host == "gateway"
        body = json.loads(request.content)
        requests.append(body)
        last = body["messages"][-1]["content"]
        if '"completed_research"' in last:
            data = json.loads(last)["completed_research"]
            integrated.append(body)
            assert "favourite colour" in json.dumps(body)
            assert "Blue sounds good" in json.dumps(body)
            return httpx.Response(
                200,
                text=prose(
                    json.dumps(
                        {
                            "use": [data[0]["id"]] if relevant else [],
                            "answer": "About Python earlier, the verified release is ready."
                            if relevant
                            else "",
                        }
                    )
                ),
            )
        if last == "Tell me your favourite colour instead":
            return httpx.Response(200, text=prose("Blue sounds good."))
        return httpx.Response(200, text=steps.pop(0))

    async def execute(name, arguments):
        if name == blocked:
            entered.set()
            await release.wait()
        evidence = (
            {"kind": "search_results_not_read_articles", "results": [{"url": URL}]}
            if name == "web_search"
            else {"kind": "page_text", "url": URL, "text": "Verified Python version"}
        )
        return json.dumps({"untrusted_evidence": evidence})

    settings = Settings(
        chat_base_url="http://gateway/v1",
        chat_model="local",
        web_tools_enabled=True,
        web_search_url="http://search",
        web_tool_models="local",
    )
    store.save(Event(VOICE, 100, "Me", "Look up Python", kind="voice"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings, client)
        chat.web_tools.execute = AsyncMock(side_effect=execute)
        session, played, written = session_for(settings, store, chat, wav_bytes)
        session.latest_request = "Look up Python"
        session.task = asyncio.create_task(session._answer())
        try:
            async with asyncio.timeout(3):
                await entered.wait()
                session.speech_activity(100)
                await session.transcript(
                    100, "Me", Transcript("Tell me your favourite colour instead", False)
                )
                assert session.research.status()["research_running"] == 1
                job = session.research.jobs[0]
                assert job.detached and not job.task.cancelled()
                await session.transcript(100, "Me", Transcript("Tell me your favourite colour instead", True))
                session.pending = False
                session.task = asyncio.create_task(session._answer())
                await session.task
                assert played == ["Blue sounds good."]
                assert not job.task.done()  # Foreground chat proceeds while research remains blocked.
                release.set()
                await job.task
                assert not written and len(session.research.ready()) == 1
                await session._answer(research_only=True)
                assert len(integrated) == 1
                assert not session.research.jobs
                assert "OLD DRAFT" not in " ".join(played)
                assert len(written) == (1 if relevant else 0)
                assert any("About Python earlier" in text for text in played) == relevant
                assert session.metrics.last_outcome == ("completed" if relevant else "research_irrelevant")
        finally:
            await session.close()
            await chat.close()


@pytest.mark.parametrize("action", ["hold", "invalidate", "close", "cancel"])
async def test_research_lifetime_distinguishes_output_hold_from_cancellation(store, wav_bytes, action):
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def stream(messages, model):
        from bentebot.tools import WEB_LOOKUP_NOTICE, guard_allowed

        await WEB_LOOKUP_NOTICE.get()("")
        entered.set()
        try:
            await release.wait()
            assert await guard_allowed()
            yield "Verified earlier evidence."
        finally:
            stopped.set()

    settings = Settings()
    store.save(Event(VOICE, 100, "Me", "Look it up", kind="voice"))
    session, played, written = session_for(settings, store, SimpleNamespace(stream=stream), wav_bytes)
    session.latest_request = "Look it up"
    session.task = asyncio.create_task(session._answer())
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            job = session.research.jobs[0]
            if action == "hold":
                await session.hold_output(100)
                assert not stopped.is_set()
                assert session.held_reply_id is None
                release.set()
                await job.task
                assert job in session.research.ready()
                await session._answer(research_only=True)
                assert not played and not written and job in session.research.jobs
                session.release_output(100, "Continue")
                await session._answer(control_checked=True)
                assert not session.output_held and job in session.research.ready()
                assert not played and not written
            elif action == "invalidate":
                session.invalidate()
                await asyncio.gather(session.task, job.task, return_exceptions=True)
            elif action == "cancel":
                await session.cancel_research()
            else:
                await session.close()
            if action != "hold":
                assert stopped.is_set() and not session.research.jobs
                assert not played and not written
    finally:
        await session.close()


@pytest.mark.parametrize("text", ["Cancel the lookup", "Please stop searching", "Bente annuller søgningen"])
def test_explicit_abandonment_is_separate_from_speech_stop(text):
    assert spoken_control(text) == "cancel_research"
    assert spoken_control("Stop talking") == "stop"
    assert spoken_control("Don't cancel the lookup") is None


async def test_job_limits_expiry_and_cleanup_are_bounded():
    jobs = ResearchJobs(asyncio.Event())

    async def wait():
        await asyncio.Event().wait()

    first = ResearchJob("one", 1, task=asyncio.create_task(wait()))
    second = ResearchJob("two", 1, task=asyncio.create_task(wait()))
    third = ResearchJob("three", 1, task=asyncio.create_task(wait()))
    try:
        jobs.admit(first)
        jobs.admit(second)
        with pytest.raises(ValueError, match="bounded"):
            jobs.admit(third)
        first.created = time.monotonic() - 601
        jobs.prune()
        assert first not in jobs.jobs
        await asyncio.gather(first.task, return_exceptions=True)
        assert first.task.cancelled()
    finally:
        third.task.cancel()
        await asyncio.gather(third.task, return_exceptions=True)
        await jobs.close()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_native_gateway_lookup_survives_turn_and_returns_with_citations(store, wav_bytes, provider):
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def handler(request):
        assert request.url.host == "gateway"
        body = json.loads(request.content)
        requests.append(body)
        if request.url.path.endswith("chat/completions"):
            assert "tools" not in body
            data = json.loads(body["messages"][-1]["content"])["completed_research"]
            return httpx.Response(
                200,
                text=prose(
                    json.dumps(
                        {
                            "use": [data[0]["id"]],
                            "answer": "Back to your earlier lookup, here is the verified answer.",
                        }
                    )
                ),
            )
        search = body["tools"][0].get("type", "").startswith("web_search")
        if search:
            entered.set()
            await release.wait()
            return httpx.Response(200, text=native_fixture(provider, url=URL))
        content = json.dumps(body.get("input", body.get("messages")))
        if "A completely different question" in content:
            return httpx.Response(200, text=native_fixture(provider, ordinary=True))
        return httpx.Response(200, text=native_fixture(provider, plan=True))

    settings = Settings(
        chat_base_url="http://gateway/v1",
        chat_model="cloud",
        web_tools_enabled=True,
        **{f"native_web_{provider}_models": "cloud"},
    )
    store.save(Event(VOICE, 100, "Me", "Look up the current release", kind="voice"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(settings, client)
        session, played, written = session_for(settings, store, chat, wav_bytes)
        session.latest_request = "Look up the current release"
        session.task = asyncio.create_task(session._answer())
        try:
            async with asyncio.timeout(3):
                await entered.wait()
                await session.interrupt()
                job = session.research.jobs[0]
                assert job.detached and not job.task.cancelled()
                await session.transcript(100, "Me", Transcript("A completely different question", True))
                session.pending = False
                session.task = asyncio.create_task(session._answer())
                await session.task
                assert played == ["Verified answer."]
                release.set()
                await job.task
                await session._answer(research_only=True)
                assert not session.research.jobs
                assert len(written) == 1 and URL in written[0][0]
                assert "earlier lookup" in played[-1]
                assert chat.web_tools is None
        finally:
            await session.close()
            await chat.close()


@pytest.mark.parametrize("revoked", ["voice", "text", "screen"])
async def test_deferred_findings_are_not_offered_after_permission_or_consent_revocation(
    store, wav_bytes, revoked
):
    screen = Source("screen", 10, 1, owner_id=100, session_id="screen", audience_ids=(100,))
    origins = (VOICE, screen) if revoked == "screen" else (VOICE,)
    entered, release = asyncio.Event(), asyncio.Event()
    permitted = True

    async def stream(messages, model):
        from bentebot.tools import WEB_LOOKUP_NOTICE, guard_allowed

        await WEB_LOOKUP_NOTICE.get()("")
        entered.set()
        await release.wait()
        if not await guard_allowed():
            raise ValueError("Permission changed")
        yield "Earlier verified findings."

    async def policy(user, source):
        return permitted or (revoked == "text" and user == 100)

    store.save(Event(VOICE, 100, "Me", "Look it up", provenance=origins, kind="voice"))
    session, played, written = session_for(
        Settings(), store, SimpleNamespace(stream=stream), wav_bytes, policy=AccessPolicy(policy)
    )
    session.latest_request = "Look it up"
    session.task = asyncio.create_task(session._answer())
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            await session.interrupt()
            job = session.research.jobs[0]
            release.set()
            await job.task
            permitted = False
            await session._answer(research_only=True)
            assert not session.research.jobs and not played and not written
    finally:
        await session.close()


async def ready_job(session, text="Verified earlier findings."):
    async def completed():
        return text

    task = asyncio.create_task(completed())
    job = ResearchJob(
        "Earlier Python question",
        100,
        task=task,
        origins=(VOICE,),
        allowed=AsyncMock(return_value=True),
        detached=True,
        text=text,
    )
    job.record.web_sources[:] = [URL]
    session.research.admit(job)
    await task
    return job


async def test_coordinator_waits_for_quiet_and_returns_research_without_new_user_request(store, wav_bytes):
    completed = asyncio.Event()
    requests = []

    async def stream(messages, model):
        requests.append(messages)
        data = json.loads(messages[-1]["content"])["completed_research"]
        yield json.dumps(
            {"use": [data[0]["id"]], "answer": "About your earlier Python question, here is the release."}
        )
        completed.set()

    store.save(Event(VOICE, 100, "Me", "Earlier Python question", kind="voice"))
    settings = Settings(turn_silence_ms=40)
    session, played, written = session_for(settings, store, SimpleNamespace(stream=stream), wav_bytes)
    job = await ready_job(session)
    session.partial[100] = "still speaking"
    session.start()
    try:
        await asyncio.sleep(0.08)
        assert not requests and not played
        session.partial.clear()
        session.changed.set()
        async with asyncio.timeout(2):
            await completed.wait()
            await session.task
        assert len(requests) == 1 and len(written) == 1
        assert job not in session.research.jobs
        await asyncio.sleep(0.15)
        assert len(requests) == 1  # Completion does not poll the model or repeat itself.
    finally:
        await session.close()


async def test_short_speech_during_return_preserves_findings_and_rechecks_current_context(store, wav_bytes):
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def stream(messages, model):
        data = json.loads(messages[-1]["content"])["completed_research"]
        requests.append(data)
        yield json.dumps(
            {"use": [data[0]["id"]], "answer": "About Python earlier. The verified release is ready."}
        )

    async def playing(text):
        if not entered.is_set():
            entered.set()
            await release.wait()

    store.save(Event(VOICE, 100, "Me", "Earlier Python question", kind="voice"))
    session, played, written = session_for(
        Settings(turn_silence_ms=1), store, SimpleNamespace(stream=stream), wav_bytes, play=playing
    )
    job = await ready_job(session)
    session.task = asyncio.create_task(session._answer(research_only=True))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            session.speech_activity(100)
            await session.transcript(100, "Me", Transcript("Wait", False))
            assert job in session.research.ready() and job.heard
            session.partial.clear()
            session.last_speech = 0
            release.set()
            await session._answer(research_only=True)
            assert len(requests) == 2 and requests[1][0]["already_heard"]
            assert not session.research.jobs
            assert len(written) == 1  # A card delivered before barge-in is not posted again.
    finally:
        await session.close()


async def test_short_final_question_releases_foreground_but_keeps_lookup(store, wav_bytes):
    from bentebot.tools import WEB_LOOKUP_NOTICE

    entered = asyncio.Event()

    async def stream(messages, model):
        await WEB_LOOKUP_NOTICE.get()("")
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    store.save(Event(VOICE, 100, "Me", "Look up Python", kind="voice"))
    session, _, _ = session_for(Settings(), store, SimpleNamespace(stream=stream), wav_bytes)
    session.latest_request = "Look up Python"
    session.task = asyncio.create_task(session._answer())
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            await session.transcript(100, "Me", Transcript("And Linux?", True))
            assert session.task.done() and session.pending
            assert session.research.status()["research_running"] == 1
            assert session.research.jobs[0].detached
    finally:
        await session.close()
