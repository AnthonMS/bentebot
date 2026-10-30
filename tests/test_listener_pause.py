import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience, Event, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import ProviderError


def conversation(store, wav_bytes):
    people = {100, 200}
    policy = AccessPolicy(AsyncMock(return_value=True))
    session = VoiceConversation(
        Settings(health_port=0),
        store,
        Memory(store, policy, 8000),
        None,
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        Source("channel", 10, 1),
        999,
        AsyncMock(side_effect=lambda: Audience(frozenset(people), 1)),
        AsyncMock(),
        Mock(),
        AsyncMock(),
        control_feedback=AsyncMock(return_value="Picking up where I left off."),
    )
    return session, people


async def test_departure_with_remaining_human_preserves_actual_generation_and_audio(store, wav_bytes):
    session, people = conversation(store, wav_bytes)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def play(pcm, text, progress):
        started.set()
        await finish.wait()
        progress(text)

    async def stream(*args):
        yield "This answer continues normally. "

    session.chat = SimpleNamespace(stream=stream)
    session.play = play
    session.latest_request, session.latest_speaker_id = "Tell me something", 100
    store.save(Event(session.source, 100, "Owner", session.latest_request))
    bot = BenteBot(session.settings, store)
    channel = SimpleNamespace(id=10, members=[])
    guild = SimpleNamespace(id=1, get_channel=lambda _: channel, voice_client=None)
    person = SimpleNamespace(id=100, bot=False, guild=guild)
    channel.members = [SimpleNamespace(id=200, bot=False)]
    bot.get_guild = lambda _: guild
    bot.voice_sessions[1] = (session, SimpleNamespace(close=AsyncMock()), None)
    try:
        session.task = asyncio.create_task(session._answer())
        await started.wait()
        people.remove(100)
        await bot.on_voice_state_update(
            person, SimpleNamespace(channel=channel), SimpleNamespace(channel=None)
        )
        assert not session.task.done()
        assert session.epoch == 0 and session.research.generation == 0
        session.stop_playback.assert_not_called()
        finish.set()
        await session.task
        assert session.metrics.last_outcome == "completed"
        session.notify.assert_not_awaited()
    finally:
        await bot.close()


async def test_empty_call_finishes_inference_and_resumes_whole_sentence_after_ack(store, wav_bytes):
    session, people = conversation(store, wav_bytes)
    entered = asyncio.Event()
    generated = asyncio.Event()
    continue_generation = asyncio.Event()
    played = []
    attempts = 0

    async def play(pcm, text, progress):
        nonlocal attempts
        played.append(text)
        if text.startswith("The whole sentence"):
            attempts += 1
            if attempts == 1:
                progress("The whole")
                entered.set()
                await asyncio.Event().wait()
        progress(text)

    async def stream(*args):
        yield "The whole sentence starts here. "
        await continue_generation.wait()
        yield "The next sentence follows. "
        generated.set()

    session.chat = SimpleNamespace(stream=stream)
    session.play = play
    session.latest_request, session.latest_speaker_id = "Tell me something", 100
    store.save(Event(session.source, 100, "Owner", session.latest_request))
    try:
        session.task = asyncio.create_task(session._answer())
        await entered.wait()
        people.clear()
        session.set_listener_presence(False)
        continue_generation.set()
        await generated.wait()
        await asyncio.sleep(0.01)
        assert played == ["The whole sentence starts here."]
        assert not session.task.done() and session.epoch == 0
        people.add(200)  # Another authorized listener may return, not only the requester.
        session.listener_joined()
        session.set_listener_presence(True)
        await asyncio.wait_for(session.task, 2)
        assert played == [
            "The whole sentence starts here.",
            "Picking up where I left off.",
            "The whole sentence starts here.",
            "The next sentence follows.",
        ]
        assert session.metrics.last_outcome == "completed"
        session.notify.assert_not_awaited()
    finally:
        await session.close()


async def test_return_replays_split_sentence_from_start_not_middle(store, wav_bytes):
    session, _ = conversation(store, wav_bytes)
    entered = asyncio.Event()
    played = []
    first = True

    async def play(pcm, text, progress):
        nonlocal first
        played.append(text)
        if first:
            first = False
            entered.set()
            await asyncio.Event().wait()
        progress(text)

    session.play = play
    overlap = ["This long sentence begins"]
    try:
        session.task = asyncio.create_task(
            session._play_with_listeners(
                b"pcm", "and finishes right here.", Mock(), AsyncMock(return_value=True), overlap
            )
        )
        await entered.wait()
        session.set_listener_presence(False)
        session.set_listener_presence(True)  # Rapid rejoin must not miss the transition.
        await asyncio.wait_for(session.task, 2)
        assert played == [
            "and finishes right here.",
            "Picking up where I left off.",
            "This long sentence begins",
            "and finishes right here.",
        ]
        assert not overlap
    finally:
        await session.close()


async def test_return_cannot_replay_private_context_to_new_listener(store, wav_bytes):
    session, _ = conversation(store, wav_bytes)
    session.listener_resume_pending = True
    try:
        with pytest.raises(ProviderError):
            await session._play_with_listeners(
                b"private", "Private answer.", Mock(), AsyncMock(return_value=False), []
            )
        session.play.assert_not_awaited()
        session.control_feedback.assert_not_awaited()
    finally:
        await session.close()


async def test_user_hold_is_not_released_by_listener_return(store, wav_bytes):
    session, _ = conversation(store, wav_bytes)
    try:
        await session.hold_output(100)
        session.set_listener_presence(False)
        session.set_listener_presence(True)
        assert session.output_held and session.held_by == 100
        assert not session.pending
        session.play.assert_not_awaited()
    finally:
        await session.close()


async def test_close_cancels_answer_waiting_for_listeners(store, wav_bytes):
    session, _ = conversation(store, wav_bytes)
    session.set_listener_presence(False)
    session.task = asyncio.create_task(
        session._play_with_listeners(b"pcm", "Waiting answer.", Mock(), AsyncMock(return_value=True), [])
    )
    await asyncio.sleep(0)
    await session.close()
    assert session.task.cancelled()
    session.play.assert_not_awaited()


@pytest.mark.parametrize("return_before_timeout", [True, False])
async def test_lookup_survives_empty_grace_and_is_closed_at_timeout(store, wav_bytes, return_before_timeout):
    from bentebot.tools import WEB_LOOKUP_NOTICE, guard_allowed

    session, people = conversation(store, wav_bytes)
    session.settings = Settings(health_port=0, voice_alone_timeout=0.04)
    started, release, finished, cancelled = (asyncio.Event() for _ in range(4))

    async def stream(*args):
        await WEB_LOOKUP_NOTICE.get()("")
        started.set()
        try:
            await release.wait()
            assert await guard_allowed()
            finished.set()
            yield "The lookup completed during the empty call. "
        except asyncio.CancelledError:
            cancelled.set()
            raise

    session.chat = SimpleNamespace(stream=stream)
    session.latest_request, session.latest_speaker_id = "Look this up", 100
    store.save(Event(session.source, 100, "Owner", session.latest_request))
    bot = BenteBot(session.settings, store)
    channel = SimpleNamespace(id=10, members=[])
    client = SimpleNamespace(disconnect=AsyncMock())
    guild = SimpleNamespace(id=1, get_channel=lambda _: channel, voice_client=client)
    bot.get_guild = lambda _: guild
    bot.voice_sessions[1] = (session, SimpleNamespace(close=AsyncMock()), None)
    try:
        session.task = asyncio.create_task(session._answer())
        await asyncio.wait_for(started.wait(), 2)
        people.clear()
        bot.update_alone_timer(1)
        timer = bot._alone_tasks[1]
        assert session.research.status()["research_running"] == 1
        if return_before_timeout:
            release.set()
            await asyncio.wait_for(finished.wait(), 2)
            assert not cancelled.is_set()
            people.add(200)
            channel.members = [SimpleNamespace(id=200, bot=False)]
            session.listener_joined()
            bot.update_alone_timer(1)
            await asyncio.gather(timer, return_exceptions=True)
            assert timer.cancelled()
            await asyncio.wait_for(session.task, 2)
            client.disconnect.assert_not_awaited()
            assert session.metrics.last_outcome == "completed"
        else:
            await asyncio.wait_for(timer, 2)
            assert cancelled.is_set() and session.closed
            assert session.task.cancelled() and not session.research.jobs
            client.disconnect.assert_awaited_once_with(force=True)
    finally:
        await bot.close()


async def test_return_still_acknowledges_if_optional_feedback_model_fails(store, wav_bytes):
    session, _ = conversation(store, wav_bytes)
    session.control_feedback.side_effect = RuntimeError("gateway unavailable")
    session.listener_resume_pending = True
    try:
        await session._play_with_listeners(
            b"pcm", "Whole sentence.", Mock(), AsyncMock(return_value=True), []
        )
        assert [call.args[1] for call in session.play.await_args_list] == [
            "Picking up where I left off.",
            "Whole sentence.",
        ]
    finally:
        await session.close()
