import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience, Event, Source, Transcript
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.providers import OpenAIChat, compatible_messages
from bentebot.voice_control import control_candidate, spoken_control


@pytest.mark.parametrize(
    "text,action",
    [
        ("Sorry about that Bentebot, please continue", "continue"),
        ("Don't worry about that, Bentebot. Please continue.", "continue"),
        ("Let's continue Bentebot", "continue"),
        ("Just continue. Thank you.", "continue"),
        ("Carry on Bente", "continue"),
        ("Resume the story please", "continue"),
        ("You can carry on now Bente", "continue"),
        ("Go ahead Bentebot", "continue"),
        ("Just forget that, tell me a different story", "replace"),
        ("Never mind, tell me something different", "replace"),
        (
            "Excellent. never mind, just forget that. Can you tell me a different story about a space pirate?",
            "replace",
        ),
        ("Bentebot, tell me a different story", "replace"),
        ("Bentebot, what is the capital of Sweden?", "replace"),
    ],
)
def test_explicit_handback_only_acts_in_held_context(text, action):
    assert spoken_control(text, output_held=True) == action
    assert control_candidate(text, output_held=True)
    assert spoken_control(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "John, continue your story",
        "Don't continue Bentebot",
        "If I say continue, start again",
        'Say "continue Bentebot"',
        "You continued talking",
        "We should continue later",
        "Tell John to continue",
        "Let's talk about the garden",
        "Can you pause the video?",
    ],
)
def test_side_conversation_is_not_handback(text):
    assert spoken_control(text, output_held=True) is None
    assert not control_candidate(text, output_held=True)


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("label", ["continue", "replace"])
async def test_resolver_new_actions_require_explicit_held_context(held, label):
    def handler(request):
        body = json.loads(request.content)
        assert ("currently on hold" in body["messages"][0]["content"]) == held
        return httpx.Response(200, json={"choices": [{"message": {"content": label}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)
        assert await chat.resolve_spoken_control("Continue Bentebot", "local", output_held=held) == (
            label if held else None
        )


@pytest_asyncio.fixture
async def call(store, wav_bytes):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1}), turn_silence_ms=200), store)
    channel = SimpleNamespace(id=10)
    members = {
        user: SimpleNamespace(id=user, bot=False, voice=SimpleNamespace(channel=channel))
        for user in (100, 200)
    }
    bot.get_guild = lambda _: SimpleNamespace(get_member=members.get)
    source = Source("channel", 10, 1)
    calls = []

    async def stream(messages, model):
        calls.append(messages)
        yield "A new sentence."

    async def audience():
        return Audience(frozenset({100, 200}), 1)

    async def can_read(*args):
        return True

    async def play(pcm, text, progress):
        progress(text)

    memory = Memory(store, AccessPolicy(can_read), 4000)
    conversation = VoiceConversation(
        bot.settings,
        store,
        memory,
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        source,
        999,
        audience,
        AsyncMock(side_effect=play),
        Mock(),
        AsyncMock(),
        control=lambda user, text: bot.handle_spoken_control(1, 10, user, text),
    )
    bot.voice_sessions[1] = (conversation, None, None)
    old = Event(
        source,
        999,
        "Bentebot",
        "The dad opened his laptop.",
        kind="assistant",
        generated="The dad opened his laptop. Then he fixed a tricky bug.",
        spoken="The dad opened his laptop.",
        interrupted=True,
        delivery="voice",
        provenance=(source,),
    )
    store.save(old)
    conversation.last_reply_id = old.id
    conversation.metrics.last_outcome = "interrupted"
    await conversation.hold_output(100)
    try:
        yield SimpleNamespace(
            bot=bot, conversation=conversation, calls=calls, old=old, store=store, members=members
        )
    finally:
        await asyncio.gather(*list(bot.voice_jobs))
        bot.voice_sessions.clear()
        await conversation.close()
        await bot.close()


async def control(call, text, user=100):
    c = call.conversation
    await c.transcript(user, "Caller", Transcript(text, final=True))
    await c._dispatch_control(user, text)
    await asyncio.gather(*list(call.bot.voice_jobs))


async def test_held_output_ignores_both_callers_but_retains_their_transcripts(call):
    c = call.conversation
    c.start()
    for user, text in [(100, "John, how was your day?"), (200, "Pretty good, I fixed the garden fence")]:
        await c.transcript(user, "Caller", Transcript(text, final=True))
    await asyncio.sleep(0.4)
    assert c.output_held and c.metrics.output_held
    assert not c.pending and not call.calls
    assert len([e for e in call.store.recent(1, frozenset()) if e.kind == "voice"]) == 2


@pytest.mark.parametrize(
    "utterance",
    ["Sorry about that Bentebot, please continue", "Don't worry about that, Bentebot. Please continue."],
)
async def test_explicit_continue_recovers_paused_answer_after_side_conversation(call, utterance):
    call.conversation.memory.voice_max_chars = 1500
    for _ in range(25):
        call.store.save(Event(call.old.source, 200, "Other", "Side conversation " * 30))
    await control(call, utterance)
    c = call.conversation
    assert not c.output_held and c.pending
    assert c.metrics.last_control_action == "continue"
    await c._answer()
    messages = call.calls[0]
    background = next(
        m for m in messages if m["role"] == "user" and "paused_reply_background" in m["content"]
    )
    assert "Then he fixed a tricky bug" in background["content"]
    assert "heard_so_far_estimated" in background["content"]
    assert any(m["role"] == "system" and "Do not restart" in m["content"] for m in messages)
    assert any("brief natural lead-in" in m["content"] for m in messages if m["role"] == "system")
    assert c.metrics.last_chat["held_turn_action"] == "continue"
    assert c.metrics.last_chat["voice_history_budget_chars"] == 1500
    wire = compatible_messages(messages)
    assert wire[0]["role"] == "system"
    assert all(m["role"] != "system" for m in wire[1:])


@pytest.mark.parametrize(
    "utterance",
    [
        "Just forget that, tell me a different story",
        "Excellent. never mind, just forget that. Can you tell me a different story about a space pirate?",
    ],
)
async def test_replacement_discards_paused_task_and_generates_new_answer(call, utterance):
    await control(call, utterance)
    c = call.conversation
    assert not c.output_held and c.held_reply_id is None
    assert c.metrics.last_control_action == "replace"
    await c._answer()
    messages = call.calls[0]
    assert not any("paused_reply_background" in m["content"] for m in messages)
    assert any(m["role"] == "system" and "Discard the old answer" in m["content"] for m in messages)
    assert c.metrics.last_chat["held_turn_action"] == "replace"
    wire = compatible_messages(messages)
    assert wire[0]["role"] == "system"
    assert all(m["role"] != "system" for m in wire[1:])


@pytest.mark.parametrize("utterance", ["Please continue", "Just forget that, tell me a different story"])
async def test_held_turn_generation_passes_strict_initial_system_backend(call, utterance):
    requests = []

    def strict_backend(request):
        payload = json.loads(request.content)
        requests.append(payload)
        messages = payload["messages"]
        if any(m["role"] == "system" for m in messages[1:]):
            return httpx.Response(500, json={"error": {"message": "System message must be at the beginning"}})
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"The pirate sailed north."}}]}\n\ndata: [DONE]\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(strict_backend)) as client:
        c = call.conversation
        c.chat = OpenAIChat(Settings(), client)
        await control(call, utterance)
        await c._answer()
        assert len(requests) == 1 and c.metrics.responses == 1
        assert c.metrics.last_outcome == "completed" and not c.output_held
        assert requests[0]["messages"][0]["role"] == "system"


@pytest.mark.parametrize(
    "utterance,classified",
    [
        ("Oh, sorry, sorry, sorry. Can you please pause again?", False),
        ("Can you du plads pause for et sekund?", True),
    ],
)
async def test_reported_pause_transcripts_activate_hold_and_suppress_side_reply(call, utterance, classified):
    c = call.conversation
    c.release_output(100, "Continue")
    c.pending = False
    call.bot.chat.resolve_spoken_control = AsyncMock(return_value="stop")
    await control(call, utterance)
    assert c.output_held and c.metrics.control_requests == 1
    assert c.metrics.last_control_action == "stop"
    assert c.metrics.last_control_result == "completed"
    assert c.metrics.last_control_method == ("intent" if classified else "direct")
    assert call.bot.chat.resolve_spoken_control.await_count == int(classified)
    c.start()
    await c.transcript(100, "Caller", Transcript("I'm just checking something over here.", final=True))
    await asyncio.sleep(0.35)
    assert c.output_held and not c.pending and not call.calls


async def test_other_caller_must_address_bot_to_give_its_turn_back(call):
    await control(call, "Continue", 200)
    assert call.conversation.output_held
    assert call.conversation.metrics.last_control_result == "ignored_side_conversation"
    await control(call, "Continue Bentebot", 200)
    assert not call.conversation.output_held


async def test_natural_unlisted_request_reaches_semantic_router_through_voice_loop(call):
    c = call.conversation
    c.release_output(100, "Continue")
    c.pending = False
    call.bot.chat.resolve_spoken_control = AsyncMock(return_value="leave")
    done = asyncio.Event()
    call.bot.leave_voice = AsyncMock(side_effect=lambda _: done.set())
    c.start()
    await c.transcript(100, "Caller", Transcript("Get out of the call now Bentebot", final=True))
    await asyncio.wait_for(done.wait(), 2)
    await asyncio.gather(*list(call.bot.voice_jobs))
    call.bot.chat.resolve_spoken_control.assert_awaited_once()
    assert c.metrics.last_control_action == "leave"
    assert c.metrics.last_control_method == "intent"


async def test_semantic_none_is_conversation_not_failed_control_clarification(call):
    c = call.conversation
    c.release_output(100, "Continue")
    c.pending = False
    call.bot.chat.resolve_spoken_control = AsyncMock(return_value="none")
    c.control_feedback = AsyncMock()
    await control(call, "Bente, would you tell me about penguins?")
    assert call.calls and c.metrics.last_control_result == "not_control"
    c.control_feedback.assert_not_awaited()
    assert c.metrics.responses == 1


@pytest.mark.parametrize("new_final", [False, True])
async def test_classified_leave_ignores_raw_vad_but_respects_newer_transcripts(call, new_final, caplog):
    caplog.set_level("INFO", logger="bentebot.discord_app")
    c = call.conversation
    c.speaker_last_voice[100] = 1.0
    call.bot.leave_voice = AsyncMock()

    async def resolve(*args, **kwargs):
        c.speaker_last_voice[100] = 2.0  # Noise/voice activity is not a new instruction.
        if new_final:
            await c.transcript(100, "Caller", Transcript("Actually don't leave", final=True))
        return "leave"

    call.bot.chat.resolve_spoken_control = AsyncMock(side_effect=resolve)
    await control(call, "Can you get out of the call now Bentebot?")
    assert call.bot.leave_voice.await_count == (0 if new_final else 1)
    assert c.metrics.last_control_result == ("stale" if new_final else "completed")
    if new_final:
        assert "reason=newer_transcript" in caplog.text


async def test_failed_continuation_reholds_original_anchor_until_explicit_retry(call):
    c = call.conversation
    original_stream = c.chat.stream

    async def broken(*args):
        raise RuntimeError("Generation unavailable")
        yield "unreachable"

    c.chat.stream = broken
    await control(call, "Please continue")
    await c._answer()
    assert c.output_held and c.held_reply_id == call.old.id
    assert c.held_by == 100 and not c.pending
    await c.transcript(100, "Caller", Transcript("I'm checking something over here", final=True))
    assert c.output_held and not c.pending
    c.chat.stream = original_stream
    await control(call, "Please continue")
    await c._answer()
    assert not c.output_held and call.calls
    assert any("paused_reply_background" in m["content"] for m in call.calls[-1])


async def test_actual_new_stop_request_can_cancel_pending_farewell_without_disconnect(call):
    c = call.conversation
    entered = asyncio.Event()
    c.control_feedback = AsyncMock(return_value="Catch you later.")
    call.bot.leave_voice = AsyncMock()

    async def play(*args):
        entered.set()
        await asyncio.Event().wait()

    c.play = play
    job = asyncio.create_task(control(call, "Leave the call Bentebot"))
    await asyncio.wait_for(entered.wait(), 1)
    await c.transcript(100, "Caller", Transcript("Don't leave Bentebot", final=True))
    await asyncio.wait_for(job, 1)
    call.bot.leave_voice.assert_not_awaited()
    assert c.metrics.last_control_result == "stale" and c.output_held


@pytest.mark.parametrize("remove", [False, True])
async def test_resume_cannot_recover_revoked_or_deleted_paused_context(call, remove):
    private = Source("channel", 99, 1)
    call.old.provenance = (private,)
    call.old.generated = "PRIVATE SECRET DRAFT"
    call.store.save(call.old)
    if remove:
        call.store.delete(call.old.id)

    async def can_read(user, source):
        return source.channel_id != 99

    call.conversation.memory.policy = AccessPolicy(can_read)
    call.conversation.control_feedback = AsyncMock(return_value="Remind me where we were?")
    await control(call, "Please continue")
    await call.conversation.task
    assert not call.calls
    spoken = call.store.recent(1, frozenset())[-1].spoken
    assert spoken == "Remind me where we were?" and "SECRET" not in spoken


async def test_quick_farewell_during_ack_does_not_cancel_leave(call):
    c = call.conversation
    c.speaker_last_voice[100] = 1.0
    c.control_feedback = AsyncMock(return_value="See you later.")

    async def play(pcm, text, progress):
        c.speaker_last_voice[100] = 2.0
        progress(text)

    c.play = play
    call.bot.leave_voice = AsyncMock()
    await control(call, "Leave the call Bentebot")
    call.bot.leave_voice.assert_awaited_once_with(1)


async def test_explicit_interruption_of_ack_cancels_pending_leave(call):
    c = call.conversation
    entered = asyncio.Event()

    async def feedback(*args):
        entered.set()
        await asyncio.Event().wait()

    c.control_feedback = feedback
    call.bot.leave_voice = AsyncMock()
    task = asyncio.create_task(control(call, "Leave the call Bentebot"))
    await entered.wait()
    await c.interrupt()
    await asyncio.wait_for(task, 1)
    call.bot.leave_voice.assert_not_awaited()
    assert c.metrics.last_control_result == "stale"


async def test_hold_still_allows_leave(call):
    call.bot.leave_voice = AsyncMock()
    await control(call, "Please leave the call Bentebot")
    call.bot.leave_voice.assert_awaited_once_with(1)


async def test_explicit_continue_is_not_executed_after_caller_leaves(call):
    call.members[100].voice = None
    await control(call, "Continue Bentebot")
    assert call.conversation.output_held and not call.conversation.pending
    assert call.conversation.metrics.last_control_result == "denied"


async def test_slash_resume_releases_output_without_restarting_stt(call):
    call.bot._replacement_receiver = Mock(side_effect=AssertionError("STT is still running"))
    assert await call.bot.resume_voice(1, 100)
    assert not call.conversation.output_held and call.conversation.pending
    assert call.conversation.next_reply_mode == "continue"


async def test_stale_classified_continue_cannot_release_a_new_hold(call):
    async def resolve(text, model, **kwargs):
        assert kwargs == {"output_held": True}
        await call.conversation.hold_output(200)
        return "continue"

    call.bot.chat.resolve_spoken_control = AsyncMock(side_effect=resolve)
    await control(call, "Bentebot, you have the floor, carry on")
    assert call.conversation.output_held and call.conversation.held_by == 200
    assert call.conversation.metrics.last_control_result == "stale"


async def test_held_resume_does_not_execute_on_interim_transcript(call):
    c = call.conversation
    c.start()
    await c.transcript(100, "Me", Transcript("Continue Bentebot"))
    await asyncio.sleep(0.25)
    assert c.output_held and not c.pending_controls and not call.calls


async def test_held_classifier_can_replace_answer_with_new_task(call):
    call.bot.chat.resolve_spoken_control = AsyncMock(return_value="replace")
    await control(call, "Cancel that, I want a new story")
    assert call.conversation.metrics.last_control_action == "replace"
    assert not call.conversation.output_held
    assert call.conversation.next_reply_anchor is None


@pytest.mark.parametrize(
    "action,utterance", [("leave", "Leave the call Bentebot"), ("pause", "Stop listening")]
)
async def test_model_ack_speaks_before_leave_or_capture_pause(call, action, utterance):
    c = call.conversation
    events = []
    c.control_feedback = AsyncMock(return_value="Right, see you later.")

    async def play(pcm, text, progress):
        assert events == []
        events.append("spoken")
        progress(text)

    c.play = play

    async def execute(*args):
        events.append("executed")

    if action == "leave":
        call.bot.leave_voice = AsyncMock(side_effect=execute)
    else:
        call.bot.pause_voice = AsyncMock(side_effect=execute)
    await control(call, utterance)
    assert events == ["spoken", "executed"]
    assert c.metrics.last_control_result == "completed"
    assert c.hold_replies == 0


@pytest.mark.parametrize("utterance", ["Please continue Bentebot", "Forget that, tell me a different story"])
async def test_resume_generates_during_ack_but_never_overlaps_audio(call, utterance):
    c = call.conversation
    c.control_feedback = AsyncMock(return_value="Right, carrying on.")
    generated = asyncio.Event()
    ack_playing = False
    heard = []

    async def stream(messages, model):
        assert ack_playing
        generated.set()
        yield "The pirate opened the airlock."

    async def play(pcm, text, progress):
        nonlocal ack_playing
        if text == "Right, carrying on.":
            ack_playing = True
            await asyncio.wait_for(generated.wait(), 1)
            await asyncio.sleep(0.05)  # Allow generated audio to reach the playback barrier.
            assert heard == []
            progress(text)
            heard.append("ack")
            ack_playing = False
        else:
            assert not ack_playing
            heard.append("answer")
            progress(text)

    c.chat = SimpleNamespace(stream=stream)
    c.play = play
    await control(call, utterance)
    await asyncio.wait_for(c.task, 2)
    assert heard == ["ack", "answer"]
    assert not c.pending and not c.output_held and c.hold_replies == 0
    assert c.metrics.last_outcome == "completed"
    assert c.metrics.last_turn["acknowledgement_playback_wait_ms"] > 0
    assert c.metrics.last_control_result == "completed"


async def test_new_pause_during_overlap_cancels_prefetched_audio_and_preserves_heard_anchor(call):
    c = call.conversation
    c.control_feedback = AsyncMock(return_value="Right, carrying on.")
    generated = asyncio.Event()
    ack_started = asyncio.Event()
    never_finish_ack = asyncio.Event()
    story_played = []

    async def stream(messages, model):
        generated.set()
        yield "Unheard continuation."

    async def play(pcm, text, progress):
        if text == "Right, carrying on.":
            ack_started.set()
            await never_finish_ack.wait()
        else:
            story_played.append(text)

    c.chat = SimpleNamespace(stream=stream)
    c.play = play
    first = asyncio.create_task(control(call, "Please continue Bentebot"))
    await asyncio.wait_for(ack_started.wait(), 1)
    await asyncio.wait_for(generated.wait(), 1)
    await asyncio.sleep(0.05)
    await asyncio.wait_for(c.hold_output(100), 1)
    await asyncio.wait_for(first, 1)
    assert c.output_held and c.held_reply_id == call.old.id
    assert not story_played and c.task.done() and c.hold_replies == 0


async def test_member_leaving_during_overlap_discards_prepared_story(call):
    c = call.conversation
    c.control_feedback = AsyncMock(return_value="Right, carrying on.")
    generated = asyncio.Event()
    heard = []

    async def stream(messages, model):
        generated.set()
        yield "Unheard continuation."

    async def play(pcm, text, progress):
        heard.append(text)
        await asyncio.wait_for(generated.wait(), 1)
        call.members[100].voice = None
        progress(text)

    c.chat = SimpleNamespace(stream=stream)
    c.play = play
    await control(call, "Please continue Bentebot")
    assert heard == ["Right, carrying on."]
    assert c.task.done() and c.hold_replies == 0
    assert c.metrics.last_control_result == "stale"
    assert c.output_held and c.held_reply_id == call.old.id


async def test_retrieved_permission_revoked_during_ack_prevents_story_audio(call):
    c = call.conversation
    private = Source("channel", 99, 1)
    call.old.provenance = (private,)
    call.store.save(call.old)
    private_allowed = True

    async def permission(user, source):
        return source != private or private_allowed

    c.memory.policy = AccessPolicy(permission)
    c.control_feedback = AsyncMock(return_value="Right, carrying on.")
    generated = asyncio.Event()
    heard = []

    async def stream(messages, model):
        generated.set()
        yield "PRIVATE continuation."

    async def play(pcm, text, progress):
        nonlocal private_allowed
        heard.append(text)
        await asyncio.wait_for(generated.wait(), 1)
        private_allowed = False
        progress(text)

    c.chat = SimpleNamespace(stream=stream)
    c.play = play
    await control(call, "Please continue Bentebot")
    await asyncio.wait_for(c.task, 2)
    assert heard == ["Right, carrying on."]
    assert c.metrics.provider_errors == 1 and c.output_held
    assert c.held_reply_id == call.old.id


async def test_real_turn_loop_does_not_dispatch_overlapped_resume_twice(call):
    c = call.conversation
    c.control_feedback = AsyncMock(return_value="Right, carrying on.")
    generated = asyncio.Event()
    played = asyncio.Event()
    generations = 0

    async def stream(messages, model):
        nonlocal generations
        generations += 1
        generated.set()
        yield "The pirate opened the airlock."

    async def play(pcm, text, progress):
        if text == "Right, carrying on.":
            await asyncio.wait_for(generated.wait(), 1)
        else:
            played.set()
        progress(text)

    c.chat = SimpleNamespace(stream=stream)
    c.play = play
    c.start()
    await c.transcript(100, "Caller", Transcript("Please continue Bentebot", final=True))
    await asyncio.wait_for(played.wait(), 2)
    await asyncio.sleep(0.2)
    assert generations == 1 and not c.pending and not c.pending_controls
    assert c.metrics.responses == 1 and not c.output_held


async def test_skipped_continue_ack_still_releases_normally_without_prefetch(call):
    c = call.conversation
    c.control_feedback = AsyncMock(return_value=None)
    await control(call, "Please continue Bentebot")
    assert not c.output_held and c.pending
    await c._answer()
    assert c.metrics.last_outcome == "completed"
    assert "acknowledgement_playback_wait_ms" not in c.metrics.last_turn


async def test_failed_ack_never_prevents_actual_command(call):
    call.conversation.control_feedback = AsyncMock(side_effect=RuntimeError("PRIVATE PROVIDER DETAIL"))
    call.bot.leave_voice = AsyncMock()
    await control(call, "Leave the call Bentebot")
    call.bot.leave_voice.assert_awaited_once_with(1)
    assert call.conversation.metrics.last_control_result == "completed"
    assert call.conversation.hold_replies == 0


async def test_farewell_finishes_even_when_preparation_budget_would_expire_during_playback(call, monkeypatch):
    monkeypatch.setattr("bentebot.conversation.CONTROL_ACK_PREPARATION_TIMEOUT", 0.06)
    c = call.conversation
    c.control_feedback = AsyncMock(return_value="Fair enough, I'll hang up now. Bye!")
    events = []

    async def play(pcm, text, progress):
        events.append("started")
        await asyncio.sleep(0.12)
        progress(text)
        events.append("finished")

    async def leave(*args):
        events.append("left")

    c.play = play
    call.bot.leave_voice = AsyncMock(side_effect=leave)
    await control(call, "Please leave the call Bentebot")
    assert events[0] == "started" and events[-1] == "left"
    assert events.count("started") == events.count("finished")
    assert c.metrics.last_control_result == "completed"


async def test_stalled_farewell_preparation_still_times_out_and_leaves(call, monkeypatch):
    monkeypatch.setattr("bentebot.conversation.CONTROL_ACK_PREPARATION_TIMEOUT", 0.03)
    c = call.conversation
    c.control_feedback = AsyncMock(return_value="Bye for now.")

    async def stalled(*args):
        await asyncio.Event().wait()

    c.tts.synthesize = stalled
    c.play = AsyncMock()
    call.bot.leave_voice = AsyncMock()
    await asyncio.wait_for(control(call, "Please leave the call Bentebot"), 1)
    c.play.assert_not_awaited()
    call.bot.leave_voice.assert_awaited_once()


async def test_stop_ack_does_not_replace_the_story_anchor(call):
    c = call.conversation
    c.control_feedback = AsyncMock(return_value="I'll wait, give me a nudge.")
    await control(call, "Stop talking")
    assert c.output_held and c.held_reply_id == call.old.id and c.last_reply_id == call.old.id
    assert call.store.recent(1, frozenset())[-1].generated == "I'll wait, give me a nudge."


async def test_speaking_again_can_cancel_only_ack_without_deadlock(call):
    c = call.conversation
    entered = asyncio.Event()

    async def feedback(*args):
        entered.set()
        await asyncio.Event().wait()

    c.control_feedback = feedback
    task = asyncio.create_task(c.acknowledge_control("stop", "pause", lambda: True))
    await entered.wait()
    await asyncio.wait_for(c.interrupt(), 1)
    await asyncio.wait_for(task, 1)
    assert c.control_ack_task is None and c.hold_replies == 0
    assert c.output_held


async def test_caller_leaving_during_ack_prevents_stale_command(call):
    async def feedback(*args):
        call.members[100].voice = None
        return "See you later."

    c = call.conversation
    c.control_feedback = feedback
    c.play = AsyncMock()
    call.bot.leave_voice = AsyncMock()
    await control(call, "Leave the call Bentebot")
    call.bot.leave_voice.assert_not_awaited()
    c.play.assert_not_awaited()
    assert c.metrics.last_control_result == "stale"
