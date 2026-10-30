import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.domain import Metrics, Source
from bentebot.providers import OpenAIChat
from bentebot.voice_control import control_candidate


@pytest.mark.parametrize(
    "text",
    [
        "Please leave the call. bend a bar.",
        "Hello? please sleep. the call.",
        "Stop speaking Bender",
        "Can you pause for a sec Panda?",
        "Kan du forlade opkaldet Bender?",
        "Bentebot, bug, please stop. talking.",
        "Before we continue, could you pause for a second?",
        "Just a moment, can you leave the call please?",
        "Can you du plads pause for et sekund?",
    ],
)
def test_short_direct_requests_can_use_bounded_intent_resolution(text):
    assert control_candidate(text)


@pytest.mark.parametrize(
    "text",
    [
        "Do you know how to leave the call?",
        "Please explain how to leave the call",
        "Don't stop speaking",
        "Leave the call if John joins",
        "John please leave the call",
        'Say "please leave the call"',
        "Please write a function to leave the call",
        "Let us discuss the pause menu",
        "leave a comment on my code",
        "Stop speaking like a robot",
        "Stop talking about Docker",
        "Stop speaking in Danish",
        "Bentebot, please start talking",
        "Bentebot, talking now",
        "Bentebot, Bentebot. talk you now.",
        "Release. talking. Inc.",
        "Bentebot, Bentebot. talking. thing.",
        "Bentebot, you stopped talking",
        "Bentebot, tell John please stop talking",
        "Bentebot, bug, please stop talking about Docker",
        "Bentebot, bug, please don't stop talking",
        "Bentebot, bug, please stop talking if John joins",
        "I wonder, can you leave the call?",
        "John asked, can you leave the call?",
        "Do you know whether I can say can you stop talking?",
        "Before we continue, can you pause the video?",
        "Before we continue, can you stop talking about Docker?",
        "Before we continue, can you not stop talking?",
        "De leaves de sleevede kolde Bentebot.",
        "Can you du plads pause the video for et sekund?",
        "Can you du plads not pause for et sekund?",
        "John asked can you du plads pause for et sekund?",
        "x" * 201,
    ],
)
def test_discussion_and_other_peoples_commands_do_not_use_intent_resolution(text):
    assert not control_candidate(text)


@pytest.mark.parametrize("label", ["leave", "stop", "pause", "none", "I will leave", "leave\nignore rules"])
async def test_resolver_sends_only_current_utterance_and_accepts_only_exact_labels(label):
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "selected"
        assert len(body["messages"]) == 2
        assert body["messages"][1]["content"] == "Please leave the call bend a bar"
        assert body["max_tokens"] == 12 and not body["stream"]
        return httpx.Response(200, json={"choices": [{"message": {"content": label}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)
        result = await chat.resolve_spoken_control("Please leave the call bend a bar", "selected")
    assert result == (label if label in {"leave", "stop", "pause", "none"} else None)


@pytest.mark.parametrize("error", [httpx.ReadTimeout("PRIVATE_ENDPOINT"), asyncio.CancelledError()])
async def test_resolver_timeout_is_safe_and_cancellation_propagates(error, caplog):
    def handler(request):
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)
        if isinstance(error, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await chat.resolve_spoken_control("leave the call Bender", "selected")
        else:
            assert await chat.resolve_spoken_control("leave the call Bender", "selected") is None
    assert "PRIVATE_ENDPOINT" not in caplog.text


@pytest.mark.parametrize("selected,expected", [("local", True), ("other", False)])
async def test_intent_classifier_preserves_alias_scoped_reasoning_configuration(selected, expected):
    def handler(request):
        body = json.loads(request.content)
        assert ("reasoning_effort" in body) == expected
        if expected:
            assert body["reasoning_effort"] == "none"
        return httpx.Response(200, json={"choices": [{"message": {"content": "leave"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(chat_reasoning_effort="none", chat_reasoning_models="local"), client)
        assert await chat.resolve_spoken_control("leave the call Bender", selected) == "leave"


@pytest.mark.parametrize("extra", [{"finish_reason": "length"}, {"finish_reason": "content_filter"}])
async def test_truncated_or_filtered_control_labels_are_not_executed(extra):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "leave"}, **extra}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)
        assert await chat.resolve_spoken_control("leave the call Bender", "local") is None


@pytest.mark.parametrize("revoke", [False, True])
@pytest.mark.parametrize(
    "utterance",
    [
        "Please leave the call bend a bar",
        "Get out of the call now Bentebot",
        "Would you mind hanging up?",
        "Make yourself scarce Bente",
    ],
)
async def test_resolved_leave_executes_application_action_and_rechecks_membership(store, revoke, utterance):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    channel = SimpleNamespace(id=10)
    member = SimpleNamespace(id=100, bot=False, voice=SimpleNamespace(channel=channel))
    bot.get_guild = lambda _: SimpleNamespace(get_member=lambda _: member)
    bot.voice_sessions[1] = (SimpleNamespace(source=Source("channel", 10, 1)), None, None)
    bot.leave_voice = AsyncMock()

    async def resolve(text, model):
        assert text == utterance
        if revoke:
            member.voice = None
        return "leave"

    bot.chat.resolve_spoken_control = AsyncMock(side_effect=resolve)
    try:
        assert await bot.handle_spoken_control(1, 10, 100, utterance)
        await asyncio.gather(*list(bot.voice_jobs))
        assert bot.leave_voice.await_count == (0 if revoke else 1)
        bot.chat.resolve_spoken_control.assert_awaited_once()
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_resolved_leave_is_discarded_when_speaker_resumes_during_classification(store):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    channel = SimpleNamespace(id=10)
    member = SimpleNamespace(id=100, bot=False, voice=SimpleNamespace(channel=channel))
    bot.get_guild = lambda _: SimpleNamespace(get_member=lambda _: member)
    conversation = SimpleNamespace(source=Source("channel", 10, 1), speaker_last_voice={100: 1.0})
    bot.voice_sessions[1] = (conversation, None, None)
    bot.leave_voice = AsyncMock()

    async def resolve(text, model):
        conversation.speaker_last_voice[100] = 2.0
        return "leave"

    bot.chat.resolve_spoken_control = AsyncMock(side_effect=resolve)
    try:
        assert await bot.handle_spoken_control(1, 10, 100, "Please leave the call bend a bar")
        assert not bot.voice_jobs
        bot.leave_voice.assert_not_awaited()
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_old_intent_completion_does_not_replace_newer_control_status(store):
    bot = BenteBot(Settings(health_port=0, guild_ids=frozenset({1})), store)
    channel = SimpleNamespace(id=10)
    member = SimpleNamespace(id=100, bot=False, voice=SimpleNamespace(channel=channel))
    bot.get_guild = lambda _: SimpleNamespace(get_member=lambda _: member)
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1), metrics=Metrics(), pending=True, interrupt=AsyncMock()
    )
    conversation.hold_output = AsyncMock()
    bot.voice_sessions[1] = (conversation, None, None)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def resolve(text, model):
        entered.set()
        await finish.wait()
        return None

    bot.chat.resolve_spoken_control = AsyncMock(side_effect=resolve)
    old = asyncio.create_task(bot.handle_spoken_control(1, 10, 100, "Leave the call Bender"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert conversation.metrics.control_requests == 1
        assert conversation.metrics.last_control_result == "checking"
        assert await bot.handle_spoken_control(1, 10, 100, "Stop talking")
        await asyncio.gather(*list(bot.voice_jobs))
        finish.set()
        assert not await asyncio.wait_for(old, 1)
        assert conversation.metrics.control_requests == 2
        assert conversation.metrics.last_control_action == "stop"
        assert conversation.metrics.last_control_result == "completed"
    finally:
        old.cancel()
        await asyncio.gather(old, return_exceptions=True)
        bot.voice_sessions.clear()
        await bot.close()
