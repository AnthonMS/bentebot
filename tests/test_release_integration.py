import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Source, Transcript
from bentebot.voice_profiles import VoiceProfile


def target_and_call():
    target = SimpleNamespace(
        id=100,
        bot=False,
        guild=SimpleNamespace(id=1),
        voice=SimpleNamespace(channel=SimpleNamespace(id=10)),
    )
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1),
        paused=False,
        hold_replies=0,
        suppressed_users=set(),
        interrupt=AsyncMock(),
    )
    return target, conversation


@pytest.mark.parametrize("revoke", ["leave", "pause"])
async def test_cloning_revoked_during_upload_deletes_returned_profile(store, revoke):
    bot = BenteBot(Settings(health_port=0, voice_clone_speech_seconds=30), store)
    target, conversation = target_and_call()
    bot.voice_sessions[1] = (conversation, None, None)
    uploaded, release = asyncio.Event(), asyncio.Event()
    profile = VoiceProfile("voice_" + "a" * 32, "Mine", True)

    async def create(*args, **kwargs):
        uploaded.set()
        await release.wait()
        return profile

    bot.profiles = SimpleNamespace(
        create=AsyncMock(side_effect=create), delete=AsyncMock(), close=AsyncMock()
    )
    notify = AsyncMock()
    try:
        await bot.begin_enrollment(target, "Mine", notify)
        assert conversation.hold_replies == 0 and conversation.suppressed_users == set()
        conversation.interrupt.assert_not_awaited()
        for _ in range(1500):
            bot.enrollments.feed(100, 10, b"\x11\x01" * 320, True)
        await asyncio.wait_for(uploaded.wait(), 1)
        if revoke == "leave":
            target.voice.channel = None
        else:
            conversation.paused = True
        jobs = list(bot.voice_jobs)
        release.set()
        await asyncio.wait_for(asyncio.gather(*jobs), 2)
        bot.profiles.delete.assert_awaited_once_with(profile.id)
        assert bot.profile_info(profile.id) is None
        assert store.count() == 0
        assert any("removed" in call.args[0] for call in notify.await_args_list)
    finally:
        release.set()
        bot.voice_sessions.clear()
        await bot.close()


async def test_assembled_clone_metadata_and_logs_identify_sample_without_audio(store, caplog):
    bot = BenteBot(Settings(health_port=0, voice_clone_speech_seconds=30), store)
    target, conversation = target_and_call()
    bot.voice_sessions[1] = (conversation, None, None)
    profile = VoiceProfile("voice_" + "b" * 32, "PRIVATE_PROFILE_NAME", True)
    bot.profiles = SimpleNamespace(
        create=AsyncMock(return_value=profile), delete=AsyncMock(), close=AsyncMock()
    )
    try:
        with caplog.at_level(logging.INFO, logger="bentebot.discord_app"):
            await bot.begin_enrollment(
                target, "PRIVATE_PROFILE_NAME", AsyncMock(), consent_basis="server_policy"
            )
            for _ in range(15):
                for _ in range(100):
                    bot.enrollments.feed(100, 10, b"\x11\x01" * 320, True)
                for _ in range(20):
                    bot.enrollments.feed(100, 10, bytes(640), False)
            await asyncio.wait_for(asyncio.gather(*list(bot.voice_jobs)), 2)
        metadata = bot.profile_info(profile.id)
        assert metadata["reference_spliced"] is True and metadata["reference_segment_count"] > 1
        assert metadata["consent_basis"] == "server_policy"
        assert "spliced=True" in caplog.text and "outcome=saved" in caplog.text
        assert "PRIVATE_PROFILE_NAME" not in caplog.text
        assert store.count() == 0 and bot.enrollments.active_count == 0
    finally:
        bot.voice_sessions.clear()
        await bot.close()


@pytest.mark.parametrize("utterance", ["Bentebot, stop", "Can you stop?", "Be quiet"])
async def test_short_spoken_stop_interrupts_an_active_answer_before_word_threshold(store, utterance):
    stop = Mock()
    conversation = VoiceConversation(
        Settings(interruption_words=5),
        store,
        None,
        None,
        None,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        stop,
        AsyncMock(),
    )
    task = asyncio.create_task(asyncio.Event().wait())
    conversation.task = task
    conversation.speech_activity(100)
    try:
        await conversation.transcript(100, "Me", Transcript(utterance))
        assert task.cancelled()
        stop.assert_called_once()
        assert conversation.metrics.interruptions == 1
    finally:
        await conversation.close()
