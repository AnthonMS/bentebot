import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import httpx
import pytest
from aiohttp import web
from PIL import Image

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience, Event, Metrics, Source
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy
from bentebot.voice_profiles import VoiceProfile


async def yes(*args):
    return True


async def test_real_screen_upload_enters_vision_prompt_but_not_database(store, unused_tcp_port, wav_bytes):
    bot = BenteBot(Settings(screen_enabled=True, vision_model="vision-local", health_port=8080), store)
    bot._connection.user = SimpleNamespace(id=999)
    owner = SimpleNamespace(id=100)
    channel = MagicMock(spec=discord.VoiceChannel)
    channel.id, channel.members = 10, [owner, bot.user]
    guild = SimpleNamespace(id=1, get_channel=lambda id: channel if id == 10 else None)
    channel.guild = guild
    bot.get_guild = lambda id: guild if id == 1 else None
    bot.policy = AccessPolicy(yes)
    bot.screen_awareness.policy = bot.policy
    source = Source("channel", 10, 1)

    async def recipients():
        return Audience(frozenset({100}), 1)

    prompts = []

    async def stream(messages, model):
        prompts.append((messages, model))
        yield "I can see a blue window."

    async def synth(text, voice):
        return wav_bytes

    async def play(pcm, text, progress):
        progress(text)

    async def notify(text):
        raise AssertionError(text)

    memory = Memory(store, bot.policy, 10000)
    conversation = VoiceConversation(
        bot.settings,
        store,
        memory,
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synth),
        source,
        999,
        recipients,
        play,
        lambda: None,
        notify,
        turn_context=lambda audience, query, at: bot.visual_context(10, audience, query, at),
        extra_guard=bot.screens.sources_active,
    )
    bot.voice_sessions[1] = (conversation, None, None)
    app = web.Application()
    bot.screens.install(app)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    try:
        grant = bot.screens.issue_pair(100, 1, 10, frozenset({100}))
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{unused_tcp_port}") as client:
            response = await client.post("/screen/pair", json={"code": grant.code})
            assert response.status_code == 200
            token = response.json()["token"]
            image = io.BytesIO()
            Image.new("RGB", (320, 200), "blue").save(image, "JPEG")
            response = await client.post(
                "/screen/frame",
                content=image.getvalue(),
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "image/jpeg",
                    "X-Bentebot-Sequence": "1",
                },
            )
            assert response.status_code == 200
            store.save(Event(source, 100, "Me", "What do you see?", kind="voice"))
            conversation.latest_request = "What do you see?"
            await conversation._answer()
            assert prompts[0][1] == "vision-local"
            assert "data:image/jpeg;base64," in str(prompts[0][0])
            answer = store.recent(1, frozenset())[-1]
            assert any(origin.kind == "screen" for origin in answer.provenance)
            for (payload,) in store.db.execute("SELECT payload FROM events"):
                decoded = store._decode(payload)
                assert "data:image" not in repr(decoded) and "base64" not in repr(decoded)
            assert len((await memory.context(await recipients()))[0]) > 1
            assert "blue window" not in str((await memory.context(Audience(frozenset({200}), 1)))[0])
            live = tuple(origin for origin in answer.provenance if origin.kind == "screen")
            assert await bot.screens.sources_active(live)
            await client.post("/screen/stop", headers={"Authorization": "Bearer " + token})
            assert not await bot.screens.sources_active(live)
            assert (await bot.visual_context(10, await recipients())) == ([], (), None)
    finally:
        await runner.cleanup()
        bot.voice_sessions.clear()
        await bot.close()


async def test_voice_enrollment_job_saves_owner_but_no_audio_history(store):
    bot = BenteBot(Settings(health_port=0), store)
    bot.profiles = SimpleNamespace(
        create=AsyncMock(return_value=VoiceProfile("voice_" + "a" * 32, "Mine", True)),
        delete=AsyncMock(),
        close=AsyncMock(),
    )
    channel = SimpleNamespace(id=10)
    target = SimpleNamespace(
        id=100, bot=False, guild=SimpleNamespace(id=1), voice=SimpleNamespace(channel=channel)
    )
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1),
        paused=False,
        hold_replies=0,
        suppressed_users=set(),
        changed=asyncio.Event(),
        interrupt=AsyncMock(),
        metrics=Metrics(),
    )
    bot.voice_sessions[1] = (conversation, None, None)
    notices = []

    async def notify(text):
        notices.append(text)

    try:
        await bot.begin_enrollment(target, "Mine", notify)
        assert conversation.hold_replies == 0 and not conversation.suppressed_users
        conversation.interrupt.assert_not_awaited()
        for _ in range(45 * 50):
            bot.enrollments.feed(200, 10, b"\x77\x07" * 320, True)
            bot.enrollments.feed(100, 10, b"\x11\x01" * 320, True)
        await asyncio.gather(*list(bot.voice_jobs))
        sample = bot.profiles.create.call_args.args[1]
        assert sample.startswith(b"RIFF") and b"\x77\x07" * 100 not in sample
        assert bot.profiles.create.call_args.kwargs["consent_confirmed"] is True
        assert bot.profile_info("voice_" + "a" * 32)["owner_id"] == 100
        assert store.count() == 0 and conversation.hold_replies == 0
        assert bot.enrollments.active_count == 0 and any("saved" in text for text in notices)
    finally:
        bot.voice_sessions.clear()
        await bot.close()


@pytest.mark.parametrize("notice_fails", [True, False])
async def test_enrollment_rechecks_membership_and_cleans_failed_start_notice(store, notice_fails):
    bot = BenteBot(Settings(health_port=0), store)
    bot.profiles = SimpleNamespace(close=AsyncMock())
    target = SimpleNamespace(
        id=100, bot=False, guild=SimpleNamespace(id=1), voice=SimpleNamespace(channel=SimpleNamespace(id=10))
    )
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1),
        paused=False,
        hold_replies=0,
        suppressed_users=set(),
        changed=asyncio.Event(),
        interrupt=AsyncMock(),
    )
    if not notice_fails:
        target.voice.channel = None
    bot.voice_sessions[1] = (conversation, None, None)
    try:
        with pytest.raises(RuntimeError if notice_fails else ValueError):
            await bot.begin_enrollment(target, "Mine", AsyncMock(side_effect=RuntimeError("no notice")))
        assert conversation.hold_replies == 0 and conversation.suppressed_users == set()
        assert bot.enrollments.active_count == 0 and not bot.voice_jobs
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_receiver_reconfiguration_preserves_hooks_and_language_and_serializes_join(store):
    bot = BenteBot(Settings(health_port=0), store)
    old = SimpleNamespace(client=object(), notify=AsyncMock(), close=AsyncMock())
    conversation = SimpleNamespace(
        source=Source("channel", 10, 1),
        paused=True,
        audio=None,
        output_held=False,
        pending=True,
        partial={100: "unfinished"},
        speaker_started={100: 0},
        speaker_clock={},
        speaker_last_voice={},
        metrics=Metrics(),
        changed=asyncio.Event(),
        interrupt=AsyncMock(),
    )
    bot.voice_sessions[1] = (conversation, old, None)
    created = []
    original = bot._replacement_receiver

    def replacement(guild_id, conversation, previous):
        value = original(guild_id, conversation, previous)
        value.start = lambda: None
        value.close = AsyncMock()
        created.append(value)
        return value

    bot._replacement_receiver = replacement
    try:
        assert await bot.resume_voice(1)
        assert created[-1].on_frame == bot.enrollments.feed
        store.set("guild:1", "language", "en")
        assert created[-1].language() == "auto"
        await bot.pause_voice(1)
        assert not conversation.partial and not conversation.pending
        assert conversation.paused
        maximum, active = 0, 0

        async def join(member, channel):
            nonlocal maximum, active
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "ok"

        bot._join_voice = join
        member = SimpleNamespace(guild=SimpleNamespace(id=2))
        await asyncio.gather(bot.join_voice(member, None), bot.join_voice(member, None))
        assert maximum == 1
    finally:
        bot.voice_sessions.clear()
        await bot.close()
