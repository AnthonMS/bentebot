import asyncio
import json
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_music import Frames, fake_bot, make_session, next_started

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.domain import Metrics
from bentebot.music import MusicService
from bentebot.music_audio import SILENCE, SpeechGate
from bentebot.music_commands import handle_music_voice, owner_allowed
from bentebot.music_intent import music_candidate, music_intent
from bentebot.music_library import MusicError, Track
from bentebot.providers import OpenAIChat
from bentebot.voice import DiscordPlayback


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("Yo, can you pause the music?", "pause"),
        ("Kan du pause musik?", "pause"),
        ("Sæt musikken på pause", "pause"),
        ("We start the music again. please.", "restart_music"),
        ("Can you restart the music again please?", "restart_music"),
        ("No, can you play some Agnes Obel?", "play"),
        ("Hej, vil du spille noget jazz?", "play"),
        ("Don't pause the music", None),
        ("No, don't pause the music", None),
    ],
)
def test_reported_music_phrases_and_negation(text, action):
    intent = music_intent(text)
    assert (intent[0] if intent else None) == action


@pytest.mark.parametrize(
    "text",
    [
        "Could you put what was playing on hold for a moment?",
        "Can we get that music going again?",
        "I'd like to hear Agnes Obel, could you put her on?",
        "What is music?",
        "Don't pause the music",
    ],
)
def test_broad_candidates_reach_reasoning_without_becoming_actions(text):
    assert music_candidate(text, context=True)


@pytest.mark.parametrize(
    "plan",
    [
        {"action": "pause", "query": "", "mode": ""},
        {"action": "play", "query": "Agnes Obel", "mode": "artist"},
        {"action": "none", "query": "", "mode": ""},
        {"action": "delete", "query": "", "mode": ""},
        {"action": "play", "query": "", "mode": "artist"},
        {"action": "stop", "query": "something else", "mode": ""},
        {"action": "play", "query": "Agnes", "mode": "shell"},
        {"action": "stop", "query": "", "mode": "", "shell": "anything"},
    ],
)
async def test_semantic_music_proposals_are_validated_and_only_use_current_request(plan):
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "selected" and not body["stream"]
        assert len(body["messages"]) == 2 and "tools" not in body
        data = json.loads(body["messages"][1]["content"])
        assert data == {"request": "Put the music on hold", "playback": {"active": True}}
        assert body["max_tokens"] == 180
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)
        result = await chat.resolve_music_control("Put the music on hold", "selected", {"active": True})
    valid = plan in [
        {"action": "pause", "query": "", "mode": ""},
        {"action": "play", "query": "Agnes Obel", "mode": "artist"},
        {"action": "none", "query": "", "mode": ""},
    ]
    assert result == ((plan["action"], plan["query"], plan["mode"]) if valid else None)


async def test_music_resolver_failure_is_quiet_and_cancellation_propagates(caplog):
    def handler(request):
        raise httpx.ReadTimeout("PRIVATE_ENDPOINT_AND_TOKEN")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(Settings(), client)
        assert await chat.resolve_music_control("put music on", "selected", {}) is None
    assert "PRIVATE_ENDPOINT_AND_TOKEN" not in caplog.text


@pytest.mark.parametrize("prefix", ["", "I will run this action: "])
async def test_exact_json_fence_supported_but_explanatory_prose_is_not_executable(prefix):
    content = prefix + '```json\n{"action":"pause","query":"","mode":""}\n```'
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        )
    ) as client:
        result = await OpenAIChat(Settings(), client).resolve_music_control("Pause music", "selected", {})
    assert result == (None if prefix else ("pause", "", ""))


@pytest.mark.parametrize("stale", [False, True])
async def test_model_interpreted_action_executes_only_for_current_authorized_turn(tmp_path, stale):
    session, playback = make_session(tmp_path)
    bot = fake_bot()
    bot.voice_sessions[5][0].metrics = Metrics()
    bot.voice_sessions[5][0].speech_revision = {1: 10}
    bot.music = SimpleNamespace(sessions={5: session}, session=lambda *args: session)
    bot.model = lambda *args: "selected"

    async def resolve(*args):
        if stale:
            bot.voice_sessions[5][0].speech_revision[1] += 1
        return "pause", "", ""

    bot.chat = SimpleNamespace(resolve_music_control=AsyncMock(side_effect=resolve))
    try:
        await session.start("artist", "artist")
        await next_started(playback)
        assert await handle_music_voice(bot, 5, 10, 1, "Could you put what was playing on hold for a moment?")
        bot.chat.resolve_music_control.assert_awaited_once()
        assert playback.paused == (not stale)
        metrics = bot.voice_sessions[5][0].metrics
        assert metrics.last_control_method == "semantic_music"
        assert metrics.last_control_result == ("stale" if stale else "accepted")
    finally:
        await session.stop()


async def test_track_notifications_once_and_stopped_playlist_can_resume(tmp_path):
    session, playback = make_session(tmp_path)
    try:
        await session.start("artist", "artist")
        first = await next_started(playback)
        await session.control("next")
        second = await next_started(playback)
        assert second != first
        assert session.notify.await_count == 1
        await session.control("stop")
        assert session.task is None and not session.prepared
        await session.control("resume")
        assert await next_started(playback) == second
        await session.control("stop")
        await session.control("restart_music")
        assert await next_started(playback) == first
    finally:
        await session.stop()


def test_noise_never_ducks_without_recognition_and_words_need_audible_sustained_speech():
    now = [10.0]
    gate = SpeechGate(clock=lambda: now[0])
    loud = struct.pack("<h", 1000) * 320
    quiet = struct.pack("<h", 10) * 320
    for _ in range(40):
        assert not gate.observe(1, loud, True)
    assert gate.confirm(1)
    assert gate.observe(1, loud, True)
    for _ in range(20):
        assert not gate.observe(2, quiet, True)
    assert not gate.confirm(2)
    for _ in range(12):
        assert not gate.observe(1, SILENCE[:640], False)
    now[0] += 3
    assert not gate.observe(1, loud, True)


class FakeClient:
    def __init__(self):
        self.playing = self.paused = False
        self.starts = self.pauses = self.resumes = 0

    def is_playing(self):
        return self.playing and not self.paused

    def is_paused(self):
        return self.paused

    def play(self, source, after, **kwargs):
        assert kwargs["signal_type"] == "music" and kwargs["bitrate"] == 192
        self.starts += 1
        self.playing = True

    def pause(self):
        self.paused = True
        self.pauses += 1

    def resume(self):
        self.paused = False
        self.resumes += 1

    def stop_playing(self):
        self.playing = self.paused = False


def test_idle_mixer_pauses_transmission_and_reuses_output_when_music_or_speech_returns():
    client = FakeClient()
    playback = DiscordPlayback(client, mixed=True)
    playback.mixer.set_music(Frames(), lambda _: None)
    playback._start_mixer()
    playback.pause_music(True)
    assert client.paused and not playback.mixer.closed
    playback.pause_music(False)
    assert client.starts == 1 and client.resumes == 1
    playback.set_listener_presence(False)
    assert client.paused
    playback.set_listener_presence(True)
    assert not client.paused
    playback.mixer.take_music()
    playback._sync_mixer()
    assert client.paused
    playback.mixer.set_speech(Frames(), lambda _: None)
    playback._start_mixer()
    assert not client.paused and client.starts == 1
    playback.stop()
    assert client.paused
    playback.close()


async def test_maintenance_and_artist_choices_work_without_bot_voice_session(store):
    bot = BenteBot(
        Settings(health_port=0, music_enabled=True, owner_ids=frozenset({1}), guild_ids=frozenset({5})), store
    )
    bot.music.rescan = AsyncMock(return_value=42)
    bot.music.download = AsyncMock(return_value=Track("Agnes Obel", "Riverside"))
    bot.music.library.scanned = True
    bot.music.library.tracks = [Track("Agnes Obel", "Riverside"), Track("Bruno Mars", "Treasure")]
    interaction = SimpleNamespace(
        guild_id=5,
        user=SimpleNamespace(id=1),
        namespace=SimpleNamespace(mode="artist"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    group = bot.tree.get_command("music")
    try:
        assert owner_allowed(bot, 5, 1) and not bot.voice_sessions
        await group.get_command("rescan").callback(interaction)
        bot.music.rescan.assert_awaited_once()
        await group.get_command("download").callback(interaction, "Agnes Obel", "Riverside")
        bot.music.download.assert_awaited_once_with("Agnes Obel", "Riverside")
        choices = await group.get_command("play")._params["query"].autocomplete(interaction, "ag")
        assert [(c.name, c.value) for c in choices] == [("Agnes Obel", "Agnes Obel")]
        interaction.user.id = 2
        assert not await group.get_command("play")._params["query"].autocomplete(interaction, "")
        assert not bot.voice_sessions
    finally:
        await bot.close()


async def test_maintenance_bounds_jobs_and_shuts_down_cleanly(tmp_path):
    service = MusicService(Settings(music_library_path=str(tmp_path)))
    gate = asyncio.Event()
    service.library.scan = AsyncMock(side_effect=gate.wait)
    first = asyncio.create_task(service.rescan())
    second = asyncio.create_task(service.rescan())
    await asyncio.sleep(0)
    try:
        with pytest.raises(MusicError, match="busy"):
            await service.rescan()
        await service.close()
        assert first.cancelled() and second.cancelled() and not service.jobs
    finally:
        await asyncio.gather(first, second, return_exceptions=True)


async def test_music_play_joins_call_automatically_while_maintenance_does_not(store):
    bot = BenteBot(
        Settings(health_port=0, music_enabled=True, owner_ids=frozenset({1}), guild_ids=frozenset({5})), store
    )
    user = SimpleNamespace(id=1, bot=False, voice=SimpleNamespace(channel=SimpleNamespace(id=10)))
    interaction = SimpleNamespace(
        guild_id=5,
        user=user,
        channel=object(),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    session = SimpleNamespace(commands=asyncio.Lock(), start=AsyncMock())
    bot.music.session = lambda *args: session
    bot.get_guild = lambda _: SimpleNamespace(get_member=lambda _: user)

    async def join(*args):
        bot.voice_sessions[5] = (SimpleNamespace(source=SimpleNamespace(channel_id=10)), object(), object())

    bot.join_voice = AsyncMock(side_effect=join)
    try:
        await bot.tree.get_command("music").get_command("play").callback(interaction, "Agnes Obel", "artist")
        bot.join_voice.assert_awaited_once_with(user, interaction.channel)
        session.start.assert_awaited_once_with("Agnes Obel", "artist")
    finally:
        bot.voice_sessions.clear()
        await bot.close()
