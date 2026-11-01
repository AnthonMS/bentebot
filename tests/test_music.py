import asyncio
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from bentebot.config import Settings
from bentebot.music import MusicSession, music_intent
from bentebot.music_audio import SILENCE, Mixer
from bentebot.music_commands import authorized, handle_music_voice
from bentebot.music_download import MeTube, run_metadata
from bentebot.music_library import Catalog, Library, MusicError, Track, component
from bentebot.voice import DiscordPlayback, TrackedPCM


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Play some Agnes Obel", ("play", "agnes obel", "artist")),
        ("Bente, play some RNB music.", ("play", "rnb", "radio")),
        ("Can you play songs by Agnes Obel?", ("play", "agnes obel", "artist")),
        ("Spil noget RNB musik", ("play", "rnb", "radio")),
        ("Kan du spille noget musik", ("play", "musik", "artist")),
        ("play Riverside", ("play", "riverside", "auto")),
        ("pause the music", ("pause", "", "")),
        ("pause music", ("pause", "", "")),
        ("pause musikken", ("pause", "", "")),
        ("stop musikken", ("stop", "", "")),
        ("resume the music", ("resume", "", "")),
        ("fortsæt musikken", ("resume", "", "")),
        ("play the next song", ("next", "", "")),
        ("næste nummer", ("next", "", "")),
        ("play the previous song", ("previous", "", "")),
        ("genstart sang", ("restart", "", "")),
        ("restart this song", ("restart", "", "")),
        ("Don't pause the music", None),
        ("Do not play Agnes Obel", None),
        ("He said play some rock", None),
        ('Play "some rock" as an example', None),
        ("What does pause music mean?", None),
        ("stop", None),
        ("pause", None),
        ("play", None),
        ("play devil's advocate", None),
        ("play the role of a friend", None),
    ],
)
def test_explicit_music_intent(utterance, expected):
    assert music_intent(utterance) == expected


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48000)
        output.writeframes(b"\0" * 3840)
    return path


async def test_library_fallback_metadata_and_order_independent_search(tmp_path):
    path = wav(tmp_path / "Agnes Obel" / "Riverside.wav")
    wav(tmp_path / ".hidden" / "private.wav")
    library = Library(str(tmp_path))
    assert await library.scan() == 1
    assert library.tracks[0].artist == "Agnes Obel"
    assert library.select("Riverside by Agnes Obel", "track")[0].path == str(path)
    assert library.select("Agnes Obel", "artist") == library.tracks
    assert library.existing(Track("AGNES OBEL", "riverside")) == library.tracks[0]


async def test_missing_library_fails_actionably(tmp_path):
    with pytest.raises(MusicError, match="mount"):
        await Library(str(tmp_path / "absent")).scan()


async def test_external_library_addition_is_found_before_downloading(tmp_path):
    library = Library(str(tmp_path))
    await library.scan()
    path = wav(tmp_path / "artist" / "title.wav")
    downloader = MeTube("http://metube.test", library)
    try:
        # The index was empty before another program added this recording.
        result = await downloader.prepare(Track("artist", "title", video_id="abcdefghijk"))
        assert result.path == str(path)
    finally:
        await downloader.close()


def test_library_rejects_traversal_and_unsafe_components(tmp_path):
    library = Library(str(tmp_path))
    with pytest.raises(MusicError):
        library.safe(tmp_path / ".." / "outside.mp3")
    for unsafe in ["../evil", "CON", 'a/b:c"d', "..", "NUL", "LPT1"]:
        assert not any(c in component(unsafe) for c in '/\\:"')
        assert component(unsafe) not in {"..", "CON", "NUL", "LPT1"}


async def test_atomic_import_no_overwrite_and_duplicate_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("mutagen.mp3.MP3", lambda path: SimpleNamespace(info=SimpleNamespace(length=20)))
    library = Library(str(tmp_path))
    original = tmp_path / ".incoming"
    original.write_bytes(b"a" * 2048)
    track = Track("Agnes Obel", "Riverside", video_id="abcdefghijk")
    result = await library.import_file(original, track)
    contents = await asyncio.to_thread(Path(result.path).read_bytes)
    assert Path(result.path).parent.name == "Agnes Obel"
    again = tmp_path / ".incoming2"
    again.write_bytes(b"b" * 2048)
    duplicate = await library.import_file(again, Track("different", "title", video_id="abcdefghijk"))
    assert duplicate == result and await asyncio.to_thread(Path(result.path).read_bytes) == contents
    assert len(library.tracks) == 1


async def test_import_rejects_html_before_publishing(tmp_path):
    library = Library(str(tmp_path))
    staging = tmp_path / ".bad"
    staging.write_bytes(b"<html>not audio</html>" * 100)
    with pytest.raises(MusicError, match="MP3"):
        await library.import_file(staging, Track("a", "b", video_id="abcdefghijk"))
    assert not list(tmp_path.glob("**/*.mp3"))


def test_genre_alias_and_source_id_dedup(tmp_path):
    path = wav(tmp_path / "artist" / "song.wav")
    library = Library(str(tmp_path))
    track = Track("artist", "song", "R&B / Soul", str(path), "abcdefghijk")
    library.tracks = [track]
    assert library.select("rnb", "radio") == [track]
    assert library.existing(Track("other", "name", video_id="abcdefghijk")) == track


async def test_metube_single_job_import_and_history_reuse(tmp_path, monkeypatch):
    library = Library(str(tmp_path))
    downloader = MeTube("http://metube.test", library)
    calls = []
    submitted = False
    deleted = False
    track = Track("artist", "title", video_id="abcdefghijk")
    monkeypatch.setattr("mutagen.mp3.MP3", lambda path: SimpleNamespace(info=SimpleNamespace(length=20)))
    monkeypatch.setattr(downloader, "cleanup_mode", AsyncMock())

    async def handler(request):
        nonlocal submitted, deleted
        calls.append((request.method, request.url.path))
        if request.url.path == "/history":
            rows = (
                [
                    {
                        "url": "https://www.youtube.com/watch?v=abcdefghijk",
                        "id": "abcdefghijk",
                        "format": "mp3",
                        "download_type": "audio",
                        "status": "finished",
                        "filename": "artist song.mp3",
                    }
                ]
                if submitted and not deleted
                else []
            )
            return httpx.Response(200, json={"done": rows, "queue": [], "pending": []})
        if request.url.path == "/add":
            import json

            data = json.loads(request.content)
            assert data["playlist_item_limit"] == 1 and data["format"] == "mp3"
            assert "list=" not in data["url"] and data["quality"] == "192"
            submitted = True
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/delete":
            import json

            assert json.loads(request.content) == {
                "ids": ["https://www.youtube.com/watch?v=abcdefghijk"],
                "where": "done",
                "delete_files": True,
            }
            assert library.existing(track) is not None
            deleted = True
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/audio_download/artist song.mp3"
        return httpx.Response(404 if deleted else 200, content=b"a" * 2048)

    await downloader.http.aclose()
    downloader.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await downloader.prepare(track)
        assert result.video_id == track.video_id and await asyncio.to_thread(Path(result.path).is_file)
        assert deleted and not list(tmp_path.rglob("*.part"))
        await downloader.prepare(track)
        assert sum(method == "POST" for method, _ in calls) == 2
        assert not any(path in {"/cancel-add", "/start"} for _, path in calls)
    finally:
        await downloader.close()


@pytest.mark.parametrize(
    "filename", ["../outside.mp3", "/absolute.mp3", "x/../../z.mp3", "audio.m3u", "a\\..\\b.mp3"]
)
async def test_metube_untrusted_filename_never_requested(tmp_path, filename):
    downloader = MeTube("http://metube.test", Library(str(tmp_path)))

    async def handler(request):
        assert request.url.path == "/history"
        return httpx.Response(
            200,
            json={
                "done": [
                    {
                        "url": "https://www.youtube.com/watch?v=abcdefghijk",
                        "format": "mp3",
                        "download_type": "audio",
                        "status": "finished",
                        "filename": filename,
                    }
                ]
            },
        )

    await downloader.http.aclose()
    downloader.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(MusicError, match="filename"):
            await downloader.prepare(Track("a", "b", video_id="abcdefghijk"))
    finally:
        await downloader.close()


async def test_catalog_filters_live_mismatched_artist_and_compilations(monkeypatch):
    catalog = Catalog()
    monkeypatch.setattr(
        "bentebot.music_download.run_metadata",
        AsyncMock(
            return_value={
                "entries": [
                    {"id": "abcdefghij1", "title": "Other Person - Riverside", "duration": 200},
                    {
                        "id": "abcdefghij2",
                        "title": "Agnes Obel - Riverside",
                        "duration": 200,
                        "is_live": True,
                    },
                    {"id": "abcdefghij3", "title": "Agnes Obel - Riverside", "duration": 800},
                    {
                        "id": "abcdefghij4",
                        "title": "Agnes Obel - Riverside (Official Audio)",
                        "duration": 200,
                    },
                ]
            }
        ),
    )
    try:
        result = await catalog.video(Track("Agnes Obel", "Riverside", duration=195))
        assert result.video_id == "abcdefghij4"
    finally:
        await catalog.close()


class Frames:
    def __init__(self, amplitude=1000):
        self.frame = struct.pack("<h", amplitude) * 1920
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.frame

    def cleanup(self):
        pass


async def test_mixer_ducks_preserves_music_through_speech_stop_and_freezes_when_empty():
    mixer = Mixer(volume=1)
    music = Frames(1000)
    mixer.set_music(music, lambda _: None)
    assert struct.unpack_from("<h", mixer.read())[0] == 1000
    mixer.speaking()
    ducked = [struct.unpack_from("<h", mixer.read())[0] for _ in range(30)]
    assert 390 <= ducked[-1] <= 420
    done = []
    speech = TrackedPCM(b"\1\0" * 1920, "one", lambda _: None, asyncio.get_running_loop())
    mixer.set_speech(speech, done.append)
    mixer.read()
    mixer.stop_speech()
    assert done == [None] and mixer.music is music
    before = music.reads
    mixer.occupied = False
    assert mixer.read() == SILENCE and music.reads == before
    mixer.occupied = True
    mixer.paused = True
    mixer.read()
    assert music.reads == before
    mixer.paused = False
    mixer.read()
    assert music.reads == before + 1


async def test_mixed_playback_speech_cancel_does_not_stop_discord_music():
    stops = []
    client = SimpleNamespace(playing=False, stop_playing=lambda: stops.append(True))

    def play(source, after, **kwargs):
        client.playing = True

    client.play, client.is_playing = play, lambda: client.playing
    playback = DiscordPlayback(client, mixed=True)
    playback.mixer.set_music(Frames(), lambda _: None)
    task = asyncio.create_task(playback.play(b"\0" * 3840, "hello", lambda _: None))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not stops and playback.mixer.music is not None
    playback.stop()
    assert not stops
    playback.close()
    assert stops == [True]


class FakePlayback:
    def __init__(self):
        self.started = asyncio.Queue()
        self.paused = False
        self.finishes = []

    async def play_music(self, path):
        self.started.put_nowait(path)
        future = asyncio.get_running_loop().create_future()
        self.finishes.append(future)
        await future

    def pause_music(self, value):
        self.paused = value


def make_session(tmp_path, n=4):
    library = Library(str(tmp_path))
    library.tracks = [
        Track("artist", f"song {i}", path=str(wav(tmp_path / "artist" / f"song{i}.wav"))) for i in range(n)
    ]
    library.scanned = True
    service = SimpleNamespace(
        library=library, settings=Settings(), prepare=AsyncMock(side_effect=lambda t: t)
    )
    playback = FakePlayback()
    return MusicSession(service, playback, AsyncMock()), playback


async def next_started(playback):
    return await asyncio.wait_for(playback.started.get(), 2)


async def test_playlist_next_previous_restart_loop_and_bounded_prefetch(tmp_path):
    session, playback = make_session(tmp_path)
    try:
        await session.start("artist", "artist")
        first = await next_started(playback)
        assert first.endswith("song0.wav") and len(session.prepared) <= 3
        await session.control("next")
        assert (await next_started(playback)).endswith("song1.wav")
        await session.control("restart")
        assert (await next_started(playback)).endswith("song1.wav")
        await session.control("previous")
        assert await next_started(playback) == first
        await session.control("previous")
        assert (await next_started(playback)).endswith("song3.wav")
        playback.finishes[-1].set_result(None)
        assert await next_started(playback) == first
        await session.control("pause")
        assert playback.paused
        await session.control("resume")
        assert not playback.paused
    finally:
        await session.stop()
    assert not session.prepared and not session.audio_task


async def test_skip_while_preparing_does_not_play_stale_song(tmp_path):
    session, playback = make_session(tmp_path)
    blocked = asyncio.Event()

    async def prepare(track):
        if track.title == "song 0":
            await blocked.wait()
        return track

    session.service.prepare = prepare
    try:
        await session.start("artist", "artist")
        for _ in range(10):
            await asyncio.sleep(0)
        await session.control("next")
        assert (await next_started(playback)).endswith("song1.wav")
    finally:
        await session.stop()
    assert not blocked.is_set() and not session.prepared


async def test_failures_stop_instead_of_spinning_and_closed_session_cannot_restart(tmp_path):
    session, _ = make_session(tmp_path)
    session.service.prepare = AsyncMock(side_effect=MusicError("unavailable"))
    await session.start("artist", "artist")
    await asyncio.wait_for(session.task, 2)
    assert session.error == "unavailable" and not session.prepared
    session.closed = True
    with pytest.raises(MusicError, match="ended"):
        await session.start("artist", "artist")


def fake_bot(owner=1):
    call = (SimpleNamespace(source=SimpleNamespace(channel_id=10)), None, None)
    member = SimpleNamespace(bot=False, voice=SimpleNamespace(channel=SimpleNamespace(id=10)))
    guild = SimpleNamespace(get_member=lambda user: member)
    return SimpleNamespace(
        music=object(),
        settings=Settings(owner_ids=frozenset({owner}), guild_ids=frozenset({5})),
        _closing=False,
        voice_sessions={5: call},
        get_guild=lambda _: guild,
    )


def test_music_requires_owner_current_call_and_allowed_server():
    bot = fake_bot()
    assert authorized(bot, 5, 1)
    assert not authorized(bot, 5, 2)
    assert not authorized(bot, 6, 1)
    bot._closing = True
    assert not authorized(bot, 5, 1)


async def test_unprivileged_spoken_control_is_consumed_without_action():
    bot = fake_bot()
    assert await handle_music_voice(bot, 5, 10, 2, "stop music")
    assert not await handle_music_voice(bot, 5, 10, 2, "stop")


async def test_metadata_cancel_kills_owned_subprocess(monkeypatch):
    killed = []

    async def blocked(*args):
        await asyncio.Event().wait()

    process = SimpleNamespace(
        returncode=None,
        stdout=SimpleNamespace(read=blocked),
        kill=lambda: killed.append(True),
        wait=AsyncMock(),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    task = asyncio.create_task(run_metadata("some song"))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert killed == [True]
    assert process.wait.await_count == 1


@pytest.mark.parametrize(
    "url", ["file:///tmp/music", "http://user:pass@host", "http://host?a=1", "http://host/#x"]
)
def test_music_configuration_rejects_credential_urls(url):
    with pytest.raises(ValueError, match="music_metube_url"):
        Settings(music_metube_url=url).validate(login=False)
