"""Real CPU codec checks using generated audio, never network media or Discord."""

import asyncio
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from bentebot.music_audio import BufferedMusic
from bentebot.music_download import audio_signature
from bentebot.music_library import Library, Track
from bentebot.voice import DiscordPlayback

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required for codec integration")


def generate(path):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "2",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
        timeout=15,
    )


@pytest.mark.parametrize("extension", ["wav", "flac", "mp3"])
async def test_real_decoder_frame_clock_and_cleanup(tmp_path, extension):
    path = tmp_path / ("generated." + extension)
    await asyncio.to_thread(generate, path)
    source = await asyncio.to_thread(BufferedMusic, path)
    frames = []
    try:
        while frame := await asyncio.to_thread(source.frames.get, True, 3):
            frames.append(frame)
    finally:
        await asyncio.to_thread(source.cleanup)
    assert len(frames) == 100 and all(len(frame) == 3840 for frame in frames)
    assert not source.thread.is_alive()


async def test_real_mp3_import_and_rescan_reuses_source_id(tmp_path):
    staging = tmp_path / "generated.mp3"
    await asyncio.to_thread(generate, staging)
    original_audio = audio_signature(staging.read_bytes())
    library = Library(str(tmp_path))
    track = await library.import_file(
        staging, Track("Synthetic Artist", "Generated sine", "jazz", video_id="abcdefghijk")
    )
    await asyncio.to_thread(staging.unlink)
    from pathlib import Path

    assert audio_signature(await asyncio.to_thread(Path(track.path).read_bytes)) == original_audio
    await library.scan()
    assert library.select("jazz", "radio")[0].path == track.path
    assert library.existing(Track("other", "title", video_id="abcdefghijk")).path == track.path


async def test_real_mixed_playback_finishes_music_and_preserves_speech(tmp_path):
    path = tmp_path / "generated.flac"
    await asyncio.to_thread(generate, path)
    pumps = []
    stopped = False
    started = asyncio.Event()

    async def pump(source):
        while not stopped and source.read():  # noqa: ASYNC110 - accelerated Discord audio clock
            await asyncio.sleep(0.001)

    def play(source, after, **kwargs):
        pumps.append(asyncio.create_task(pump(source)))
        started.set()

    client = SimpleNamespace(
        is_playing=lambda: bool(pumps),
        play=play,
        stop_playing=lambda: None,
        pause=lambda: None,
    )
    playback = DiscordPlayback(client, mixed=True)
    music = asyncio.create_task(playback.play_music(path))
    try:
        await asyncio.wait_for(started.wait(), 3)
        words = []
        await asyncio.wait_for(playback.play(b"\1\0" * 1920 * 5, "a spoken sentence", words.append), 3)
        assert words[-1] == "a spoken sentence" and not music.done()
        await asyncio.wait_for(music, 5)
        assert not playback.mixer.closed
    finally:
        stopped = True
        music.cancel()
        await asyncio.gather(music, *pumps, return_exceptions=True)
        playback.close()
