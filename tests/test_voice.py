import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest

from bentebot.domain import Metrics, Transcript
from bentebot.voice import DiscordPlayback, ReceiveSink, Speaker, TrackedPCM, VoiceReceiver


async def test_progress_tracks_sent_frames_and_stop_closes_gate():
    progress = []
    source = TrackedPCM(
        b"\1\0" * 19200,
        "one two three four five six seven eight nine ten",
        progress.append,
        asyncio.get_running_loop(),
    )
    for _ in range(5):
        assert len(source.read()) == 3840
    await asyncio.sleep(0)
    assert progress[-1] == "one two three four five"
    source.cleanup()
    assert source.read() == b""


async def test_lead_in_does_not_consume_pcm_or_mark_words_heard():
    progress = []
    source = TrackedPCM(b"\1\0" * 1920, "Bye", progress.append, asyncio.get_running_loop(), lead_in_frames=4)
    for _ in range(4):
        assert source.read() == b"\0" * 3840
    await asyncio.sleep(0)
    assert source.position == 0 and not progress
    assert source.read() == b"\1\0" * 1920
    await asyncio.sleep(0)
    assert progress[-1] == "Bye" and source.read() == b""


async def test_adjacent_sentence_playback_does_not_repeat_idle_lead_in():
    frames = []
    client = SimpleNamespace(is_playing=lambda: False, stop_playing=lambda: None)

    def play(source, after):
        current = []
        while data := source.read():
            current.append(data)
        frames.append(current)
        after(None)

    client.play = play
    playback = DiscordPlayback(client)
    await playback.play(b"\1\0" * 1920, "One", lambda _: None)
    await playback.play(b"\1\0" * 1920, "Two", lambda _: None)
    assert len(frames[0]) == 5 and len(frames[1]) == 1


async def test_playback_yields_and_cancellation_stops_only_output():
    client = SimpleNamespace()
    client.playing = False
    client.stops = 0

    def play(source, after):
        client.playing = True
        client.after = after

    def stop():
        client.stops += 1
        client.playing = False
        client.after(None)

    client.play, client.stop_playing, client.is_playing = play, stop, lambda: client.playing
    playback = DiscordPlayback(client)
    task = asyncio.create_task(playback.play(b"\0" * 3840, "hello", lambda text: None))
    await asyncio.sleep(0)
    assert client.playing and not task.done()
    gate = playback.source.gate
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert client.stops == 1 and not gate.is_set()


def test_receive_callback_is_bounded_and_thread_safe():
    frames, metrics = queue.Queue(maxsize=2), Metrics()
    sink = ReceiveSink(frames, metrics)
    user = SimpleNamespace(id=100, display_name="Me", bot=False)
    thread = threading.Thread(
        target=lambda: [sink.write(user, SimpleNamespace(pcm=b"\0" * 3840)) for _ in range(10)]
    )
    thread.start()
    thread.join()
    assert frames.qsize() == 2 and metrics.dropped_frames == 8


async def test_receiver_streams_before_silence_and_pads_missing_discord_packets():
    got_text, got_silence = asyncio.Event(), asyncio.Event()
    received = []

    class STT:
        async def transcribe(self, audio):
            async for block in audio:
                received.append(block)
                if len(received) == 1:
                    yield Transcript("speaking now")
                elif not any(block):
                    got_silence.set()

    async def transcript(user_id, name, update):
        assert user_id == 100 and name == "Me" and not update.final
        got_text.set()

    conversation = SimpleNamespace(
        metrics=Metrics(),
        settings=SimpleNamespace(max_speakers=6),
        partial={},
        speech_activity=lambda user_id: None,
        transcript=transcript,
    )
    client = SimpleNamespace(listen=lambda sink: None, stop_listening=lambda: None)
    receiver = VoiceReceiver(client, conversation, STT(), lambda message: None)
    receiver.speakers[100] = Speaker(100, "Me", vad=SimpleNamespace(is_speech=lambda frame, rate: True))
    receiver.incoming.put((100, "Me", b"\0\1" * 1920))
    receiver.start()
    try:
        await asyncio.wait_for(got_text.wait(), 2)
        await asyncio.wait_for(got_silence.wait(), 2)
        assert all(len(frame) == 3200 for frame in received)
        assert receiver.speakers[100].task and not receiver.speakers[100].task.done()
    finally:
        await receiver.close()
