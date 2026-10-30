import asyncio
import struct

import pytest

from bentebot.domain import ResponseRecord
from bentebot.providers import ProviderError
from bentebot.speech import SentenceBuffer, speak_response, wav_to_pcm
from bentebot.text import discord_chunks


def test_sentence_context_and_final_unpunctuated_tail():
    splitter = SentenceBuffer()
    assert splitter.feed("This is") == []
    assert splitter.feed(" a complete sentence. Another") == ["This is a complete sentence."]
    assert splitter.feed(" thought", final=True) == ["Another thought"]


def test_code_never_becomes_speech_even_with_split_fences():
    splitter = SentenceBuffer()
    output = []
    for part in ("Here is code. `", "``python\nprint('hello')\n`", "``\nThat explains it."):
        output.extend(splitter.feed(part))
    output.extend(splitter.feed("", final=True))
    assert "print" not in str(output) and "python" not in str(output)
    assert "That explains it." in output
    assert splitter.had_code


def test_wav_decodes_to_discord_pcm(wav_bytes):
    pcm = wav_to_pcm(wav_bytes)
    assert 18000 < len(pcm) <= 19200
    with pytest.raises(ProviderError):
        wav_to_pcm(b"garbage")


def test_nonseekable_speaches_wav_header_is_supported(wav_bytes):
    data = bytearray(wav_bytes)
    struct.pack_into("<I", data, 4, 0xFFFFFFFF)
    offset = data.index(b"data")
    struct.pack_into("<I", data, offset + 4, 0xFFFFFFFF)
    assert wav_to_pcm(bytes(data)) == wav_to_pcm(wav_bytes)


def test_finite_truncated_wav_is_still_rejected(wav_bytes):
    with pytest.raises(ProviderError):
        wav_to_pcm(wav_bytes[:-20])


def test_token_boundaries_do_not_split_decimals_or_titles():
    splitter = SentenceBuffer()
    assert splitter.feed("Dr. Smith measured 3.") == []
    assert splitter.feed("14 volts. That's enough. ") == ["Dr. Smith measured 3.14 volts.", "That's enough."]


@pytest.mark.parametrize("filter_followups", [False, True])
async def test_pipeline_overlaps_generation_synthesis_and_playback(wav_bytes, filter_followups):
    playback_started = asyncio.Event()
    spoken = []

    async def tokens():
        yield "First sentence. "
        # Generation must continue independently while playback is active.
        await asyncio.wait_for(playback_started.wait(), 2)
        yield "Second sentence"

    async def synth(text):
        await asyncio.sleep(0)
        return wav_bytes

    async def play(pcm, text, progress):
        playback_started.set()
        await asyncio.sleep(0)
        spoken.append(text)
        progress(text)

    async def allowed():
        return True

    record = ResponseRecord()
    async with asyncio.timeout(5):
        await speak_response(tokens(), synth, play, allowed, record, filter_stock_followups=filter_followups)
    assert spoken == ["First sentence.", "Second sentence"]
    assert record.generated == "First sentence. Second sentence"
    assert record.spoken == "First sentence. Second sentence"


async def test_cancellation_kills_generation_tts_and_queued_playback(wav_bytes):
    playing, generator_closed = asyncio.Event(), asyncio.Event()
    played = []

    async def tokens():
        try:
            for _ in range(50):
                yield "Another sentence. "
        finally:
            generator_closed.set()

    async def synth(text):
        return wav_bytes

    async def play(pcm, text, progress):
        played.append(text)
        progress("Another")
        playing.set()
        await asyncio.Event().wait()

    async def allowed():
        return True

    record = ResponseRecord()
    task = asyncio.create_task(speak_response(tokens(), synth, play, allowed, record))
    await asyncio.wait_for(playing.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert record.interrupted and record.spoken == "Another"
    assert len(played) == 1
    assert generator_closed.is_set()


async def test_permission_change_prevents_next_sentence(wav_bytes):
    permission = True
    played = []

    async def tokens():
        yield "First sentence. Second sentence."

    async def synth(text):
        return wav_bytes

    async def play(pcm, text, progress):
        nonlocal permission
        played.append(text)
        progress(text)
        permission = False

    async def allowed():
        return permission

    with pytest.raises(ProviderError):
        await speak_response(tokens(), synth, play, allowed, ResponseRecord())
    assert played == ["First sentence."]


def test_long_code_is_readable_across_discord_chunks():
    text = "```python\n" + "print('hello')\n" * 300 + "```\nDone."
    chunks = discord_chunks(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1900 for chunk in chunks)
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)
