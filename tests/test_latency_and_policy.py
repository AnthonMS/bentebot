import ast
import asyncio
import json
import logging
import runpy
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import httpx
import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience, ResponseRecord, Source, Transcript
from bentebot.providers import OpenAIChat
from bentebot.speech import speak_response


async def test_real_sse_client_plays_sentence_before_generation_finishes(wav_bytes):
    played = asyncio.Event()
    record = ResponseRecord()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for text in ("First complete sentence. ", "Second sentence."):
                if text.startswith("Second"):
                    await played.wait()  # Upstream cannot finish until playback actually starts.
                payload = {"choices": [{"delta": {"content": text}}]}
                yield ("data: " + json.dumps(payload) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

    async def play(pcm, sentence, progress):
        if not played.is_set():
            assert record.generated == "First complete sentence. "
            played.set()
        progress(sentence)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=Stream()))
    ) as client:
        chat = OpenAIChat(Settings(), client)
        async with asyncio.timeout(2):
            await speak_response(
                chat.stream([], "test"),
                AsyncMock(return_value=wav_bytes),
                play,
                AsyncMock(return_value=True),
                record,
            )
    assert played.is_set()
    assert record.timings["chat_first_token_ms"] <= record.timings["first_sentence_ms"]
    assert record.timings["tts_first_ms"] >= 0
    assert record.timings["pipeline_first_audio_ms"] >= record.timings["tts_first_ms"]


@pytest.mark.parametrize("utterance", ["How are you?", "Explain why the sky is blue."])
async def test_stt_and_response_timings_are_visible_without_logging_content(
    store, wav_bytes, caplog, utterance
):
    source = Source("channel", 10, 1)
    memory = SimpleNamespace(
        context=AsyncMock(return_value=([{"role": "system", "content": "test"}], (source,))),
        context_budget=lambda **kwargs: 8000,
        policy=SimpleNamespace(allows_all=AsyncMock(return_value=True)),
    )
    played = asyncio.Event()

    async def play(pcm, sentence, progress):
        played.set()
        progress(sentence)

    class Chat:
        async def stream(self, messages, model):
            yield "PRIVATE_ANSWER. "
            await played.wait()
            yield "All done."

    conversation = VoiceConversation(
        Settings(),
        store,
        memory,
        Chat(),
        SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes)),
        source,
        999,
        AsyncMock(return_value=Audience(frozenset({100}), 1)),
        play,
        lambda: None,
        AsyncMock(),
    )
    with caplog.at_level(logging.INFO, logger="bentebot.conversation"):
        conversation.speech_activity(100)
        await conversation.transcript(100, "User", Transcript("PRIVATE_REQUEST"))
        await conversation.transcript(100, "User", Transcript("PRIVATE_REQUEST", final=True))
        conversation.latest_request = utterance
        # Later raw VAD activity during an acknowledgement must not reset the
        # speech-end estimate belonging to the committed request.
        conversation.final_received_at = time.monotonic() - 0.1
        conversation.metrics.stt_final_lag_ms = 250.0
        conversation.last_speech = time.monotonic()
        async with asyncio.timeout(2):
            await conversation._answer()
    assert conversation.metrics.stt_interims == conversation.metrics.stt_finals == 1
    assert conversation.metrics.stt_first_update_ms is not None
    assert conversation.metrics.stt_final_lag_ms is not None
    assert conversation.metrics.last_turn["speech_end_to_audio_dispatch_ms"] >= 350
    assert "voice_turn" in caplog.text and "stt_first_update" in caplog.text
    assert "PRIVATE_REQUEST" not in caplog.text and "PRIVATE_ANSWER" not in caplog.text


@pytest.mark.parametrize("policy,prompt", [("server", False), ("individual", True)])
async def test_clone_uses_standing_server_consent_without_a_repeated_prompt(store, policy, prompt):
    bot = BenteBot(Settings(health_port=0, voice_clone_policy=policy), store)
    channel = SimpleNamespace(id=10)
    target = SimpleNamespace(id=200, bot=False, voice=SimpleNamespace(channel=channel), mention="@target")
    interaction = SimpleNamespace(
        guild_id=1,
        user=SimpleNamespace(id=100, voice=SimpleNamespace(channel=channel), mention="@actor"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    bot.profiles = SimpleNamespace(close=AsyncMock())
    bot.voice_sessions[1] = (SimpleNamespace(source=Source("channel", 10, 1)), None, None)
    bot.begin_enrollment = AsyncMock()
    try:
        await bot.tree.get_command("speech").get_command("clone").callback(interaction, "Friend", target)
        if prompt:
            view = interaction.response.send_message.call_args.kwargs["view"]
            view.stop()
            bot.begin_enrollment.assert_not_awaited()
        else:
            interaction.response.send_message.assert_not_awaited()
            assert bot.begin_enrollment.call_args.kwargs["consent_basis"] == "server_policy"
    finally:
        bot.voice_sessions.clear()
        await bot.close()


async def test_voice_join_has_no_public_retention_announcement(store, monkeypatch, wav_bytes):
    bot = BenteBot(
        Settings(
            health_port=0,
            chat_model="local",
            voice_warm_models="local",
            stt_url="ws://stt/v1/listen",
            tts_base_url="http://tts/v1",
            tts_model="tts",
            tts_voice="test",
        ),
        store,
    )
    bot._connection.user = SimpleNamespace(id=999)
    bot.chat.warm = AsyncMock()
    bot.tts.synthesize = AsyncMock(return_value=wav_bytes)
    channel = MagicMock(spec=discord.VoiceChannel)
    channel.id, channel.mention = 10, "<#10>"
    channel.permissions_for.return_value = SimpleNamespace(connect=True, speak=True, view_channel=True)
    client = MagicMock()
    client.channel = channel
    channel.connect = AsyncMock(return_value=client)
    channel.send = AsyncMock()
    member = SimpleNamespace(
        guild=SimpleNamespace(id=1, me=object(), voice_client=None), voice=SimpleNamespace(channel=channel)
    )
    monkeypatch.setattr("bentebot.discord_app.VoiceReceiver.start", lambda receiver: None)
    try:
        assert await bot.join_voice(member, MagicMock()) == "Joined <#10>."
        async with asyncio.timeout(1):
            await bot.model_warmth.models["local"].ready.wait()
            await bot.tts_warmth.models["test"].ready.wait()
        bot.chat.warm.assert_awaited_once_with("local")
        bot.tts.synthesize.assert_awaited_once_with("Ready.", "test")
        speech_task = bot.tts_warmth.models["test"].task
        warm_task = bot.model_warmth.models["local"].task
        channel.send.assert_not_awaited()
        assert bot.voice_sessions[1][1].language() == "auto"
        assert bot.tree.get_command("speech").get_command("language") is None
        await bot.pause_voice(1)
        assert not warm_task.done()
        assert not speech_task.done()
        await bot.leave_voice(1)
        assert warm_task.cancelled()
        assert speech_task.cancelled()
        assert not bot.model_warmth.models
    finally:
        await bot.close()


def test_external_backend_tuning_is_narrow_configurable_and_keeps_auto_detection():
    tune = runpy.run_path(str(Path(__file__).parents[1] / "deploy/speech/tune_backend.py"))["tuned_source"]
    original = """def transcribe(self, audio):
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type
        return self.model.transcribe(audio,
            beam_size=5,
            language=self.original_language,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
    )
"""
    patched = tune(original)
    tree = ast.parse(patched)
    options = {
        keyword.arg: keyword.value
        for keyword in next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "transcribe"
        ).keywords
    }
    assert isinstance(options["beam_size"], ast.Call)
    assert options["condition_on_previous_text"].value is False
    assert isinstance(options["multilingual"], ast.BoolOp)
    with pytest.raises(ValueError, match="changed"):
        tune(patched)


@pytest.mark.parametrize(
    "effort,aliases,selected,expected",
    [
        ("", "", "default", False),
        ("none", "", "default", True),
        ("none", "", "other-provider", False),
        ("none", "qwen-fast, qwen-backup", "qwen-backup", True),
    ],
)
async def test_reasoning_control_is_opt_in_and_scoped_to_selected_aliases(
    effort, aliases, selected, expected
):
    def handler(request):
        payload = json.loads(request.content)
        assert ("reasoning_effort" in payload) is expected
        if expected:
            assert payload["reasoning_effort"] == "none"
        return httpx.Response(
            200, content=b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = OpenAIChat(
            Settings(chat_model="default", chat_reasoning_effort=effort, chat_reasoning_models=aliases),
            client,
        )
        assert [token async for token in chat.stream([], selected)] == ["Hello"]


def test_invalid_reasoning_effort_is_rejected_before_network_access():
    with pytest.raises(ValueError, match="reasoning_effort"):
        Settings(chat_reasoning_effort="magic").validate(login=False)


@pytest.mark.parametrize("unfinished", [False, True])
def test_speech_clock_resets_after_idle_but_preserves_unfinished_transcript(store, monkeypatch, unfinished):
    conversation = VoiceConversation(
        Settings(),
        store,
        None,
        None,
        None,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        lambda: None,
        AsyncMock(),
    )
    conversation.speaker_clock[100] = 9.0
    conversation.speaker_last_voice[100] = 10.0
    if unfinished:
        conversation.partial[100] = "still waiting"
    monkeypatch.setattr("bentebot.conversation.time.monotonic", lambda: 100.0)
    conversation.speech_activity(100)
    assert conversation.speaker_clock[100] == (9.0 if unfinished else 100.0)
