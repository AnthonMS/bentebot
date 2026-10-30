from unittest.mock import AsyncMock

import pytest

from bentebot.domain import Event, ResponseRecord, Source
from bentebot.memory import Memory
from bentebot.speech import speak_response
from bentebot.voice_style import VoiceFollowupFilter, without_stock_tail


@pytest.mark.parametrize(
    "question",
    [
        "Anything else?",
        "Is there anything else I can help you with?",
        "Anything else you'd like to know?",
        "Anything else you'd like to ask?",
        "Is there anything else I can help you with today?",
        "Anything else you need help with right now?",
        "Are you ready to test it?",
        "Ready to try it out?",
        "Er der noget andet jeg kan hjælpe dig med?",
        "Er der andet du vil vide?",
        "Noget andet du gerne vil spørge om?",
        "Er du klar til at teste det?",
    ],
)
def test_generic_followups_are_removed_after_content(question):
    gate = VoiceFollowupFilter()
    assert gate("That worked.") == "That worked."
    assert gate(question) == ""
    assert without_stock_tail(question) == ""


@pytest.mark.parametrize(
    "question",
    [
        "Which Python version?",
        "What happened?",
        "Anything else crashed?",
        "Are you ready to delete the backup?",
        "Hvilken funktion mener du?",
        "Is there anything else in the logs?",
        "Anything else failed today?",
        "Are you ready to test the backup restore?",
    ],
)
def test_specific_questions_are_preserved(question):
    gate = VoiceFollowupFilter()
    gate("Let's check.")
    assert gate(question) == question


def test_standalone_question_is_not_silenced():
    assert VoiceFollowupFilter()("Ready to test?") == "Ready to test?"


async def test_streamed_followup_never_reaches_tts_or_becomes_unheard_memory(wav_bytes):
    async def tokens():
        for token in ["That worked. ", "Is there anything ", "else I can help you with?"]:
            yield token

    synth = AsyncMock(return_value=wav_bytes)

    async def play(pcm, text, progress):
        progress(text)

    record = ResponseRecord()
    await speak_response(
        tokens(), synth, play, AsyncMock(return_value=True), record, filter_stock_followups=True
    )
    assert [call.args[0] for call in synth.await_args_list] == ["That worked."]
    assert record.spoken == "That worked."
    assert "anything else" in record.generated
    event = Event(
        Source("channel", 10, 1),
        999,
        "Bentebot",
        record.spoken,
        kind="assistant",
        delivery="voice",
        generated=record.generated,
        spoken=record.spoken,
    )
    assert Memory.render_voice(None, event) == ("That worked.", "assistant")
