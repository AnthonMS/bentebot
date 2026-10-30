import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Metrics, Source, Transcript
from bentebot.voice import Speaker, VoiceReceiver


async def test_departed_speaker_packets_and_late_transcript_cannot_survive_rejoin():
    conversation = SimpleNamespace(metrics=Metrics(), transcript=AsyncMock())
    receiver = VoiceReceiver(SimpleNamespace(stop_listening=Mock()), conversation, None, AsyncMock())
    speaker = Speaker(100, "departed")
    entered = asyncio.Event()

    async def recognition():
        entered.set()
        await asyncio.Event().wait()

    speaker.task = asyncio.create_task(recognition())
    receiver.speakers[100] = speaker
    receiver.incoming.put((100, "departed", b"old audio"))
    receiver.incoming.put((200, "remaining", b"current audio"))
    receiver.incoming.put((100, "departed", b"more old audio"))
    try:
        await entered.wait()
        receiver.set_members([200])
        await receiver.drop_speaker(100)
        assert speaker.task.cancelled() and not speaker.active
        assert receiver.incoming.get_nowait() == (200, "remaining", b"current audio")
        assert receiver.incoming.empty()
        receiver.set_members([100, 200])

        async def late():
            yield Transcript("Old answer arriving after rejoin", True)

        await receiver._receive_updates(speaker, late())
        conversation.transcript.assert_not_awaited()
    finally:
        await receiver.close()


async def test_receiver_cleanup_cancels_streams_even_if_stop_listening_raises():
    client = SimpleNamespace(stop_listening=Mock(side_effect=RuntimeError("already disconnected")))
    receiver = VoiceReceiver(client, SimpleNamespace(metrics=Metrics()), None, AsyncMock())
    speaker = Speaker(100, "Me")
    speaker.task = asyncio.create_task(asyncio.Event().wait())
    receiver.speakers[100] = speaker
    with pytest.raises(RuntimeError):
        await receiver.close()
    assert receiver.closed and speaker.task.cancelled() and not receiver.speakers
    await receiver.close()
    client.stop_listening.assert_called_once()


async def test_departure_clears_quiet_gate_and_pending_work_without_invalidating_accepted_revision(store):
    session = VoiceConversation(
        Settings(),
        store,
        None,
        None,
        None,
        Source("channel", 10, 1),
        999,
        AsyncMock(),
        AsyncMock(),
        Mock(),
        AsyncMock(),
    )
    session.partial = {100: "unfinished", 200: "still speaking"}
    session.pending_controls = {100: "leave", 200: "stop"}
    session.speech_revision[100] = 3
    session.latest_speaker_id = 100
    session.pending = True
    session.member_left(100)
    assert session.partial == {200: "still speaking"}
    assert session.pending_controls == {200: "stop"}
    assert session.pending and session.speech_revision[100] == 3
    await session.close()
