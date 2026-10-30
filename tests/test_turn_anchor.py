import asyncio
from types import SimpleNamespace

from bentebot.config import Settings
from bentebot.conversation import VoiceConversation
from bentebot.domain import Audience, Source, Transcript
from bentebot.memory import Memory
from bentebot.permissions import AccessPolicy


async def test_voice_prompt_remains_anchored_while_a_new_final_arrives(store, wav_bytes):
    entered, release = asyncio.Event(), asyncio.Event()
    original_request = "Write a Python function."
    later_request = "explain it"
    source = Source("channel", 10, 1)
    prompts = []

    async def can_read(*args):
        return True

    async def context(audience, *, query="", voice=False, persona_channel_id=None):
        assert persona_channel_id == source.channel_id
        assert query == original_request
        assert voice
        entered.set()
        await release.wait()
        return [{"role": "system", "content": "Companion"}], (source,)

    async def current_audience():
        return Audience(frozenset({100}), 1)

    async def stream(messages, model):
        prompts.append(messages)
        yield "```python\ndef example():\n    return 1\n```"

    async def synthesize(text, voice):
        return wav_bytes

    async def play(pcm, text, progress):
        progress(text)

    async def noop(*args):
        pass

    async def posted(*args):
        return True

    memory = Memory(store, AccessPolicy(can_read), 10000)
    memory.context = context
    session = VoiceConversation(
        Settings(),
        store,
        memory,
        SimpleNamespace(stream=stream),
        SimpleNamespace(synthesize=synthesize),
        source,
        999,
        current_audience,
        play,
        lambda: None,
        noop,
        text_output=posted,
    )
    session.latest_request = original_request
    session.task = asyncio.create_task(session._answer())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await session.transcript(100, "Me", Transcript(later_request, final=True))
        release.set()
        await asyncio.wait_for(session.task, 2)
        assert prompts[0][-1] == {"role": "user", "content": original_request}
        assert session.latest_request == later_request and session.pending
    finally:
        await session.close()
