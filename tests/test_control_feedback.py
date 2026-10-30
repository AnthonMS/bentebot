import asyncio
from types import SimpleNamespace

import pytest

from bentebot.control_feedback import control_acknowledgement
from bentebot.tools import WEB_TOOLS_ALLOWED


async def test_ack_is_model_generated_and_web_tools_disabled_only_for_its_request():
    async def stream(messages, model):
        assert model == "selected"
        assert not WEB_TOOLS_ALLOWED.get()
        assert "accepted_action" in messages[1]["content"]
        yield "Right, "
        yield "I'll wait for you."

    before = WEB_TOOLS_ALLOWED.get()
    text = await control_acknowledgement(SimpleNamespace(stream=stream), "selected", "stop", "pause please")
    assert text == "Right, I'll wait for you."
    assert WEB_TOOLS_ALLOWED.get() == before


@pytest.mark.parametrize(
    "text", ["x" * 241, "```command```", '{"action":1}', "https://example.com", "word " * 19, ""]
)
async def test_invalid_or_oversized_ack_is_skipped_without_scripted_fallback(text):
    closed = False

    async def stream(*args):
        nonlocal closed
        try:
            yield text
        finally:
            closed = True

    assert await control_acknowledgement(SimpleNamespace(stream=stream), "selected", "leave", "bye") is None
    assert closed


async def test_slow_ack_times_out_and_closes_generation():
    closed = False

    async def stream(*args):
        nonlocal closed
        try:
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closed = True

    assert await control_acknowledgement(SimpleNamespace(stream=stream), "selected", "leave", "bye") is None
    assert closed


async def test_ack_cancellation_propagates_without_swallowing_parent_shutdown():
    entered = asyncio.Event()

    async def stream(*args):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    task = asyncio.create_task(
        control_acknowledgement(SimpleNamespace(stream=stream), "selected", "stop", "stop")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
