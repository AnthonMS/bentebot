import asyncio
from types import SimpleNamespace

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.domain import Event, Source


async def test_deletion_waits_for_generation_finalizers(store):
    bot = BenteBot(Settings(health_port=0), store)
    source = Source("channel", 10, 1)
    started = asyncio.Event()

    async def generation():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            # This is how text/voice generation preserve their unspoken draft on cancellation.
            store.save(Event(source, 999, "Bot", "secret draft", kind="assistant"))

    task = asyncio.create_task(generation())
    bot.text_tasks.add(task)
    await started.wait()
    try:
        await bot.on_raw_message_delete(SimpleNamespace(guild_id=1, channel_id=10))
        assert task.cancelled() and store.count() == 0
        await asyncio.sleep(0)
        assert store.count() == 0
    finally:
        await bot.close()
