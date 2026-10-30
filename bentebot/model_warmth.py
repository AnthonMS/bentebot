"""Session-owned best-effort residency through a shared OpenAI-compatible gateway.

The gateway has no portable pin/unpin API. Small uncached requests renew a local
backend's idle TTL; eviction by competing workloads remains a backend decision.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings

log = logging.getLogger(__name__)


class WarmProvider(Protocol):
    async def warm(self, resource: str) -> str | None: ...

    def recently_used(self, resource: str, seconds: float) -> bool: ...


@dataclass
class WarmState:
    model: str
    status: str = "warming"
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    last_success: float | None = None
    upstream_host: str = "unreported"


class VoiceModelWarmth:
    def __init__(self, settings: Settings, chat: WarmProvider, *, kind: str = "model"):
        self.settings, self.chat = settings, chat
        self.kind = kind
        self.allowed = {name.strip() for name in settings.voice_warm_models.split(",") if name.strip()}
        self.owners: dict[int, str] = {}
        self.models: dict[str, WarmState] = {}
        self.retired: set[asyncio.Task] = set()
        self.closed = False

    def select(self, owner: int, model: str) -> WarmState | None:
        """Start immediately, deduplicating the same alias across voice sessions."""
        if self.closed:
            return None
        previous = self.owners.get(owner)
        if previous == model:
            return self.models[model]
        self.owners.pop(owner, None)
        if previous and previous not in self.owners.values():
            old = self.models.pop(previous)
            old.ready.set()
            if old.task:
                old.task.cancel()
                self.retired.add(old.task)
                old.task.add_done_callback(self.retired.discard)
        if model not in self.allowed:
            return None
        self.owners[owner] = model
        if model not in self.models:
            state = self.models[model] = WarmState(model)
            state.task = asyncio.create_task(self._run(state), name=f"voice-model-warm-{model}")
        return self.models[model]

    async def prepare(self, owner: int, model: str) -> None:
        state = self.select(owner, model)
        if state:
            # Cancelling an interrupted answer must not cancel shared prewarming.
            # A failed attempt also releases this gate: normal generation can retry.
            await state.ready.wait()

    async def _run(self, state: WarmState) -> None:
        while True:
            if state.ready.is_set() and self.chat.recently_used(
                state.model, self.settings.voice_warm_interval
            ):
                await asyncio.sleep(self.settings.voice_warm_interval)
                continue
            # Gate only initial loading. A periodic keep-alive must not hold up a
            # real response; the backend arbitrates requests if both overlap.
            state.status = "warming"
            started = asyncio.get_running_loop().time()
            try:
                async with asyncio.timeout(self.settings.voice_warm_timeout):
                    host = await self.chat.warm(state.model)
                    state.upstream_host = host if isinstance(host, str) else "unreported"
                state.last_success = asyncio.get_running_loop().time()
                state.status = "ready"
                log.info(
                    "voice_%s_warm resource=%s elapsed_ms=%.1f",
                    self.kind,
                    state.model,
                    (state.last_success - started) * 1000,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.status = "failed"
                # No provider bodies, credentials, prompts or generated text in logs.
                log.warning(
                    "voice_%s_warm_failed resource=%s error=%s", self.kind, state.model, type(exc).__name__
                )
            finally:
                state.ready.set()
            await asyncio.sleep(self.settings.voice_warm_interval)

    def describe(self, owner: int) -> str:
        caption = "TTS" if self.kind == "tts" else "Model"
        state = self.models.get(self.owners.get(owner, ""))
        if not state:
            return f"{caption} prewarming: disabled for this alias"
        age = (
            f"; last successful refresh {asyncio.get_running_loop().time() - state.last_success:.0f}s ago"
            if state.last_success is not None
            else ""
        )
        return f"{caption} prewarming: {state.status} ({state.model}); upstream {state.upstream_host}{age}"

    async def release(self, owner: int) -> None:
        self.select(owner, "")
        await asyncio.gather(*self.retired, return_exceptions=True)

    async def close(self) -> None:
        self.closed = True
        tasks = set(self.retired)
        for state in self.models.values():
            state.ready.set()
            if state.task:
                state.task.cancel()
                tasks.add(state.task)
        await asyncio.gather(*tasks, return_exceptions=True)
        self.owners.clear()
        self.models.clear()
