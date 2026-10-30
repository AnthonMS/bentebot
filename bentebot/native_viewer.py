"""Bot-authorized, short-lived leases for the isolated Discord user-account viewer."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

from .screen import ScreenService


def control_key(path: str) -> str:
    try:
        key = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        raise ValueError("Cannot read the native viewer control-key file.") from None
    if not 32 <= len(key) <= 128 or not key.isascii():
        raise ValueError("Native viewer control key must contain 32-128 ASCII characters.")
    return key


@dataclass(frozen=True)
class ViewerTarget:
    guild_id: int
    channel_id: int
    owner_id: int
    bot_user_id: int
    audience_ids: frozenset[int]


class NativeViewerBroker:
    def __init__(self, screens: ScreenService, desired, *, viewer_id: int, key: str, clock=time.monotonic):
        self.screens, self.desired = screens, desired
        self.viewer_id, self._key, self.clock = viewer_id, key, clock
        self._lock = asyncio.Lock()
        self._target = None
        self._token = ""
        self._binding = None
        self.last_seen = None
        self.worker_state = "not_connected"
        self.decoded_frames = 0
        self.source_size = None
        self.sample_size = None
        self._closed = False

    def install(self, app):
        app.router.add_get("/native-viewer/lease", self.lease)
        app.router.add_post("/native-viewer/report", self.report)

    def _authorize(self, request):
        auth = request.headers.get("Authorization", "")
        if len(auth) > 200 or not auth.isascii() or not secrets.compare_digest(auth, "Bearer " + self._key):
            raise web.HTTPUnauthorized(text="Viewer authentication required")
        if request.headers.get("X-Bentebot-Viewer-Id") != str(self.viewer_id):
            raise web.HTTPForbidden(text="Viewer account identity does not match configuration")

    def revoke(self):
        if self._token:
            self.screens.revoke_native(self._token)
        self._target, self._token, self._binding = None, "", None
        self.source_size = self.sample_size = None

    async def lease(self, request):
        self._authorize(request)
        async with self._lock:
            target = None if self._closed else await self.desired()
            self.last_seen = self.clock()
            if target != self._target:
                self.revoke()
            if target is None:
                return web.json_response({"active": False}, headers={"Cache-Control": "no-store"})
            if not self._token or not self.screens.renew_native(self._token):
                self._token, self._binding = self.screens.open_native(
                    target.owner_id, target.guild_id, target.channel_id, target.audience_ids
                )
                self._target = target
            return web.json_response(
                {
                    "active": True,
                    "lease_id": self._binding.session_id,
                    "lease_seconds": 3,
                    "guild_id": str(target.guild_id),
                    "channel_id": str(target.channel_id),
                    "owner_id": str(target.owner_id),
                    "bot_user_id": str(target.bot_user_id),
                    "viewer_user_id": str(self.viewer_id),
                    "frame_token": self._token,
                },
                headers={"Cache-Control": "no-store"},
            )

    async def report(self, request):
        self._authorize(request)
        body = await self.screens._body(request, 2048)
        try:
            payload = json.loads(body)
            state = payload["state"]
            frames = payload.get("decoded_frames", 0)
            sizes = []
            for name in ("source_size", "sample_size"):
                size = payload.get(name)
                if size is not None and (
                    not isinstance(size, list)
                    or len(size) != 2
                    or any(type(value) is not int or not 8 <= value <= 8192 for value in size)
                    or size[0] * size[1] > 16_000_000
                ):
                    raise ValueError
                sizes.append(tuple(size) if size is not None and state == "watching" else None)
            if state not in {"idle", "connecting", "watching", "error"}:
                raise ValueError
            if not isinstance(frames, int) or isinstance(frames, bool) or not 0 <= frames < 2**53:
                raise ValueError
            if state != "idle" and (
                self._binding is None or payload.get("lease_id") != self._binding.session_id
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise web.HTTPBadRequest(text="Invalid or expired viewer report") from None
        self.worker_state, self.decoded_frames = state, frames
        self.source_size, self.sample_size = sizes
        return web.json_response({"accepted": True}, headers={"Cache-Control": "no-store"})

    def describe(self):
        age = None if self.last_seen is None else self.clock() - self.last_seen
        return {
            "viewer_state": self.worker_state if age is not None and age < 5 else "not_connected",
            "viewer_seen_seconds_ago": None if age is None else round(age, 1),
            "viewer_decoded_frames": self.decoded_frames,
            "viewer_stream_size": "x".join(map(str, self.source_size)) if self.source_size else None,
            "viewer_sample_size": "x".join(map(str, self.sample_size)) if self.sample_size else None,
        }

    def close(self):
        self._closed = True
        self.revoke()
