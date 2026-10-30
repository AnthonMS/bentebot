"""Screen samples with short-lived RAM history and exact Discord audience binding.

The native viewer uploads under a server-issued lease; optional companion capture
uses one-time pairing. Neither route persists pixels or grants access across calls.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import math
import secrets
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from importlib.resources import files
from urllib.parse import urlsplit

from aiohttp import web
from PIL import Image, ImageOps, UnidentifiedImageError

from .domain import Audience, Source


@dataclass(frozen=True)
class ScreenBinding:
    owner_id: int
    guild_id: int
    channel_id: int
    audience_ids: frozenset[int]
    session_id: str
    transport: str = "companion"


@dataclass(frozen=True)
class PairGrant:
    code: str
    expires_in: int


@dataclass(frozen=True)
class ScreenSnapshot:
    owner_id: int
    guild_id: int
    channel_id: int
    audience_ids: frozenset[int]
    session_id: str
    created_at: float
    received_at: float
    data_url: str = field(repr=False)
    width: int = 0
    height: int = 0
    fingerprint: str = field(default="", repr=False)
    crop_box: tuple[int, int, int, int] | None = None

    @property
    def source(self) -> Source:
        return Source(
            "screen",
            self.channel_id,
            self.guild_id,
            owner_id=self.owner_id,
            session_id=self.session_id,
            audience_ids=tuple(sorted(self.audience_ids)),
        )

    @property
    def source_label(self) -> str:
        return f"Screen shared by Discord user {self.owner_id} in voice channel {self.channel_id}"


@dataclass
class _Pending:
    binding: ScreenBinding
    expires_at: float


@dataclass
class _Session:
    binding: ScreenBinding
    expires_at: float
    last_upload: float = -100.0
    sequence: int = 0
    frames_received: int = 0
    snapshot: ScreenSnapshot | None = None
    history: list[ScreenSnapshot] = field(default_factory=list, repr=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalise_image(data: bytes, *, max_pixels: int) -> tuple[bytes, int, int]:
    """Decode before accepting, reject bombs/animation, strip metadata and downscale."""
    try:
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in {"JPEG", "PNG"} or getattr(original, "n_frames", 1) != 1:
                raise ValueError("Use a single JPEG or PNG frame")
            width, height = original.size
            if width < 8 or height < 8 or width > 4096 or height > 4096 or width * height > max_pixels:
                raise ValueError("Frame dimensions exceed limits")
            original.load()
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1600, 1000), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=82)
            return output.getvalue(), image.width, image.height
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Invalid image frame") from exc


class ScreenService:
    """Only the Discord command layer can issue a binding; HTTP callers cannot pick identity."""

    def __init__(
        self,
        validate: Callable[[ScreenBinding], Awaitable[bool]],
        *,
        pair_ttl: int = 120,
        session_ttl: int = 3600,
        frame_ttl: int = 10,
        max_sessions: int = 6,
        max_context_frames: int = 2,
        max_frame_bytes: int = 1_000_000,
        max_pixels: int = 4_000_000,
        history_seconds: float = 10.0,
        max_history_frames: int = 10,
        max_history_bytes: int = 4 * 1024 * 1024,
        max_total_history_bytes: int = 16 * 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not math.isfinite(history_seconds) or not 0 < history_seconds <= 60:
            raise ValueError("Screen history must retain between 0 and 60 seconds")
        if not 1 <= max_history_frames <= 60 or min(max_history_bytes, max_total_history_bytes) <= 0:
            raise ValueError("Screen history requires positive byte limits and between 1 and 60 frames")
        self.validate = validate
        self.pair_ttl, self.session_ttl, self.frame_ttl = pair_ttl, session_ttl, frame_ttl
        self.max_sessions, self.max_context_frames = max_sessions, max_context_frames
        self.max_frame_bytes, self.max_pixels = max_frame_bytes, max_pixels
        self.history_seconds, self.max_history_frames = history_seconds, max_history_frames
        self.max_history_bytes, self.max_total_history_bytes = max_history_bytes, max_total_history_bytes
        self.clock = clock
        self.pending: dict[str, _Pending] = {}
        self.sessions: dict[str, _Session] = {}
        self.attempts: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._decode_limit = asyncio.Semaphore(2)
        self._cleaner: asyncio.Task | None = None
        self._revision = 0
        self._closed = False

    def issue_pair(
        self, owner_id: int, guild_id: int, channel_id: int, audience_ids: frozenset[int]
    ) -> PairGrant:
        if self._closed:
            raise ValueError("Screen service is closed")
        if owner_id not in audience_ids or not audience_ids or not guild_id or not channel_id:
            raise ValueError("Screen capture must include its owner in a bound voice audience")
        self.expire()
        self.invalidate(owner_id=owner_id)
        if len(self.sessions) + len(self.pending) >= self.max_sessions:
            raise ValueError("The screen-sharing session limit has been reached")
        code = secrets.token_urlsafe(18)
        binding = ScreenBinding(
            owner_id, guild_id, channel_id, frozenset(audience_ids), secrets.token_hex(16)
        )
        self.pending[_digest(code)] = _Pending(binding, self.clock() + self.pair_ttl)
        return PairGrant(code, self.pair_ttl)

    def open_native(
        self, owner_id: int, guild_id: int, channel_id: int, audience_ids: frozenset[int]
    ) -> tuple[str, ScreenBinding]:
        """Issue a server-bound upload grant without any streamer-side pairing flow."""
        grant = self.issue_pair(owner_id, guild_id, channel_id, audience_ids)
        pending = self.pending.pop(_digest(grant.code))
        binding = replace(pending.binding, transport="native")
        token = secrets.token_urlsafe(32)
        self.sessions[_digest(token)] = _Session(binding, self.clock() + 5)
        return token, binding

    def renew_native(self, token: str) -> bool:
        self.expire()
        session = self.sessions.get(_digest(token))
        if self._closed or session is None or session.binding.transport != "native":
            return False
        session.expires_at = self.clock() + 5
        return True

    def revoke_native(self, token: str) -> None:
        self._drop_session(_digest(token))

    def _drop_session(self, key: str, expected: _Session | None = None) -> None:
        """Clear owned pixel references, including sessions held by an in-flight task."""
        session = self.sessions.get(key)
        if session is not None and (expected is None or session is expected):
            del self.sessions[key]
            session.snapshot = None
            session.history.clear()

    @staticmethod
    def _history_bytes(session: _Session) -> int:
        # Count every sample conservatively, even when identical immutable strings
        # share an allocation. Base64 ASCII length is the actual encoded payload size.
        return sum(len(frame.data_url) for frame in session.history)

    @property
    def buffered_bytes(self) -> int:
        return sum(self._history_bytes(session) for session in self.sessions.values())

    def _trim_history(self, session: _Session) -> None:
        now = self.clock()
        session.history[:] = [
            frame
            for frame in session.history
            if now - frame.received_at <= self.history_seconds
            or (frame is session.snapshot and now - frame.received_at <= self.frame_ttl)
        ]
        while len(session.history) > self.max_history_frames or (
            self._history_bytes(session) > self.max_history_bytes
        ):
            session.history.pop(0)
        session.snapshot = (
            session.history[-1]
            if session.history and now - session.history[-1].received_at <= self.frame_ttl
            else None
        )

    def _remember(self, session: _Session, frame: ScreenSnapshot) -> None:
        session.history.append(frame)
        session.snapshot = frame
        self._trim_history(session)
        while self.buffered_bytes > self.max_total_history_bytes:
            oldest = min(
                (item for item in self.sessions.values() if item.history),
                key=lambda item: item.history[0].received_at,
            )
            oldest.history.pop(0)
            self._trim_history(oldest)

    def invalidate(
        self, *, guild_id: int | None = None, channel_id: int | None = None, owner_id: int | None = None
    ) -> int:
        self._revision += 1

        def matches(binding: ScreenBinding) -> bool:
            return (
                (guild_id is None or binding.guild_id == guild_id)
                and (channel_id is None or binding.channel_id == channel_id)
                and (owner_id is None or binding.owner_id == owner_id)
            )

        count = 0
        for mapping in (self.pending, self.sessions):
            for key, item in list(mapping.items()):
                if matches(item.binding):
                    if mapping is self.sessions:
                        self._drop_session(key)
                    else:
                        del mapping[key]
                    count += 1
        return count

    def expire(self) -> None:
        now = self.clock()
        for key, pending in list(self.pending.items()):
            if pending.expires_at <= now:
                del self.pending[key]
        for key, session in list(self.sessions.items()):
            if session.expires_at <= now:
                self._drop_session(key, session)
            else:
                self._trim_history(session)

    async def _valid(self, binding: ScreenBinding) -> bool:
        try:
            return not self._closed and bool(await self.validate(binding)) and not self._closed
        except Exception:
            return False

    async def snapshots(
        self, channel_id: int, audience: Audience, owner_id: int | None = None
    ) -> tuple[ScreenSnapshot, ...]:
        """Fresh frames only, for the exact call audience that explicitly opted in."""
        self.expire()
        result = []
        for key, session in list(self.sessions.items()):
            binding = session.binding
            if binding.channel_id != channel_id or binding.guild_id != audience.guild_id:
                continue
            if not await self._valid(binding):
                self._drop_session(key, session)
                continue
            if binding.audience_ids != audience.user_ids:
                # A broader text destination is not entitled to these pixels, but asking
                # there must not stop capture for the original, still-valid call audience.
                continue
            if self.sessions.get(key) is not session:
                continue
            if owner_id is not None and binding.owner_id != owner_id:
                continue
            if session.snapshot and self.clock() - session.snapshot.received_at <= self.frame_ttl:
                result.append((key, session, session.snapshot))
        # A later permission check can yield while an earlier grant is revoked.
        self.expire()
        frames = [
            frame
            for key, session, frame in result
            if self.sessions.get(key) is session and self.clock() - frame.received_at <= self.frame_ttl
        ]
        frames.sort(key=lambda frame: frame.created_at, reverse=True)
        return tuple(frames[: self.max_context_frames])

    async def history(
        self,
        channel_id: int,
        audience: Audience,
        owner_id: int | None = None,
        *,
        before: float | None = None,
        lookback: float = 8.0,
        limit: int | None = None,
    ) -> tuple[ScreenSnapshot, ...]:
        """Return chronological recent samples after validating the current call.

        ``before`` uses the service's monotonic clock, for example the time speech
        began. ``received_at`` on each immutable sample preserves its original
        ordering. Default output spans the interval with at most max_context_frames;
        a local selector may request up to max_history_frames without inference.
        """
        if not math.isfinite(lookback) or lookback <= 0:
            raise ValueError("Screen history lookback must be positive and finite")
        if before is not None and not math.isfinite(before):
            raise ValueError("Screen history reference time must be finite")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("Screen history limit must be a positive integer")
        count = min(limit if limit is not None else self.max_context_frames, self.max_history_frames)
        self.expire()
        end = min(self.clock(), before) if before is not None else self.clock()
        start = end - min(lookback, self.history_seconds)
        candidates = []
        for key, session in list(self.sessions.items()):
            binding = session.binding
            if (
                binding.channel_id != channel_id
                or binding.guild_id != audience.guild_id
                or binding.audience_ids != audience.user_ids
                or (owner_id is not None and binding.owner_id != owner_id)
            ):
                continue
            if not await self._valid(binding):
                self._drop_session(key, session)
                continue
            if self.sessions.get(key) is not session:
                continue
            candidates.extend(
                (key, session, frame) for frame in session.history if start <= frame.received_at <= end
            )
        self.expire()
        frames = sorted(
            (
                frame
                for key, session, frame in candidates
                if self.sessions.get(key) is session
                and self.clock() - frame.received_at <= self.history_seconds
            ),
            key=lambda frame: frame.received_at,
        )
        if len(frames) <= count:
            return tuple(frames)
        if count == 1:
            return (frames[-1],)
        # Include both ends of the interval instead of returning only post-event
        # near-duplicates. The caller may use fingerprints to choose evidence.
        return tuple(frames[round(index * (len(frames) - 1) / (count - 1))] for index in range(count))

    async def sources_active(self, sources: tuple[Source, ...]) -> bool:
        """Validate only live frames used for an in-flight response, not historical text memories."""
        self.expire()
        checked = []
        for source in sources:
            if source.kind != "screen":
                continue
            match = next(
                (
                    (key, session)
                    for key, session in self.sessions.items()
                    if session.binding.session_id == source.session_id
                ),
                None,
            )
            if match is None:
                return False
            key, session = match
            binding = session.binding
            if (
                binding.owner_id != source.owner_id
                or binding.guild_id != source.guild_id
                or binding.channel_id != source.channel_id
                or binding.audience_ids != frozenset(source.audience_ids)
                or not await self._valid(binding)
            ):
                return False
            if self.sessions.get(key) is not session or session.expires_at <= self.clock():
                return False
            checked.append((key, session))
        self.expire()
        return all(self.sessions.get(key) is session for key, session in checked)

    def describe(self, owner_id: int, channel_id: int) -> dict:
        """Metadata for slash status; never returns grant codes, tokens, or image content."""
        self.expire()
        matches = [
            session
            for session in self.sessions.values()
            if session.binding.owner_id == owner_id and session.binding.channel_id == channel_id
        ]
        pending = any(
            grant.binding.owner_id == owner_id and grant.binding.channel_id == channel_id
            for grant in self.pending.values()
        )
        ages = [self.clock() - session.snapshot.received_at for session in matches if session.snapshot]
        frames = [session.snapshot for session in matches if session.snapshot]
        latest = max(frames, key=lambda frame: frame.received_at) if frames else None
        return {
            "paired": bool(matches),
            "pending": pending,
            "fresh": bool(ages),
            "frames_received": sum(session.frames_received for session in matches),
            "frame_age_seconds": round(min(ages), 1) if ages else None,
            "retained_frame_size": f"{latest.width}x{latest.height}" if latest else None,
            "history_frames": sum(len(session.history) for session in matches),
            "history_bytes": sum(self._history_bytes(session) for session in matches),
            "history_oldest_age_seconds": round(
                max(
                    (self.clock() - frame.received_at for session in matches for frame in session.history),
                    default=0.0,
                ),
                1,
            ),
            "buffered_bytes": self.buffered_bytes,
            "expires_in": max(
                (max(0, int(session.expires_at - self.clock())) for session in matches), default=0
            ),
        }

    def install(self, app: web.Application) -> None:
        app.router.add_get("/screen", self.page)
        app.router.add_get("/screen/", self.page)
        app.router.add_get("/screen/app.js", self.javascript)
        app.router.add_get("/screen/style.css", self.stylesheet)
        app.router.add_post("/screen/pair", self.pair)
        app.router.add_post("/screen/frame", self.frame)
        app.router.add_post("/screen/stop", self.stop)
        app.router.add_get("/screen/status", self.status)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)

    async def _startup(self, app):
        async def purge():
            while True:
                await asyncio.sleep(2)
                self.expire()

        self._cleaner = asyncio.create_task(purge(), name="screen-frame-expiry")

    async def _cleanup(self, app):
        await self.close()

    async def close(self):
        self._closed = True
        self._revision += 1
        if self._cleaner:
            self._cleaner.cancel()
            await asyncio.gather(self._cleaner, return_exceptions=True)
            self._cleaner = None
        self.pending.clear()
        for key in list(self.sessions):
            self._drop_session(key)
        self.attempts.clear()

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; media-src 'self' blob:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        }

    @staticmethod
    def _check_origin(request: web.Request) -> None:
        origin = request.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() != request.host.casefold():
                raise web.HTTPForbidden(text="Cross-origin capture requests are not allowed")

    @staticmethod
    async def _body(request: web.Request, limit: int) -> bytes:
        # StreamReader.read(n) can return a partial network chunk, not the whole body.
        # Keep reading to EOF while enforcing the limit before allocating more data.
        data = bytearray()
        try:
            async with asyncio.timeout(15):
                while block := await request.content.read(min(65536, limit - len(data) + 1)):
                    data.extend(block)
                    if len(data) > limit:
                        raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=len(data))
        except TimeoutError:
            raise web.HTTPRequestTimeout(text="Screen upload timed out") from None
        return bytes(data)

    async def _asset(self, name: str, content_type: str):
        data = files("bentebot").joinpath("screen_assets", name).read_bytes()
        return web.Response(body=data, content_type=content_type, headers=self._headers())

    async def page(self, request):
        return await self._asset("index.html", "text/html")

    async def javascript(self, request):
        return await self._asset("app.js", "application/javascript")

    async def stylesheet(self, request):
        return await self._asset("style.css", "text/css")

    def _pair_attempt(self, request):
        now, identity = self.clock(), request.remote or "unknown"
        started, count = self.attempts.pop(identity, (now, 0))
        if now - started > 60:
            started, count = now, 0
        self.attempts[identity] = (started, count + 1)
        while len(self.attempts) > 256:
            self.attempts.popitem(last=False)
        if count >= 8:
            raise web.HTTPTooManyRequests(text="Too many pairing attempts; wait one minute")

    async def pair(self, request):
        self._check_origin(request)
        self.expire()
        self._pair_attempt(request)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(text="Use a JSON pairing request")
        body = await self._body(request, 2048)
        import json

        try:
            payload = json.loads(body)
            code = payload.get("code", "")
            if not isinstance(code, str) or len(code) > 200:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise web.HTTPBadRequest(text="Invalid pairing request") from None
        revision = self._revision
        pending = self.pending.pop(_digest(code.strip()), None)
        if not pending or not await self._valid(pending.binding) or revision != self._revision:
            raise web.HTTPUnauthorized(text="Invalid or expired pairing code; use /screen start again")
        token = secrets.token_urlsafe(32)
        self.sessions[_digest(token)] = _Session(pending.binding, self.clock() + self.session_ttl)
        return web.json_response(
            {
                "token": token,
                "expires_in": self.session_ttl,
                "owner_id": str(pending.binding.owner_id),
                "audience_ids": [str(id_) for id_ in sorted(pending.binding.audience_ids)],
                "frame_interval_ms": 1000,
                "max_frame_bytes": self.max_frame_bytes,
                "max_width": 1600,
                "max_height": 1000,
            },
            headers=self._headers(),
        )

    async def _session(self, request):
        self._check_origin(request)
        self.expire()
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or len(auth) > 200:
            raise web.HTTPUnauthorized(text="Pair this capture page first")
        key = _digest(auth[7:])
        session = self.sessions.get(key)
        if not session:
            raise web.HTTPUnauthorized(text="Screen session expired or was stopped")
        if not await self._valid(session.binding):
            self._drop_session(key, session)
            raise web.HTTPForbidden(text="Call audience changed; pair again to continue sharing")
        if self.sessions.get(key) is not session:
            raise web.HTTPForbidden(text="Screen session was stopped")
        return key, session

    async def frame(self, request):
        key, session = await self._session(request)
        if request.headers.get("Content-Encoding", "identity").casefold() != "identity":
            raise web.HTTPUnsupportedMediaType(text="Compressed HTTP request bodies are not accepted")
        if request.content_type not in {"image/jpeg", "image/png"}:
            raise web.HTTPUnsupportedMediaType(text="Send a JPEG or PNG frame")
        try:
            sequence = int(request.headers.get("X-Bentebot-Sequence", ""))
        except ValueError:
            raise web.HTTPBadRequest(text="A strictly increasing frame sequence is required") from None
        if not 0 < sequence < 2**53 or sequence <= session.sequence:
            raise web.HTTPConflict(text="Frame sequence has already been used")
        now = self.clock()
        if now - session.last_upload < 0.8:
            raise web.HTTPTooManyRequests(text="Send no more than one screen frame per second")
        if request.content_length and request.content_length > self.max_frame_bytes:
            raise web.HTTPRequestEntityTooLarge(
                max_size=self.max_frame_bytes, actual_size=request.content_length
            )
        body = await self._body(request, self.max_frame_bytes)
        session.last_upload = now
        try:
            async with self._decode_limit:
                data, width, height = await asyncio.to_thread(
                    _normalise_image, body, max_pixels=self.max_pixels
                )
        except ValueError:
            raise web.HTTPBadRequest(text="Invalid JPEG/PNG or frame dimensions exceed limits") from None
        # Do not resurrect a frame after stop, expiry or membership invalidation during decode.
        if (
            self.sessions.get(key) is not session
            or session.expires_at <= self.clock()
            or not await self._valid(session.binding)
        ):
            self._drop_session(key, session)
            raise web.HTTPForbidden(text="Screen session stopped or its audience changed")
        if self.sessions.get(key) is not session:
            raise web.HTTPForbidden(text="Screen session was stopped")
        if sequence <= session.sequence:
            raise web.HTTPConflict(text="A newer frame has already arrived")
        fingerprint = hashlib.sha256(data).hexdigest()
        if session.snapshot and session.snapshot.fingerprint == fingerprint:
            snapshot = replace(session.snapshot, created_at=time.time(), received_at=self.clock())
            changed = False
        else:
            b = session.binding
            snapshot = ScreenSnapshot(
                b.owner_id,
                b.guild_id,
                b.channel_id,
                b.audience_ids,
                b.session_id,
                time.time(),
                self.clock(),
                "data:image/jpeg;base64," + base64.b64encode(data).decode(),
                width,
                height,
                fingerprint,
            )
            changed = True
        if len(snapshot.data_url) > min(self.max_history_bytes, self.max_total_history_bytes):
            raise web.HTTPRequestEntityTooLarge(
                max_size=min(self.max_history_bytes, self.max_total_history_bytes),
                actual_size=len(snapshot.data_url),
            )
        session.sequence = sequence
        session.frames_received += 1
        self._remember(session, snapshot)
        return web.json_response({"accepted": True, "changed": changed}, headers=self._headers())

    async def stop(self, request):
        key, _ = await self._session(request)
        self._drop_session(key)
        return web.json_response({"stopped": True}, headers=self._headers())

    async def status(self, request):
        _, session = await self._session(request)
        frame = session.snapshot
        return web.json_response(
            {
                "active": True,
                "frames_received": session.frames_received,
                "frame_age_seconds": round(self.clock() - frame.received_at, 1) if frame else None,
                "expires_in": max(0, int(session.expires_at - self.clock())),
            },
            headers=self._headers(),
        )
