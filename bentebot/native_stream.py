"""Passive voice media inspection and a bounded Go Live bot-token access experiment.

No user tokens, capture helpers, frame recording, or automatic retries. Stream
credentials are inspected only for presence and never retained in results.
The passive inspector counts existing UDP headers; neither tool decodes video.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceMediaObservation:
    seconds: float
    rtp_types: tuple[tuple[int, int], ...]
    rtcp_types: tuple[tuple[int, int], ...]
    other_headers: int

    def describe(self) -> str:
        def counts(items):
            return ", ".join(f"{kind}={count}" for kind, count in items) or "none"

        return (
            f"Passive voice media inspection ({self.seconds:g}s)\n"
            f"RTP payload types: {counts(self.rtp_types)}\n"
            f"RTCP packet types: {counts(self.rtcp_types)}\n"
            f"Other/short headers: {self.other_headers}\n"
            "RTP 120 is Discord Opus audio; other types require investigation. "
            "No media was decrypted, decoded or saved by this inspection. "
            "Go Live normally uses a separate connection; these counts do not prove screen access."
        )


class VoiceMediaAudit:
    """Count existing UDP headers without changing voice negotiation or receiving audio twice."""

    def __init__(self, *, seconds: float = 12):
        if not 0 < seconds <= 30:
            raise ValueError("Voice inspection must last between zero and 30 seconds.")
        self.seconds = seconds
        self._busy = False

    async def collect(self, voice_client) -> VoiceMediaObservation:
        if self._busy:
            raise ValueError("A passive voice inspection is already running.")
        if not voice_client or not voice_client.is_connected():
            raise ValueError("The voice connection is not connected.")
        # The pinned discord.py/voice-recv already uses this listener API. Observe
        # the same datagrams; never socket.recv(), decrypt or replace its reader.
        connection = getattr(voice_client, "_connection", None)
        if not all(
            callable(getattr(connection, name, None))
            for name in ("add_socket_listener", "remove_socket_listener")
        ):
            raise ValueError("This Discord library does not support passive voice inspection.")
        rtp, rtcp = Counter(), Counter()
        other = 0
        lock = threading.Lock()

        def observe(data):
            nonlocal other
            with lock:
                if len(data) < 2 or data[0] >> 6 != 2:
                    other += 1
                elif 192 <= data[1] <= 223 and len(data) >= 8:
                    rtcp[data[1]] += 1
                elif len(data) >= 12 + 4 * (data[0] & 15) and not 192 <= data[1] <= 223:
                    rtp[data[1] & 127] += 1
                else:
                    other += 1

        self._busy = True
        registered = False
        try:
            connection.add_socket_listener(observe)
            registered = True
            await asyncio.sleep(self.seconds)
            with lock:
                return VoiceMediaObservation(
                    self.seconds, tuple(sorted(rtp.items())), tuple(sorted(rtcp.items())), other
                )
        finally:
            try:
                if registered:
                    with contextlib.suppress(ValueError):
                        connection.remove_socket_listener(observe)
            finally:
                self._busy = False


@dataclass(frozen=True)
class StreamProbeResult:
    outcome: str
    stream_created: bool = False
    server_allocated: bool = False
    reason: str = ""

    def describe(self) -> str:
        return (
            f"Native stream probe: {self.outcome}\n"
            f"Stream created: {self.stream_created}; server allocated: {self.server_allocated}"
            + (f"; reason: {self.reason}" if self.reason else "")
            + "\nThis checks gateway access only. No video frames were received or viewed."
        )


class NativeStreamProbe:
    def __init__(self, client, *, timeout: float = 8, cooldown: float = 60):
        self.client, self.timeout, self.cooldown = client, timeout, cooldown
        self._busy = False
        self._next = 0.0
        self._disabled_after_gateway_failure = False

    async def run(self, guild_id: int, channel_id: int, owner_id: int) -> StreamProbeResult:
        if getattr(getattr(self.client, "user", None), "id", None) == owner_id:
            raise ValueError("The bot cannot probe its own stream.")
        loop = asyncio.get_running_loop()
        if self._disabled_after_gateway_failure:
            raise ValueError(
                "Stream probing was disabled after a gateway failure. "
                "Keep the experiment off until the failure has been reviewed."
            )
        if self._busy or loop.time() < self._next:
            raise ValueError("A stream probe is active or cooling down; wait a minute.")
        socket = self.client.ws
        if socket is None:
            raise ValueError("The Discord gateway is not connected.")
        self._busy = True
        self._next = loop.time() + self.cooldown
        key = f"guild:{guild_id}:{channel_id}:{owner_id}"
        done = loop.create_future()
        created, allocated = False, False
        gateway_usable = True

        def finish(outcome, reason=""):
            if not done.done():
                done.set_result(StreamProbeResult(outcome, created, allocated, reason))

        async def receive(raw):
            nonlocal created, allocated, gateway_usable
            if done.done() or not isinstance(raw, str) or len(raw) > 1_000_000:
                return
            try:
                packet = json.loads(raw)
            except (ValueError, TypeError):
                return
            if not isinstance(packet, dict):
                return
            if packet.get("op") == 9:
                # INVALID_SESSION is a gateway control packet, not a stream dispatch.
                # Do not send viewer cleanup on a session Discord has invalidated.
                gateway_usable = False
                self._disabled_after_gateway_failure = True
                finish("inconclusive", "gateway_invalid_session")
                return
            if packet.get("op") != 0:
                return
            data = packet.get("d")
            if not isinstance(data, dict) or data.get("stream_key") != key:
                return
            event = packet.get("t")
            if event == "STREAM_CREATE":
                created = bool(data.get("rtc_server_id"))
            elif event == "STREAM_SERVER_UPDATE":
                allocated = all(
                    isinstance(data.get(name), str) and data[name] for name in ("endpoint", "token")
                )
            elif event == "STREAM_DELETE":
                reason = data.get("reason")
                known = {
                    "unauthorized",
                    "stream_ended",
                    "stream_full",
                    "user_requested",
                    "safety_guild_rate_limited",
                    "parse_failed",
                    "invalid_channel",
                }
                finish(
                    "denied" if reason == "unauthorized" else "ended",
                    reason if isinstance(reason, str) and reason in known else "other",
                )
            if created and allocated:
                finish("gateway_access_observed")

        async def disconnect():
            nonlocal gateway_usable
            gateway_usable = False
            self._disabled_after_gateway_failure = True
            finish("inconclusive", "gateway_disconnected")

        self.client.add_listener(receive, "on_socket_raw_receive")
        self.client.add_listener(disconnect, "on_disconnect")
        attempted = False
        try:
            attempted = True
            async with asyncio.timeout(self.timeout):
                await socket.send_as_json({"op": 20, "d": {"stream_key": key}})
                return await asyncio.shield(done)
        except TimeoutError:
            return StreamProbeResult("inconclusive", created, allocated, "timeout")
        except asyncio.CancelledError:
            raise
        except Exception:
            gateway_usable = False
            self._disabled_after_gateway_failure = True
            return StreamProbeResult("inconclusive", created, allocated, "gateway_error")
        finally:
            self.client.remove_listener(receive, "on_socket_raw_receive")
            self.client.remove_listener(disconnect, "on_disconnect")
            if not done.done():
                done.cancel()
            try:
                if (
                    attempted
                    and gateway_usable
                    and self.client.ws is socket
                    and getattr(socket, "open", True)
                ):
                    # Opcode 19 disconnects this viewer; this bot is never the owner.
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(3):
                            await socket.send_as_json({"op": 19, "d": {"stream_key": key}})
            finally:
                self._busy = False
