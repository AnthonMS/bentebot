"""Audio-only compatibility boundary for the pinned Discord receive extension.

The extension's video event parser requires legacy optional fields. Modern
receive-only peers may omit them; a KeyError there kills the voice WS poller
and prevents subsequent DAVE commits and heartbeat ACKs being processed.
"""

from __future__ import annotations

import logging

from discord.ext import voice_recv
from discord.ext.voice_recv import gateway
from discord.voice_state import VoiceConnectionState

log = logging.getLogger(__name__)


async def audio_gateway_hook(socket, payload):
    if payload.get("op") != gateway.VIDEO:
        await gateway.hook(socket, payload)
        return
    # Video is handled by the isolated viewer. This bot only needs the audio
    # SSRC from the peer update, not its optional camera/simulcast descriptions.
    data = payload.get("d")
    try:
        if not isinstance(data, dict):
            raise ValueError
        user_id, ssrc = int(data["user_id"]), int(data["audio_ssrc"])
        if not 0 < user_id < 2**64 or not 0 <= ssrc < 2**32:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        log.warning("Ignored malformed peer audio mapping on a video-state event")
        return
    if ssrc:
        socket._connection.voice_client._add_ssrc(user_id, ssrc)


class AudioReceiveClient(voice_recv.VoiceRecvClient):
    def create_connection_state(self):
        return VoiceConnectionState(self, hook=audio_gateway_hook)
