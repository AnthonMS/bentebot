"""Warm local speech APIs for the lifetime of voice sessions, without playing audio."""

from __future__ import annotations

import ipaddress
from dataclasses import replace
from urllib.parse import urlsplit

from .config import Settings
from .model_warmth import VoiceModelWarmth, WarmState
from .providers import OpenAITTS
from .speech import wav_to_pcm


class _SpeechWarmProvider:
    def __init__(self, tts: OpenAITTS, host: str):
        self.tts, self.host = tts, host

    async def warm(self, voice: str) -> str:
        # Exercise synthesis and decoding; never enqueue playback or save a message.
        wav_to_pcm(await self.tts.synthesize("Ready.", voice))
        return self.host

    def recently_used(self, voice: str, seconds: float) -> bool:
        return self.tts.recently_used(voice, seconds)


class VoiceTTSWarmth(VoiceModelWarmth):
    def __init__(self, settings: Settings, tts: OpenAITTS):
        host = urlsplit(settings.tts_base_url).hostname or ""
        try:
            address = ipaddress.ip_address(host)
            local = address.is_private or address.is_loopback
        except ValueError:
            local = bool(host) and (
                host == "localhost" or "." not in host or host.endswith((".local", ".lan", ".internal"))
            )
        self.enabled = bool(settings.tts_warm_enabled and local and settings.tts_model)
        super().__init__(
            replace(
                settings,
                voice_warm_models="",
                voice_warm_interval=settings.tts_warm_interval,
                voice_warm_timeout=settings.tts_warm_timeout,
            ),
            _SpeechWarmProvider(tts, host),
            kind="tts",
        )

    def select(self, owner: int, model: str) -> WarmState | None:
        if self.enabled and model:
            self.allowed.add(model)
        return super().select(owner, model)
