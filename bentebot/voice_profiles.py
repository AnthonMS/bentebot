"""Explicit, opt-in adapter for the official self-hosted KittenTTS 2 voice API.

This module does not capture audio or infer consent. Discord command code must establish
speaker identity, obtain that person's consent, and enforce ownership before deletion/use.
Saved reference recordings persist on the external service until explicitly deleted.
"""

from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass

import httpx

from .providers import ProviderError

CUSTOM_VOICE = re.compile(r"voice_[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    custom: bool


class KittenVoiceProfiles:
    def __init__(self, base_url: str, api_key: str = "", *, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(timeout=60)
        self._owns_client = client is None

    async def list(self) -> list[VoiceProfile]:
        try:
            response = await self.client.get(f"{self.base_url}/voices", headers=self.headers)
            response.raise_for_status()
            data = response.json()["data"]
            return [
                VoiceProfile(str(item["id"]), str(item["name"]), item.get("type") == "custom")
                for item in data
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                "Voice profiles unavailable; this requires the configured KittenTTS voice API."
            ) from exc

    async def create(
        self,
        name: str,
        wav_sample: bytes,
        *,
        consent_confirmed: bool = False,
        reference_transcript: str | None = None,
    ) -> VoiceProfile:
        if not consent_confirmed:
            raise ValueError(
                "The selected speaker must explicitly consent before a voice profile is created."
            )
        name = name.strip()
        if not 1 <= len(name) <= 100:
            raise ValueError("Voice profile names must contain 1-100 characters.")
        self._validate_sample(wav_sample)
        data = {"name": name}
        if reference_transcript and reference_transcript.strip():
            data["reference_transcript"] = reference_transcript.strip()
        try:
            response = await self.client.post(
                f"{self.base_url}/audio/voices",
                headers=self.headers,
                data=data,
                files={"audio_sample": ("reference.wav", wav_sample, "audio/wav")},
            )
            response.raise_for_status()
            voice = response.json()
            voice_id = str(voice["id"])
            if not CUSTOM_VOICE.fullmatch(voice_id):
                raise ValueError("Invalid custom voice ID")
            return VoiceProfile(voice_id, str(voice["name"]), True)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError("Voice profile creation failed; check the external TTS service.") from exc

    async def delete(self, voice_id: str) -> None:
        if not CUSTOM_VOICE.fullmatch(voice_id):
            raise ValueError("Only a valid saved custom voice can be deleted.")
        try:
            response = await self.client.delete(
                f"{self.base_url}/audio/voices/{voice_id}", headers=self.headers
            )
            response.raise_for_status()
            if response.json().get("deleted") is not True:
                raise ValueError("Deletion was not confirmed")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise ProviderError("Voice deletion was not confirmed by the external TTS service.") from exc

    @staticmethod
    def _validate_sample(sample: bytes) -> None:
        if len(sample) > 6_000_000:
            raise ValueError("Voice samples must be at most 6 MB.")
        try:
            with wave.open(io.BytesIO(sample), "rb") as wav:
                rate, frames = wav.getframerate(), wav.getnframes()
                if wav.getnchannels() not in {1, 2} or wav.getsampwidth() != 2 or rate <= 0:
                    raise ValueError("Voice samples must be mono/stereo 16-bit PCM WAV.")
                if not 5 <= frames / rate <= 30:
                    raise ValueError("Capture 5-30 seconds of the consenting speaker.")
                if len(wav.readframes(frames)) != frames * wav.getnchannels() * 2:
                    raise ValueError("The voice sample WAV is truncated.")
        except (wave.Error, EOFError) as exc:
            raise ValueError("The voice sample must be a valid PCM WAV.") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
