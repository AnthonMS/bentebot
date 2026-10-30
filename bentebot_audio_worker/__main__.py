"""Authenticated audio service; no raw recordings or request body logs."""

import asyncio
import hmac
import os

from aiohttp import web

from . import __version__
from .danish_tts import DanishTTS
from .engine import Engine


def application(engine, token, danish=None):
    if len(token) < 24:
        raise ValueError("Configure an audio service token of at least 24 characters.")
    app = web.Application(client_max_size=192000)
    slot = asyncio.Semaphore(1)
    speech_slot = asyncio.Semaphore(1)

    async def health(request):
        return web.json_response(
            {
                "ready": True,
                "version": __version__,
                "sound": "yamnet",
                "tone": "emotion2vec+ base",
                **engine.details,
                "danish_tts_device": "cpu",
                "yawn": "not_supported",
            }
        )

    async def analyze(request):
        if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
            raise web.HTTPUnauthorized()
        if slot.locked():
            raise web.HTTPTooManyRequests()
        pcm = await request.read()
        if len(pcm) % 2 or not 16000 <= len(pcm) <= 192000:
            raise web.HTTPBadRequest(text="Use 0.5-6 seconds of 16kHz mono PCM16.")
        async with slot:
            try:
                result = await asyncio.to_thread(
                    engine.analyze, pcm, request.headers.get("X-Analyze-Tone") == "1"
                )
            except Exception:
                raise web.HTTPServiceUnavailable(text="Audio analysis unavailable.") from None
        return web.json_response(result)

    async def speech(request):
        if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
            raise web.HTTPUnauthorized()
        if danish is None:
            raise web.HTTPServiceUnavailable(text="Danish voice is unavailable.")
        if speech_slot.locked():
            raise web.HTTPTooManyRequests()
        try:
            payload = await request.json()
            text = payload["input"]
            if not isinstance(text, str) or not text.strip() or len(text) > 1000:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise web.HTTPBadRequest(text="Provide 1-1000 characters of speech text.") from None
        language = await asyncio.to_thread(danish.language, text)
        if language != "da":
            return web.Response(status=204, headers={"X-Speech-Language": language})
        async with speech_slot:
            try:
                wav = await asyncio.to_thread(danish.synthesize, text)
            except Exception:
                raise web.HTTPServiceUnavailable(text="Danish synthesis unavailable.") from None
        return web.Response(body=wav, content_type="audio/wav", headers={"X-Speech-Language": "da"})

    app.router.add_get("/health", health)
    app.router.add_post("/v1/analyze", analyze)
    app.router.add_post("/v1/audio/speech", speech)
    return app


if __name__ == "__main__":
    engine = Engine(
        os.environ.get("BENTEBOT_AUDIO_MODEL_DIR", "/models"),
        device=os.environ.get("BENTEBOT_AUDIO_DEVICE", "cpu"),
        gpu_memory_mb=int(os.environ.get("BENTEBOT_AUDIO_GPU_MEMORY_MB", "1024")),
    )
    web.run_app(
        application(
            engine,
            os.environ.get("BENTEBOT_AUDIO_TOKEN", ""),
            DanishTTS(os.environ.get("BENTEBOT_AUDIO_MODEL_DIR", "/models")),
        ),
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "18769")),
        access_log=None,
    )
