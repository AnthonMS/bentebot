import io
import json
import wave

import httpx
import pytest

from bentebot.config import Settings
from bentebot.providers import OpenAITTS, ProviderError
from bentebot.voice_profiles import KittenVoiceProfiles

VOICE_ID = "voice_" + "a" * 32


def sample(seconds=5):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\1" * int(seconds * 16000))
    return output.getvalue()


async def test_saved_profile_speech_uses_kitten_id_object_and_complete_wav():
    def handler(request):
        body = json.loads(request.content)
        assert body["voice"] == {"id": VOICE_ID}
        assert body["stream"] is False and body["response_format"] == "wav"
        return httpx.Response(200, content=sample())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tts = OpenAITTS(Settings(tts_base_url="http://tts/v1", voice_profiles_enabled=True), client=client)
        assert (await tts.synthesize("Hello", VOICE_ID)).startswith(b"RIFF")


async def test_profiles_match_official_kitten_api_contract():
    requests = []

    async def handler(request):
        requests.append(request)
        if request.method == "GET":
            assert request.url.path == "/v1/voices"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "Bruno", "name": "Bruno", "type": "built_in"},
                        {"id": VOICE_ID, "name": "My own voice", "type": "custom"},
                    ]
                },
            )
        if request.method == "POST":
            assert request.url.path == "/v1/audio/voices"
            assert b'name="audio_sample"' in request.content
            assert b'name="reference_transcript"' in request.content
            assert b"Words actually spoken" in request.content
            return httpx.Response(200, json={"id": VOICE_ID, "name": "My own voice"})
        assert request.method == "DELETE" and request.url.path.endswith(VOICE_ID)
        return httpx.Response(200, json={"deleted": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        profiles = KittenVoiceProfiles("http://tts/v1", client=client)
        result = await profiles.list()
        assert result[0].id == "Bruno" and not result[0].custom
        assert result[1].custom
        created = await profiles.create(
            "My own voice", sample(), consent_confirmed=True, reference_transcript="Words actually spoken"
        )
        assert created.id == VOICE_ID
        await profiles.delete(created.id)
    assert len(requests) == 3


async def test_consent_and_sample_validation_prevent_uploads():
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    ) as client:
        profiles = KittenVoiceProfiles("http://tts/v1", client=client)
        with pytest.raises(ValueError, match="consent"):
            await profiles.create("Me", sample())
        for audio in (b"not WAV", sample(1), sample(31)):
            with pytest.raises(ValueError):
                await profiles.create("Me", audio, consent_confirmed=True)
        with pytest.raises(ValueError):
            await profiles.delete("Bruno")
    assert calls == []


async def test_provider_errors_do_not_disclose_reference_audio_or_bodies():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="PRIVATE SAMPLE TRANSCRIPT"),
        )
    ) as client:
        profiles = KittenVoiceProfiles("http://tts/v1", client=client)
        with pytest.raises(ProviderError) as error:
            await profiles.create("Me", sample(), consent_confirmed=True)
        assert "PRIVATE" not in str(error.value)
