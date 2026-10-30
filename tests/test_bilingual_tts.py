import httpx
import pytest

from bentebot.config import Settings
from bentebot.providers import OpenAITTS


@pytest.mark.parametrize("mode", ["danish", "english", "unavailable"])
async def test_danish_cpu_routing_preserves_existing_english_and_fallback(wav_bytes, mode):
    hosts = []

    def transport(request):
        hosts.append(request.url.host)
        if request.url.host == "danish":
            if mode == "danish":
                return httpx.Response(200, content=wav_bytes)
            return httpx.Response(204 if mode == "english" else 503)
        assert request.url.host == "english"
        return httpx.Response(200, content=wav_bytes)

    settings = Settings(
        tts_base_url="http://english/v1",
        tts_danish_base_url="http://danish/v1",
        audio_token="test",
        tts_voice="af_heart",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        tts = OpenAITTS(settings, client)
        assert await tts.synthesize("En dansk sætning.") == wav_bytes
    assert hosts == (["danish"] if mode == "danish" else ["danish", "english"])
