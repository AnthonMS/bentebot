import httpx
import pytest

from bentebot.config import Settings
from bentebot.providers import OpenAIChat, OpenAITTS, ProviderError
from bentebot.tools import TOOL_TRACE


@pytest.mark.parametrize("operation", ["chat", "tts"])
async def test_transport_failure_logs_status_host_and_trace_without_sensitive_data(caplog, operation):
    caplog.set_level("WARNING", logger="bentebot.providers")
    settings = Settings(
        chat_base_url="https://gateway.example/SECRET_PATH/v1",
        chat_api_key="SECRET_KEY",
        tts_base_url="https://speech.example/SECRET_PATH/v1",
        tts_api_key="SECRET_KEY",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="SECRET_BODY private transcript")
        )
    ) as client:
        provider = OpenAIChat(settings, client) if operation == "chat" else OpenAITTS(settings, client)
        token = TOOL_TRACE.set("012345abcdef")
        try:
            with pytest.raises(ProviderError) as error:
                if operation == "chat":
                    _ = [
                        piece
                        async for piece in provider.stream(
                            [{"role": "user", "content": "SECRET_TRANSCRIPT"}], "test"
                        )
                    ]
                else:
                    await provider.synthesize("SECRET_TRANSCRIPT")
            assert "SECRET" not in str(error.value)
            assert f"operation={operation}" in caplog.text
            assert "turn=012345abcdef" in caplog.text
            assert "error=HTTPStatusError" in caplog.text
            assert "http_status=500" in caplog.text
            assert f"host={'gateway' if operation == 'chat' else 'speech'}.example" in caplog.text
            assert "SECRET" not in caplog.text
            assert "private transcript" not in caplog.text
            assert "https://" not in caplog.text
        finally:
            TOOL_TRACE.reset(token)
            await provider.close()


async def test_invalid_wav_logs_safe_error_type_without_provider_payload(caplog):
    caplog.set_level("WARNING", logger="bentebot.providers")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"SECRET invalid format"))
    ) as client:
        provider = OpenAITTS(Settings(tts_base_url="http://tts:8000/v1"), client)
        with pytest.raises(ProviderError):
            await provider.synthesize("PRIVATE utterance")
        assert "operation=tts" in caplog.text and "error=ProviderError" in caplog.text
        assert "http_status=unreported" in caplog.text
        assert "SECRET" not in caplog.text and "PRIVATE" not in caplog.text


async def test_cancelled_provider_request_is_not_logged_as_a_failure(caplog):
    import asyncio

    async def handler(request):
        raise asyncio.CancelledError

    caplog.set_level("WARNING", logger="bentebot.providers")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(Settings(), client)
        with pytest.raises(asyncio.CancelledError):
            _ = [piece async for piece in provider.stream([], "test")]
        assert "provider_failure" not in caplog.text
        await provider.close()
