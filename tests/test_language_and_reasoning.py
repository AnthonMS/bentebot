import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from bentebot.config import Settings
from bentebot.providers import OpenAIChat


@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("selected", ["local-fast", "cloud"])
async def test_template_thinking_control_is_explicitly_alias_scoped(classifier, selected):
    def handler(request):
        payload = json.loads(request.content)
        assert ("chat_template_kwargs" in payload) == (selected == "local-fast")
        if selected == "local-fast":
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        if classifier:
            return httpx.Response(200, json={"choices": [{"message": {"content": "leave"}}]})
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"Hello."}}]}\n\ndata: [DONE]\n\n'
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIChat(Settings(chat_template_no_thinking_models="local-fast"), client)
        if classifier:
            assert await provider.resolve_spoken_control("Leave the call Bender", selected) == "leave"
        else:
            assert [piece async for piece in provider.stream([], selected)] == ["Hello."]


@pytest.mark.parametrize("redetect", ["true", "false"])
def test_segment_redetection_switch_does_not_force_a_language(monkeypatch, redetect):
    tune = runpy.run_path(str(Path(__file__).parents[1] / "deploy/speech/tune_backend.py"))["tuned_source"]
    source = """def transcribe(self, audio):
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type
        return self.model.transcribe(audio,
            beam_size=5,
            language=self.original_language,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
"""
    namespace = {}
    exec(tune(source), namespace)
    monkeypatch.setenv("WLK_MULTILINGUAL", redetect)
    backend = SimpleNamespace(
        original_language=None,
        transcribe_kargs={},
        model=SimpleNamespace(transcribe=lambda audio, **kwargs: kwargs),
    )
    options = namespace["transcribe"](backend, b"")
    assert options["language"] is None  # faster-whisper still automatically detects.
    assert options["multilingual"] == (redetect == "true")
    backend.original_language = "da"
    options = namespace["transcribe"](backend, b"")
    assert options["language"] == "da" and not options["multilingual"]


def test_voice_first_answer_budget_must_be_positive():
    with pytest.raises(ValueError, match="voice_first_content_timeout"):
        Settings(voice_first_content_timeout=0).validate(login=False)
