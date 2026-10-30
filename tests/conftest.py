import io
import wave

import pytest

from bentebot.storage import Store


@pytest.fixture
def store():
    value = Store(":memory:")
    yield value
    value.close()


@pytest.fixture
def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\1" * 2400)
    return output.getvalue()
