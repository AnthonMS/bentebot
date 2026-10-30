"""A CPU Danish voice beside the existing English TTS, with local text routing."""

import io
import wave
from pathlib import Path


class DanishTTS:
    def __init__(self, directory):
        from lingua import Language, LanguageDetectorBuilder
        from piper import PiperVoice

        self.danish = Language.DANISH
        self.detector = LanguageDetectorBuilder.from_languages(Language.DANISH, Language.ENGLISH).build()
        self.voice = PiperVoice.load(str(Path(directory) / "da_DK-talesyntese-medium.onnx"), use_cuda=False)

    def language(self, text):
        scores = self.detector.compute_language_confidence_values(text)
        # Very short ambiguous acknowledgements keep the existing English route.
        if not scores or scores[0].value < 0.75:
            return "unknown"
        return "da" if scores[0].language == self.danish else "en"

    def synthesize(self, text):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            self.voice.synthesize_wav(text, wav)
        result = output.getvalue()
        if len(result) > 4_000_000:
            raise ValueError("Danish audio exceeded the output limit.")
        return result
