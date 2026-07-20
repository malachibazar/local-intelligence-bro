from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

WHISPER_MODEL = "base.en"
SAMPLE_RATE = 16_000


class NoSpeechDetectedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str
    language_probability: float


class WhisperSpeechRecognitionService:
    """Lazily load and retain one CPU Faster Whisper model."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    def transcribe(self, pcm_s16le: bytes) -> Transcript:
        with self._lock:
            if self._model is None:
                self._model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                )

            audio = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32)
            audio /= 32_768
            segments, info = self._model.transcribe(
                audio,
                language="en",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                raise NoSpeechDetectedError("No speech was detected")

            return Transcript(
                text=text,
                language=info.language,
                language_probability=info.language_probability,
            )
