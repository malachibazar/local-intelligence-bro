from __future__ import annotations

import threading
from collections.abc import Generator
from dataclasses import dataclass

import torch
from pocket_tts import TTSModel

DEFAULT_VOICE = "eve"
@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    pcm_s16le: bytes
    sample_rate: int


class PocketSpeechService:
    """Load one Pocket TTS model and serialize streaming synthesis."""

    def __init__(self, *, voice: str = DEFAULT_VOICE) -> None:
        self._model = TTSModel.load_model()
        self._voice_state = self._model.get_state_for_audio_prompt(voice)
        self._lock = threading.Lock()
        self._warm_up()

    def _warm_up(self) -> None:
        for _chunk in self._model.generate_audio_stream(
            self._voice_state,
            "Hello.",
        ):
            pass

    def stream(self, text: str) -> Generator[SynthesizedAudio, None, None]:
        with self._lock:
            for audio_chunk in self._model.generate_audio_stream(
                self._voice_state,
                text,
            ):
                samples = (
                    audio_chunk.detach()
                    .cpu()
                    .clamp(-1, 1)
                    .mul(32_767)
                    .to(torch.int16)
                    .contiguous()
                    .numpy()
                )
                yield SynthesizedAudio(
                    pcm_s16le=samples.astype("<i2", copy=False).tobytes(),
                    sample_rate=self._model.sample_rate,
                )
