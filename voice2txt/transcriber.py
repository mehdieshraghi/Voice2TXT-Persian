"""Vosk-based speech-to-text with cached model loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vosk import KaldiRecognizer, Model

from voice2txt.audio import normalize_for_vosk
from voice2txt.config import Settings

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Persian speech-to-text using Vosk.

    Model is loaded once and reused. Instantiate with your own Settings
    or swap this class in a custom pipeline.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self._model: Model | None = None
        self._loaded_path: Path | None = None

    def _ensure_model(self) -> Model:
        model_path = self.settings.resolved_model_path()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'. "
                "Download vosk-model-small-fa-0.5 and set model_path in settings."
            )
        if self._model is None or self._loaded_path != model_path:
            logger.info("Loading Vosk model from %s", model_path)
            self._model = Model(str(model_path))
            self._loaded_path = model_path
        return self._model

    def reload_model(self) -> None:
        """Force reload after settings change (e.g. new model_path)."""
        self._model = None
        self._loaded_path = None

    def transcribe_pcm(
        self,
        pcm_data: bytes,
        sample_rate: int | None = None,
    ) -> str:
        """Transcribe raw PCM16 mono audio at the given sample rate."""
        sample_rate = sample_rate or self.settings.sample_rate
        model = self._ensure_model()
        recognizer = KaldiRecognizer(model, sample_rate)

        chunk_size = self.settings.chunk_frames * 2  # 16-bit = 2 bytes per frame
        parts: list[str] = []

        for offset in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[offset : offset + chunk_size]
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                if text := result.get("text", "").strip():
                    parts.append(text)

        final = json.loads(recognizer.FinalResult())
        if text := final.get("text", "").strip():
            parts.append(text)

        return " ".join(parts).strip()

    def transcribe_wav(self, wav_source: bytes | Path | str) -> str:
        """Transcribe WAV bytes or file path (auto-resampled to 16 kHz mono)."""
        pcm = normalize_for_vosk(wav_source, self.settings.sample_rate)
        return self.transcribe_pcm(pcm, self.settings.sample_rate)

    def transcribe_file(self, file_path: Path | str) -> str:
        """Transcribe an audio file on disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        suffix = path.suffix.lower()
        if suffix != ".wav":
            raise ValueError(
                f"Unsupported format '{suffix}'. Convert to WAV (16 kHz mono recommended)."
            )
        return self.transcribe_wav(path)

    def is_model_ready(self) -> bool:
        from voice2txt.models import is_valid_vosk_model

        return is_valid_vosk_model(self.settings.resolved_model_path())
