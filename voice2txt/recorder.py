"""Microphone recording via sounddevice (server-side / CLI)."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from voice2txt.audio import numpy_to_wav_bytes
from voice2txt.config import Settings

logger = logging.getLogger(__name__)


class Recorder:
    """Record audio from the system default microphone."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()

    def list_devices(self) -> list[dict]:
        """Return available input devices for UI or CLI."""
        devices = sd.query_devices()
        result = []
        for index, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                result.append(
                    {
                        "index": index,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "default_samplerate": device["default_samplerate"],
                    }
                )
        return result

    def record(
        self,
        duration: float | None = None,
        sample_rate: int | None = None,
        device: int | None = None,
    ) -> bytes:
        """
        Record from microphone and return WAV bytes (16-bit mono).
        Raises sd.PortAudioError if no microphone is available.
        """
        duration = duration if duration is not None else self.settings.record_duration
        sample_rate = sample_rate or self.settings.sample_rate

        logger.info("Recording %.1fs at %d Hz", duration, sample_rate)
        frames = int(duration * sample_rate)
        audio_data = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
            device=device,
        )
        sd.wait()
        logger.info("Recording completed")
        return numpy_to_wav_bytes(audio_data, sample_rate)
