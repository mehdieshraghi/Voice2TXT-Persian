"""Audio utilities — normalize WAV data for Vosk without temp files."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit


def _read_wav_file(source: wave.Wave_read) -> tuple[np.ndarray, int, int]:
    channels = source.getnchannels()
    sample_width = source.getsampwidth()
    sample_rate = source.getframerate()
    frames = source.readframes(source.getnframes())

    if sample_width == 1:
        dtype = np.uint8
        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        audio = (audio - 128) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate, sample_width


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(audio) == 0:
        return audio
    duration = len(audio) / source_rate
    target_length = int(duration * target_rate)
    if target_length <= 0:
        return np.array([], dtype=np.float32)
    source_times = np.linspace(0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0, duration, num=target_length, endpoint=False)
    return np.interp(target_times, source_times, audio).astype(np.float32)


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


def normalize_for_vosk(
    source: bytes | Path | str,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> bytes:
    """
    Read WAV from bytes or path and return PCM16 mono frames at target rate.
    Suitable for passing directly to KaldiRecognizer.AcceptWaveform.
    """
    if isinstance(source, (str, Path)):
        with wave.open(str(source), "rb") as wf:
            audio, sample_rate, _ = _read_wav_file(wf)
    else:
        with wave.open(io.BytesIO(source), "rb") as wf:
            audio, sample_rate, _ = _read_wav_file(wf)

    audio = _resample(audio, sample_rate, target_sample_rate)
    return _float_to_pcm16(audio)


def pcm16_to_wav_bytes(pcm_data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 mono frames in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(TARGET_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Convert int16 numpy array from sounddevice to WAV bytes."""
    if audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    return pcm16_to_wav_bytes(audio.tobytes(), sample_rate)
