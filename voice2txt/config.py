"""Application settings — load/save JSON or YAML, extensible for custom apps."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_FILENAME = "settings.json"

DEFAULTS: dict[str, Any] = {
    "model_path": "vosk-model-small-fa-0.42",
    "sample_rate": 16000,
    "record_duration": 5,
    "output_dir": "output",
    "chunk_frames": 4000,
    "max_upload_mb": 25,
    "models_catalog_url": "",
}


@dataclass
class Settings:
    """Runtime configuration for transcription pipeline."""

    model_path: str = DEFAULTS["model_path"]
    sample_rate: int = DEFAULTS["sample_rate"]
    record_duration: int = DEFAULTS["record_duration"]
    output_dir: str = DEFAULTS["output_dir"]
    chunk_frames: int = DEFAULTS["chunk_frames"]
    max_upload_mb: int = DEFAULTS["max_upload_mb"]
    models_catalog_url: str = DEFAULTS["models_catalog_url"]
    _path: Path | None = field(default=None, repr=False, compare=False)

    @property
    def settings_path(self) -> Path | None:
        return self._path

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_model_path(self, base_dir: Path | None = None) -> Path:
        """Resolve model path relative to project root or settings file."""
        raw = Path(self.model_path)
        if raw.is_absolute():
            return raw
        anchor = base_dir or (self._path.parent if self._path else Path.cwd())
        return (anchor / raw).resolve()

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        merged = {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
        return cls(**merged)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        """Load settings from JSON file, or return defaults if missing."""
        settings_path = Path(path) if path else Path(DEFAULT_SETTINGS_FILENAME)
        if settings_path.is_file():
            with open(settings_path, encoding="utf-8") as f:
                data = json.load(f)
            settings = cls.from_dict(data)
            settings._path = settings_path.resolve()
            return settings

        settings = cls.from_dict({})
        settings._path = settings_path.resolve()
        return settings

    def save(self, path: str | Path | None = None) -> Path:
        """Persist settings to JSON."""
        target = Path(path) if path else (self._path or Path(DEFAULT_SETTINGS_FILENAME))
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        self._path = target.resolve()
        return target

    def update(self, data: dict[str, Any]) -> None:
        """Apply partial updates (e.g. from web form)."""
        for key, value in data.items():
            if key not in DEFAULTS:
                continue
            if key in ("sample_rate", "record_duration", "chunk_frames", "max_upload_mb"):
                setattr(self, key, int(value))
            else:
                setattr(self, key, str(value) if key in ("model_path", "output_dir", "models_catalog_url") else value)

    def validate_settings(self) -> list[str]:
        """Validate non-model settings (allows saving while model is missing)."""
        errors: list[str] = []
        if self.sample_rate <= 0:
            errors.append("sample_rate must be positive")
        if self.record_duration <= 0:
            errors.append("record_duration must be positive")
        if self.max_upload_mb <= 0:
            errors.append("max_upload_mb must be positive")
        return errors

    def validate_model(self) -> list[str]:
        """Validate that the Vosk model directory exists."""
        model = self.resolved_model_path()
        if not model.is_dir():
            return [f"Model not found: {model}"]
        return []

    def validate(self) -> list[str]:
        """Return list of validation errors (empty if OK)."""
        return self.validate_settings() + self.validate_model()


def get_project_root() -> Path:
    """Project root (directory containing voice2txt package)."""
    return Path(__file__).resolve().parent.parent
