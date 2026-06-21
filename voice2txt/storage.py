"""Persist transcription results to disk."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from voice2txt.config import Settings

logger = logging.getLogger(__name__)


def save_transcript(text: str, settings: Settings | None = None, filename: str | None = None) -> Path:
    """
    Save transcribed text to the configured output directory.
    Returns path to the written file.
    """
    settings = settings or Settings.load()
    output_dir = settings.output_path
    if filename:
        target = output_dir / filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = output_dir / f"transcription_{timestamp}.txt"

    target.write_text(text, encoding="utf-8")
    logger.info("Saved transcript to %s", target)
    return target
