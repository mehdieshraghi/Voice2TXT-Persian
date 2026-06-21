"""Voice2TXT-Persian — modular Persian speech-to-text library."""

from voice2txt.config import Settings
from voice2txt.models import ModelCatalog, install_model, is_valid_vosk_model
from voice2txt.recorder import Recorder
from voice2txt.storage import save_transcript
from voice2txt.transcriber import Transcriber

__all__ = [
    "Settings",
    "Recorder",
    "Transcriber",
    "save_transcript",
    "ModelCatalog",
    "install_model",
    "is_valid_vosk_model",
]
__version__ = "0.2.0"
