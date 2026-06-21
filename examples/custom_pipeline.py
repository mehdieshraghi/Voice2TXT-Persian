"""
Example: minimal custom pipeline without Flask.

Run: python examples/custom_pipeline.py path/to/audio.wav
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice2txt import Settings, Transcriber, save_transcript


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/custom_pipeline.py <audio.wav>")
        sys.exit(1)

    settings = Settings.load()
    transcriber = Transcriber(settings)

    audio_path = Path(sys.argv[1])
    text = transcriber.transcribe_file(audio_path)

    print(text)
    if text:
        path = save_transcript(text, settings)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
