#!/usr/bin/env python3
"""CLI entry point for Voice2TXT-Persian."""

from __future__ import annotations

import argparse
import logging
import sys

from voice2txt.config import Settings
from voice2txt.models import ModelCatalog, install_model, is_valid_vosk_model
from voice2txt.recorder import Recorder
from voice2txt.storage import save_transcript
from voice2txt.transcriber import Transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persian speech-to-text using Vosk (offline)",
    )
    parser.add_argument(
        "--settings",
        default="settings.json",
        help="Path to settings JSON (default: settings.json)",
    )
    parser.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        help="Transcribe an existing WAV file instead of recording",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Recording duration in seconds (overrides settings)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Save transcript to this path",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print only; do not save to output directory",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphones and exit",
    )
    parser.add_argument(
        "--device",
        type=int,
        metavar="ID",
        help="Input device index for recording",
    )
    parser.add_argument(
        "--install-model",
        metavar="MODEL_ID",
        nargs="?",
        const="__default__",
        help="Download and install a Vosk model (default: recommended Persian model)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Persian models from catalog",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-confirm model installation prompt",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def _prompt_install(settings: Settings, catalog: ModelCatalog) -> bool:
    default_id = catalog.default_model_id("fa")
    model_path = settings.resolved_model_path()
    print(f"\nModel not found: {model_path}")
    print(f"Recommended: {default_id}")
    print("Download from alphacephei.com/vosk/models (via project catalog)")
    answer = input("Install now? [y/N]: ").strip().lower()
    return answer in ("y", "yes", "بله")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    settings = Settings.load(args.settings)
    catalog = ModelCatalog.load(remote_url=settings.models_catalog_url or None)

    if args.list_models:
        for m in catalog.list_models("fa"):
            tag = " (recommended)" if m.recommended else ""
            print(f"  {m.id}  {m.size}{tag}  — {m.description}")
        return 0

    if args.install_model:
        model_id = catalog.default_model_id("fa") if args.install_model == "__default__" else args.install_model
        if not model_id or catalog.get_model(model_id) is None:
            print(f"Error: Unknown model '{model_id}'", file=sys.stderr)
            return 1
        try:
            path = install_model(
                model_id,
                settings,
                catalog=catalog,
                progress_callback=lambda p, msg, _extra=None: print(f"[{p:3d}%] {msg}"),
            )
            print(f"Installed: {path}")
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    settings_errors = settings.validate_settings()
    if settings_errors and not args.list_devices:
        for err in settings_errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    if args.list_devices:
        recorder = Recorder(settings)
        for dev in recorder.list_devices():
            print(f"[{dev['index']}] {dev['name']} ({dev['channels']} ch)")
        return 0

    if not is_valid_vosk_model(settings.resolved_model_path()):
        if args.yes or _prompt_install(settings, catalog):
            try:
                model_id = catalog.default_model_id("fa")
                install_model(
                    model_id,
                    settings,
                    catalog=catalog,
                    progress_callback=lambda p, msg, _extra=None: print(f"[{p:3d}%] {msg}"),
                )
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        else:
            print("Aborted. Use --install-model or download manually.", file=sys.stderr)
            return 1

    transcriber = Transcriber(settings)

    try:
        if args.file:
            print(f"Transcribing file: {args.file}")
            text = transcriber.transcribe_file(args.file)
        else:
            duration = args.duration or settings.record_duration
            print(f"Recording {duration}s from microphone...")
            recorder = Recorder(settings)
            wav_bytes = recorder.record(duration=duration, device=args.device)
            text = transcriber.transcribe_wav(wav_bytes)

        print("\n--- Transcription ---")
        print(text or "(no speech detected)")

        if not args.no_save and text:
            if args.output:
                from pathlib import Path

                Path(args.output).write_text(text, encoding="utf-8")
                print(f"\nSaved to: {args.output}")
            else:
                path = save_transcript(text, settings)
                print(f"\nSaved to: {path}")

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
