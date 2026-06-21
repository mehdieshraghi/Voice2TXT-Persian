"""HTTP routes for the Voice2TXT web UI."""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from voice2txt.config import Settings
from voice2txt.models import ModelCatalog, get_install_state, install_model_async, is_valid_vosk_model
from voice2txt.recorder import Recorder
from voice2txt.storage import save_transcript
from voice2txt.transcriber import Transcriber

logger = logging.getLogger(__name__)


def _get_transcriber(app: Flask) -> Transcriber:
    if "TRANSCRIBER" not in app.config:
        app.config["TRANSCRIBER"] = Transcriber(app.config["SETTINGS"])
    return app.config["TRANSCRIBER"]


def register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        settings: Settings = app.config["SETTINGS"]
        transcriber = _get_transcriber(app)
        return render_template(
            "index.html",
            settings=settings.to_dict(),
            model_ready=transcriber.is_model_ready(),
            model_path_resolved=str(settings.resolved_model_path()),
        )

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        settings: Settings = app.config["SETTINGS"]
        transcriber = _get_transcriber(app)
        return jsonify(
            {
                "settings": settings.to_dict(),
                "model_ready": transcriber.is_model_ready(),
                "model_path_resolved": str(settings.resolved_model_path()),
            }
        )

    @app.route("/api/settings", methods=["POST"])
    def update_settings():
        settings: Settings = app.config["SETTINGS"]
        data = request.get_json(silent=True) or {}
        settings.update(data)
        errors = settings.validate_settings()
        if errors:
            return jsonify({"ok": False, "errors": errors}), 400

        settings.save()
        transcriber: Transcriber = app.config.get("TRANSCRIBER")
        if transcriber:
            transcriber.settings = settings
            transcriber.reload_model()

        return jsonify({"ok": True, "settings": settings.to_dict()})

    @app.route("/api/devices", methods=["GET"])
    def list_devices():
        try:
            recorder = Recorder(app.config["SETTINGS"])
            return jsonify({"ok": True, "devices": recorder.list_devices()})
        except Exception as exc:
            logger.exception("Failed to list audio devices")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/transcribe/file", methods=["POST"])
    def transcribe_file():
        settings: Settings = app.config["SETTINGS"]
        transcriber = _get_transcriber(app)

        if "audio" not in request.files:
            return jsonify({"ok": False, "error": "No audio file uploaded"}), 400

        file = request.files["audio"]
        if not file.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        data = file.read()
        if len(data) > settings.max_upload_bytes:
            return jsonify({"ok": False, "error": f"File exceeds {settings.max_upload_mb} MB limit"}), 400

        try:
            text = transcriber.transcribe_wav(data)
            return jsonify({"ok": True, "text": text})
        except Exception as exc:
            logger.exception("Transcription failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/transcribe/record", methods=["POST"])
    def transcribe_server_record():
        """Record from server microphone (CLI-like, runs on machine hosting Flask)."""
        settings: Settings = app.config["SETTINGS"]
        transcriber = _get_transcriber(app)
        payload = request.get_json(silent=True) or {}

        duration = float(payload.get("duration", settings.record_duration))
        device = payload.get("device")
        device = int(device) if device is not None else None

        try:
            recorder = Recorder(settings)
            wav_bytes = recorder.record(duration=duration, device=device)
            text = transcriber.transcribe_wav(wav_bytes)
            return jsonify({"ok": True, "text": text, "duration": duration})
        except Exception as exc:
            logger.exception("Server-side recording failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/models/catalog", methods=["GET"])
    def models_catalog():
        settings: Settings = app.config["SETTINGS"]
        language = request.args.get("lang", "fa")
        try:
            catalog = ModelCatalog.load(remote_url=settings.models_catalog_url or None)
            return jsonify({"ok": True, **catalog.to_api_dict(language)})
        except Exception as exc:
            logger.exception("Failed to load model catalog")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/models/status", methods=["GET"])
    def models_status():
        settings: Settings = app.config["SETTINGS"]
        model_path = settings.resolved_model_path()
        return jsonify(
            {
                "ok": True,
                "model_ready": is_valid_vosk_model(model_path),
                "model_path": str(model_path),
                "install": get_install_state(),
            }
        )

    @app.route("/api/models/install", methods=["POST"])
    def models_install():
        settings: Settings = app.config["SETTINGS"]
        payload = request.get_json(silent=True) or {}
        model_id = payload.get("model_id")

        catalog = ModelCatalog.load(remote_url=settings.models_catalog_url or None)
        if not model_id:
            model_id = catalog.default_model_id("fa")
        if not model_id or catalog.get_model(model_id) is None:
            return jsonify({"ok": False, "error": "Invalid model_id"}), 400

        state = get_install_state()
        if state.get("status") == "running":
            return jsonify({"ok": False, "error": "Installation already in progress"}), 409

        def on_complete(path, error) -> None:
            if path and not error:
                transcriber: Transcriber | None = app.config.get("TRANSCRIBER")
                if transcriber:
                    transcriber.settings = settings
                    transcriber.reload_model()

        try:
            install_model_async(model_id, settings, on_complete=on_complete)
            return jsonify({"ok": True, "model_id": model_id, "message": "Installation started"})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/save", methods=["POST"])
    def save_text():
        settings: Settings = app.config["SETTINGS"]
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "").strip()
        if not text:
            return jsonify({"ok": False, "error": "No text to save"}), 400

        try:
            path = save_transcript(text, settings)
            return jsonify({"ok": True, "path": str(path)})
        except Exception as exc:
            logger.exception("Save failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
