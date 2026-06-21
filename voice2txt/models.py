"""Vosk model catalog and installer."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from voice2txt.config import Settings, get_project_root

logger = logging.getLogger(__name__)

CATALOG_FILENAME = "models.catalog.json"
ProgressCallback = Callable[[int, str, dict[str, Any] | None], None]

_lock = threading.Lock()
_install_state: dict[str, Any] = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "error": None,
    "model_id": None,
    "phase": "",
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "speed_bps": 0.0,
}


@dataclass
class ModelEntry:
    id: str
    size: str
    description: str
    recommended: bool = False
    language: str = "fa"

    def download_url(self, url_template: str) -> str:
        return url_template.format(model_id=self.id)


@dataclass
class ModelCatalog:
    catalog_version: int
    source_page: str
    url_template: str
    languages: dict[str, Any]
    remote_catalog_hint: str = ""

    @classmethod
    def load(cls, path: Path | None = None, remote_url: str | None = None) -> ModelCatalog:
        if remote_url:
            try:
                return cls.from_dict(_fetch_json(remote_url))
            except Exception as exc:
                logger.warning("Remote catalog fetch failed (%s), using local", exc)

        catalog_path = path or (_package_data_dir() / CATALOG_FILENAME)
        with open(catalog_path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCatalog:
        return cls(
            catalog_version=int(data.get("catalog_version", 1)),
            source_page=data.get("source_page", "https://alphacephei.com/vosk/models"),
            url_template=data.get(
                "url_template",
                "https://alphacephei.com/vosk/models/{model_id}.zip",
            ),
            languages=data.get("languages", {}),
            remote_catalog_hint=data.get("remote_catalog_hint", ""),
        )

    def list_models(self, language: str = "fa") -> list[ModelEntry]:
        lang = self.languages.get(language, {})
        models = lang.get("models", [])
        return [
            ModelEntry(
                id=m["id"],
                size=m.get("size", "?"),
                description=m.get("description", ""),
                recommended=bool(m.get("recommended")),
                language=language,
            )
            for m in models
        ]

    def default_model_id(self, language: str = "fa") -> str | None:
        lang = self.languages.get(language, {})
        if default := lang.get("default_model"):
            return default
        models = self.list_models(language)
        for m in models:
            if m.recommended:
                return m.id
        return models[0].id if models else None

    def get_model(self, model_id: str, language: str = "fa") -> ModelEntry | None:
        for entry in self.list_models(language):
            if entry.id == model_id:
                return entry
        return None

    def to_api_dict(self, language: str = "fa") -> dict[str, Any]:
        lang = self.languages.get(language, {})
        return {
            "catalog_version": self.catalog_version,
            "source_page": self.source_page,
            "url_template": self.url_template,
            "default_model": self.default_model_id(language),
            "language_label": lang.get("label", language),
            "models": [
                {
                    "id": m.id,
                    "size": m.size,
                    "description": m.description,
                    "recommended": m.recommended,
                    "download_url": m.download_url(self.url_template),
                }
                for m in self.list_models(language)
            ],
        }


def _package_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _fetch_json(url: str) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "Voice2TXT-Persian/0.2"})
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def is_valid_vosk_model(path: Path) -> bool:
    """Check minimal Vosk model layout."""
    if not path.is_dir():
        return False
    return (path / "am" / "final.mdl").is_file() or (path / "graph").is_dir()


def get_install_state() -> dict[str, Any]:
    with _lock:
        return dict(_install_state)


def _set_install_state(**kwargs: Any) -> None:
    with _lock:
        _install_state.update(kwargs)


def install_model(
    model_id: str,
    settings: Settings,
    *,
    language: str = "fa",
    catalog: ModelCatalog | None = None,
    progress_callback: ProgressCallback | None = None,
    models_dir: Path | None = None,
) -> Path:
    """
    Download and extract a Vosk model zip into models_dir.
    Updates settings.model_path and saves settings.
    Returns path to extracted model directory.
    """
    catalog = catalog or ModelCatalog.load(
        remote_url=settings.models_catalog_url or None,
    )
    entry = catalog.get_model(model_id, language)
    if entry is None:
        raise ValueError(f"Unknown model: {model_id}")

    root = models_dir or get_project_root()
    target_dir = root / model_id
    download_url = entry.download_url(catalog.url_template)
    cache_dir = root / ".model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{model_id}.zip"

    def report(progress: int, message: str, extra: dict[str, Any] | None = None) -> None:
        if progress_callback:
            progress_callback(progress, message, extra)
        logger.info("[%d%%] %s", progress, message)

    if is_valid_vosk_model(target_dir):
        report(100, "Model already installed", {"phase": "done"})
        settings.model_path = model_id
        settings.save()
        return target_dir

    report(5, f"Downloading {model_id}...", {"phase": "preparing"})

    def on_download(downloaded: int, total: int, speed_bps: float) -> None:
        if total > 0:
            fraction = downloaded / total
            progress = 5 + int(fraction * 55)
        else:
            progress = 5 + min(55, downloaded // (1024 * 1024))
        report(
            progress,
            "Downloading...",
            {
                "phase": "downloading",
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "speed_bps": round(speed_bps, 1),
            },
        )

    _download_file(download_url, zip_path, on_download)

    report(65, "Extracting...", {"phase": "extracting", "downloaded_bytes": 0, "total_bytes": 0, "speed_bps": 0})
    _extract_zip(zip_path, root, model_id)

    if not is_valid_vosk_model(target_dir):
        raise RuntimeError(f"Extracted files do not look like a valid Vosk model: {target_dir}")

    report(95, "Updating settings...", {"phase": "configuring", "speed_bps": 0})
    settings.model_path = model_id
    settings.save()

    try:
        zip_path.unlink(missing_ok=True)
    except OSError:
        pass

    report(100, "Installation complete", {"phase": "done", "speed_bps": 0})
    return target_dir


def install_model_async(
    model_id: str,
    settings: Settings,
    *,
    language: str = "fa",
    on_complete: Callable[[Path | None, Exception | None], None] | None = None,
) -> None:
    """Run install_model in a background thread (for web UI)."""

    def worker() -> None:
        try:
            _set_install_state(
                status="running",
                progress=0,
                message="Starting...",
                error=None,
                model_id=model_id,
                phase="preparing",
                downloaded_bytes=0,
                total_bytes=0,
                speed_bps=0.0,
            )

            def progress(p: int, msg: str, extra: dict[str, Any] | None = None) -> None:
                payload: dict[str, Any] = {
                    "status": "running",
                    "progress": p,
                    "message": msg,
                    "error": None,
                    "model_id": model_id,
                }
                if extra:
                    payload.update(extra)
                _set_install_state(**payload)

            path = install_model(model_id, settings, language=language, progress_callback=progress)
            _set_install_state(
                status="done",
                progress=100,
                message="Done",
                error=None,
                model_id=model_id,
                phase="done",
                speed_bps=0.0,
            )
            if on_complete:
                on_complete(path, None)
        except Exception as exc:
            logger.exception("Model install failed")
            _set_install_state(status="error", progress=0, message=str(exc), error=str(exc), model_id=model_id)
            if on_complete:
                on_complete(None, exc)

    current = get_install_state()
    if current.get("status") == "running":
        raise RuntimeError("Installation already in progress")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def _download_file(
    url: str,
    dest: Path,
    progress_fn: Callable[[int, int, float], None] | None = None,
) -> None:
    req = Request(url, headers={"User-Agent": "Voice2TXT-Persian/0.2"})
    try:
        with urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 256 * 1024
            last_time = time.monotonic()
            last_downloaded = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    elapsed = now - last_time
                    if elapsed >= 0.4:
                        speed = (downloaded - last_downloaded) / elapsed if elapsed > 0 else 0.0
                        last_time = now
                        last_downloaded = downloaded
                        if progress_fn:
                            progress_fn(downloaded, total, speed)
                    elif progress_fn and total > 0 and downloaded == total:
                        progress_fn(downloaded, total, 0.0)
            if progress_fn:
                progress_fn(downloaded, total, 0.0)
    except URLError as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


def _extract_zip(zip_path: Path, dest_root: Path, model_id: str) -> None:
    target = dest_root / model_id
    if target.exists():
        shutil.rmtree(target)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if not members:
            raise RuntimeError("Empty zip archive")

        top_levels = {m.split("/")[0].split("\\")[0] for m in members if m.strip()}
        zf.extractall(dest_root)

        if target.is_dir() and is_valid_vosk_model(target):
            return

        if len(top_levels) == 1:
            extracted = dest_root / next(iter(top_levels))
            if extracted.is_dir() and extracted.name != model_id:
                if target.exists():
                    shutil.rmtree(target)
                extracted.rename(target)

    if not is_valid_vosk_model(target):
        raise RuntimeError(f"Could not locate valid model folder after extract (expected {target})")
