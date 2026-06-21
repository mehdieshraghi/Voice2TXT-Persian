"""Flask web application factory."""

from __future__ import annotations

from flask import Flask

from voice2txt.config import Settings, get_project_root


def create_app(settings: Settings | None = None, settings_path: str | None = None) -> Flask:
    """
    Application factory — use this to embed the UI in your own Flask app
    or run standalone via run_web.py.

    Example:
        from app import create_app
        app = create_app()
        app.run()
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    if settings is None:
        root = get_project_root()
        default_path = root / "settings.json"
        example_path = root / "settings.example.json"
        load_path = default_path if default_path.exists() else example_path
        settings = Settings.load(load_path if load_path.exists() else None)

    app.config["SETTINGS"] = settings
    app.config["PROJECT_ROOT"] = get_project_root()

    from app.routes import register_routes

    register_routes(app)
    return app
