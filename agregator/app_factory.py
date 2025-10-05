"""Flask application factory for Agregator."""

from __future__ import annotations

from flask import Flask

from agregator.config import AppConfig, load_app_config


def create_app(config: AppConfig | None = None) -> Flask:
    """Instantiate and configure the Flask application."""
    cfg = config or load_app_config()
    from app import app, setup_app  # import late to avoid circular imports

    setup_app(cfg)
    return app
