"""Flask app entry point for Contextual Analytics SPPR."""

from __future__ import annotations

from flask import Flask

from .api.routes import create_blueprint
from .bootstrap import build_context


def create_app() -> Flask:
    ctx = build_context()
    app = Flask(__name__)
    app.config["CONTEXT_CONFIG"] = ctx.config
    app.register_blueprint(create_blueprint(ctx.engine), url_prefix="/context")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8151)
