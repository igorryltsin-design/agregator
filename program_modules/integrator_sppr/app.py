"""Flask application entry point for Integrator SPPR."""

from __future__ import annotations

import atexit

from flask import Flask

from .api.routes import create_blueprint
from .bootstrap import build_context


def create_app() -> Flask:
    ctx = build_context()
    app = Flask(__name__)
    app.config["INTEGRATOR_CONFIG"] = ctx.config
    app.register_blueprint(create_blueprint(ctx.engine), url_prefix="/integrator")
    atexit.register(ctx.engine.shutdown)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8251)
