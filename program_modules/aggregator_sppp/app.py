"""Flask application entry point for Aggregator SPPR."""

from __future__ import annotations

import atexit

from flask import Flask

from .api.routes import create_blueprint
from .bootstrap import bootstrap_pipeline


def create_app() -> Flask:
    ctx = bootstrap_pipeline()
    app = Flask(__name__)
    app.config["AGGREGATOR_CONFIG"] = ctx.config
    app.register_blueprint(create_blueprint(ctx.pipeline, ctx.telemetry), url_prefix="/api")
    atexit.register(ctx.scheduler.stop)
    atexit.register(ctx.telemetry.stop)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8051)
