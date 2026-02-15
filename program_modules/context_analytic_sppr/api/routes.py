"""Flask blueprint for Contextual Analytics module."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..engine import ContextEngine
from . import schemas


def create_blueprint(engine: ContextEngine) -> Blueprint:
    bp = Blueprint("context_api", __name__)

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify(schemas.success({"status": "ok"}).to_dict())

    @bp.route("/sources", methods=["POST"])
    def register_source():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.register_source(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/observations", methods=["POST"])
    def ingest_observation():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.ingest_observation(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/analyse", methods=["POST"])
    def analyse():
        limit = int(request.args.get("limit", 64))
        result = engine.analyse(limit=limit)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 500)

    @bp.route("/stats", methods=["GET"])
    def stats():
        result = engine.stats()
        return jsonify(schemas.success(result.payload).to_dict())

    return bp
