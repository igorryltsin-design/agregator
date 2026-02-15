"""Flask blueprint for Integrator SPPR."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..engine import IntegratorEngine
from . import schemas


def create_blueprint(engine: IntegratorEngine) -> Blueprint:
    bp = Blueprint("integrator_api", __name__)

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify(schemas.success({"status": "ok"}).to_dict())

    @bp.route("/nodes", methods=["POST"])
    def register_node():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.register_node(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/edges", methods=["POST"])
    def connect_nodes():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.connect_nodes(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/heartbeat/<name>", methods=["POST"])
    def heartbeat(name: str):
        result = engine.heartbeat(name)
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/channels", methods=["POST"])
    def add_channel():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.add_channel(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/channels/metrics", methods=["POST"])
    def channel_metrics():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.channel_metrics(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/snapshot", methods=["POST"])
    def snapshot():
        result = engine.snapshot()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/audit", methods=["GET"])
    def audit():
        result = engine.audit_log()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/reports", methods=["GET"])
    def reports():
        result = engine.reports()
        return jsonify(schemas.success(result.payload).to_dict())

    return bp
