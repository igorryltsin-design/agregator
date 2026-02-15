"""Flask blueprint exposing chat API."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..engine import ChatEngine
from . import schemas


def create_blueprint(engine: ChatEngine) -> Blueprint:
    bp = Blueprint("chat_api", __name__)

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify(schemas.success({"status": "ok"}).to_dict())

    @bp.route("/users", methods=["POST"])
    def ensure_user():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.ensure_user(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/sessions", methods=["POST"])
    def start_session():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.start_session(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/sessions/<int:session_id>/ask", methods=["POST"])
    def ask(session_id: int):
        payload = request.get_json(force=True, silent=True) or {}
        payload["session_id"] = session_id
        result = engine.ask(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/sessions/<int:session_id>/history", methods=["GET"])
    def history(session_id: int):
        result = engine.history({"session_id": session_id, "limit": int(request.args.get("limit", 50))})
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/sessions/<int:session_id>/close", methods=["POST"])
    def close_session(session_id: int):
        result = engine.close_session({"session_id": session_id})
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/feedback", methods=["POST"])
    def feedback():
        payload = request.get_json(force=True, silent=True) or {}
        result = engine.feedback(payload)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/audit", methods=["GET"])
    def audit():
        result = engine.audit()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/telemetry", methods=["GET"])
    def telemetry():
        result = engine.telemetry_snapshot()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/report", methods=["GET"])
    def report():
        result = engine.report()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/watchdog", methods=["GET"])
    def watchdog():
        result = engine.watchdog()
        return jsonify(schemas.success(result.payload).to_dict())

    return bp
