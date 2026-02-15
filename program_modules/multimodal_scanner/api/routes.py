"""Flask blueprint for the scanner module."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..engine import ScannerEngine
from . import schemas


def create_blueprint(engine: ScannerEngine) -> Blueprint:
    bp = Blueprint("scanner_api", __name__)

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify(schemas.success({"status": "ok"}).to_dict())

    @bp.route("/scan", methods=["POST"])
    def trigger_scan():
        memo = (request.get_json(silent=True) or {}).get("memo")
        result = engine.scan(memo=memo)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 500)

    @bp.route("/artifacts/latest", methods=["GET"])
    def latest():
        limit = int(request.args.get("limit", 20))
        result = engine.latest_artifacts(limit=limit)
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/artifacts/<int:artifact_id>/classify", methods=["POST"])
    def classify(artifact_id: int):
        result = engine.classify(artifact_id)
        envelope = schemas.success(result.payload) if result.ok else schemas.failure(result.error or "error")
        return jsonify(envelope.to_dict()), (200 if result.ok else 404)

    @bp.route("/graph", methods=["POST"])
    def build_graph():
        payload = request.get_json(force=True, silent=True) or {}
        artifact_ids = payload.get("artifact_ids", [])
        result = engine.build_graph(artifact_ids)
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/stats", methods=["GET"])
    def stats():
        result = engine.stats_snapshot()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/logs", methods=["GET"])
    def logs():
        result = engine.logs(limit=int(request.args.get("limit", 50)))
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/report", methods=["GET"])
    def report():
        result = engine.report()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/search", methods=["GET"])
    def search():
        query = request.args.get("q", "")
        result = engine.search(query)
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/watchdog", methods=["GET"])
    def watchdog():
        result = engine.watchdog()
        return jsonify(schemas.success(result.payload).to_dict())

    return bp
