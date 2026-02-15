"""Flask blueprint exposing Aggregator SPPR API."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.pipeline import AggregationPipeline
from . import schemas


def create_blueprint(pipeline: AggregationPipeline, telemetry) -> Blueprint:
    bp = Blueprint("aggregator_api", __name__)

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify(schemas.success({"status": "ok"}).to_dict())

    @bp.route("/sources", methods=["POST"])
    def create_source():
        payload = request.get_json(force=True, silent=True) or {}
        result = pipeline.register_source(payload)
        envelope = (
            schemas.success(result.payload)
            if result.ok
            else schemas.failure(result.error or "error")
        )
        return jsonify(envelope.to_dict()), (200 if result.ok else 400)

    @bp.route("/sources", methods=["GET"])
    def list_sources():
        result = pipeline.list_sources()
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/signals", methods=["POST"])
    def ingest_signal():
        payload = request.get_json(force=True, silent=True) or {}
        result = pipeline.ingest_signal(payload)
        envelope = (
            schemas.success(result.payload)
            if result.ok
            else schemas.failure(result.error or "error")
        )
        status = 200 if result.ok else 400
        return jsonify(envelope.to_dict()), status

    @bp.route("/signals/latest", methods=["GET"])
    def latest_signals():
        limit = int(request.args.get("limit", 50))
        result = pipeline.latest_signals(limit=limit)
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/fusion", methods=["POST"])
    def fuse_window():
        payload = request.get_json(force=True, silent=True) or {}
        result = pipeline.build_fusion(payload)
        envelope = (
            schemas.success(result.payload)
            if result.ok
            else schemas.failure(result.error or "error")
        )
        status = 200 if result.ok else 400
        return jsonify(envelope.to_dict()), status

    @bp.route("/fusion/latest", methods=["GET"])
    def latest_samples():
        result = pipeline.latest_samples(limit=int(request.args.get("limit", 20)))
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/history", methods=["GET"])
    def history_window():
        cursor = request.args.get("cursor", "api")
        result = pipeline.history_window(cursor, limit=int(request.args.get("limit", 50)))
        return jsonify(schemas.success(result.payload).to_dict())

    @bp.route("/stats", methods=["GET"])
    def stats():
        result = pipeline.stats()
        snapshot = telemetry.snapshot()
        payload = result.payload | {"telemetry": snapshot}
        return jsonify(schemas.success(payload).to_dict())

    return bp
