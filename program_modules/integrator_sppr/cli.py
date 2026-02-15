"""CLI for Integrator SPPR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import build_context


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Integrator SPPR CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register-node", help="Register module node")
    register.add_argument("--name", required=True)
    register.add_argument("--kind", required=True, choices=["ingestion", "analytics", "dialog", "storage", "external"])
    register.add_argument("--api-url", required=True)
    register.add_argument("--priority", type=int)
    register.add_argument("--base-dir")
    register.set_defaults(cmd="register-node")

    connect = sub.add_parser("connect", help="Connect nodes")
    connect.add_argument("--from", dest="from_node", required=True)
    connect.add_argument("--to", dest="to_node", required=True)
    connect.add_argument("--max-bandwidth", type=float, default=100.0)
    connect.add_argument("--base-dir")
    connect.set_defaults(cmd="connect")

    channel = sub.add_parser("add-channel", help="Attach channel to edge")
    channel.add_argument("--edge-id", type=int, required=True)
    channel.add_argument("--label", required=True)
    channel.add_argument("--modality", required=True)
    channel.add_argument("--priority", type=int)
    channel.add_argument("--target-rate", type=float)
    channel.add_argument("--base-dir")
    channel.set_defaults(cmd="add-channel")

    metrics = sub.add_parser("channel-metrics", help="Update channel stats")
    metrics.add_argument("--channel-id", type=int, required=True)
    metrics.add_argument("--actual-rate", type=float, required=True)
    metrics.add_argument("--dropped", type=int, required=True)
    metrics.add_argument("--retries", type=int, required=True)
    metrics.add_argument("--base-dir")
    metrics.set_defaults(cmd="channel-metrics")

    snapshot = sub.add_parser("snapshot", help="Generate telemetry snapshot")
    snapshot.add_argument("--base-dir")
    snapshot.set_defaults(cmd="snapshot")

    audit = sub.add_parser("audit", help="Show audit log")
    audit.add_argument("--base-dir")
    audit.set_defaults(cmd="audit")

    reports = sub.add_parser("reports", help="List stored reports")
    reports.add_argument("--base-dir")
    reports.set_defaults(cmd="reports")

    serve = sub.add_parser("runserver", help="Run HTTP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8251)
    serve.add_argument("--base-dir")
    serve.set_defaults(cmd="runserver")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    ctx = build_context(Path(args.base_dir) if getattr(args, "base_dir", None) else None)

    if args.command == "register-node":
        payload = {
            "name": args.name,
            "kind": args.kind,
            "api_url": args.api_url,
            "priority": args.priority,
        }
        result = ctx.engine.register_node(payload)
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "connect":
        payload = {
            "from": args.from_node,
            "to": args.to_node,
            "max_bandwidth": args.max_bandwidth,
        }
        result = ctx.engine.connect_nodes(payload)
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "add-channel":
        payload = {
            "edge_id": args.edge_id,
            "label": args.label,
            "modality": args.modality,
            "priority": args.priority,
            "target_rate": args.target_rate,
        }
        result = ctx.engine.add_channel(payload)
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "channel-metrics":
        payload = {
            "channel_id": args.channel_id,
            "actual_rate": args.actual_rate,
            "dropped": args.dropped,
            "retries": args.retries,
        }
        result = ctx.engine.channel_metrics(payload)
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "snapshot":
        result = ctx.engine.snapshot()
        _print(result.payload)
    elif args.command == "audit":
        result = ctx.engine.audit_log()
        _print(result.payload)
    elif args.command == "reports":
        result = ctx.engine.reports()
        _print(result.payload)
    elif args.command == "runserver":
        from .app import create_app

        app = create_app()
        app.run(host=args.host, port=args.port)
    ctx.engine.shutdown()


if __name__ == "__main__":
    main()
