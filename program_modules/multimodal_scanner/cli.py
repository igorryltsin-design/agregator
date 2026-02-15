"""CLI for the multimodal scanner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import build_context


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Multimodal Scanner CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Run filesystem scan")
    scan.add_argument("--memo")
    scan.add_argument("--base-dir")
    scan.set_defaults(cmd="scan")

    classify = sub.add_parser("classify", help="Classify artifact")
    classify.add_argument("--artifact-id", type=int, required=True)
    classify.add_argument("--base-dir")
    classify.set_defaults(cmd="classify")

    graph = sub.add_parser("graph", help="Build similarity graph")
    graph.add_argument("--artifact-ids", nargs="+", type=int, required=True)
    graph.add_argument("--base-dir")
    graph.set_defaults(cmd="graph")

    stats = sub.add_parser("stats", help="Show modality stats")
    stats.add_argument("--base-dir")
    stats.set_defaults(cmd="stats")

    logs = sub.add_parser("logs", help="Show recent logs")
    logs.add_argument("--base-dir")
    logs.add_argument("--limit", type=int, default=20)
    logs.set_defaults(cmd="logs")

    watchdog = sub.add_parser("watchdog", help="Run anomaly checks")
    watchdog.add_argument("--base-dir")
    watchdog.set_defaults(cmd="watchdog")

    report = sub.add_parser("report", help="Generate text report")
    report.add_argument("--base-dir")
    report.set_defaults(cmd="report")

    search = sub.add_parser("search", help="Search artifacts by keyword")
    search.add_argument("--query", required=True)
    search.add_argument("--base-dir")
    search.set_defaults(cmd="search")

    serve = sub.add_parser("runserver", help="Start HTTP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8351)
    serve.add_argument("--base-dir")
    serve.set_defaults(cmd="runserver")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    ctx = build_context(Path(args.base_dir) if getattr(args, "base_dir", None) else None)

    if args.command == "scan":
        result = ctx.engine.scan(memo=args.memo)
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "classify":
        result = ctx.engine.classify(args.artifact_id)
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "graph":
        result = ctx.engine.build_graph(args.artifact_ids)
        _print(result.payload)
    elif args.command == "stats":
        result = ctx.engine.stats_snapshot()
        _print(result.payload)
    elif args.command == "logs":
        result = ctx.engine.logs(limit=args.limit)
        _print(result.payload)
    elif args.command == "watchdog":
        result = ctx.engine.watchdog()
        _print(result.payload)
    elif args.command == "report":
        result = ctx.engine.report()
        _print(result.payload)
    elif args.command == "search":
        result = ctx.engine.search(args.query)
        _print(result.payload)
    elif args.command == "runserver":
        from .app import create_app

        app = create_app()
        app.run(host=args.host, port=args.port)
        ctx.engine.shutdown()
        return

    ctx.engine.shutdown()


if __name__ == "__main__":
    main()
