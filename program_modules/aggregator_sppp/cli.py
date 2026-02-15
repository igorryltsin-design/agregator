"""Command line interface for Aggregator SPPR."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_pipeline


def _shutdown(ctx):
    ctx.scheduler.stop()
    ctx.telemetry.stop()


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_runserver(args):
    from .app import create_app

    app = create_app()
    app.run(host=args.host, port=args.port)


def cmd_register_source(args):
    ctx = bootstrap_pipeline(Path(args.base_dir) if args.base_dir else None)
    result = ctx.pipeline.register_source(
        {
            "name": args.name,
            "modality": args.modality,
            "reliability": args.reliability,
            "priority": args.priority,
            "description": args.description,
        }
    )
    _print(result.payload if result.ok else {"error": result.error})
    _shutdown(ctx)


def cmd_ingest_signal(args):
    ctx = bootstrap_pipeline(Path(args.base_dir) if args.base_dir else None)
    payload = {
        "source_name": args.source,
        "signal_type": args.signal_type,
        "payload_ref": args.payload_ref,
        "recorded_at": datetime.utcnow().isoformat(),
        "metadata": {},
        "intensity": args.intensity,
    }
    result = ctx.pipeline.ingest_signal(payload)
    _print(result.payload if result.ok else {"error": result.error})
    _shutdown(ctx)


def cmd_fuse(args):
    ctx = bootstrap_pipeline(Path(args.base_dir) if args.base_dir else None)
    payload = {
        "window_start": (datetime.utcnow() - timedelta(minutes=args.window)).isoformat(),
        "window_end": datetime.utcnow().isoformat(),
    }
    result = ctx.pipeline.build_fusion(payload)
    _print(result.payload if result.ok else {"error": result.error})
    _shutdown(ctx)


def cmd_stats(args):
    ctx = bootstrap_pipeline(Path(args.base_dir) if args.base_dir else None)
    result = ctx.pipeline.stats()
    _print(result.payload)
    _shutdown(ctx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregator SPPR CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    runserver = sub.add_parser("runserver", help="Start HTTP server")
    runserver.add_argument("--host", default="0.0.0.0")
    runserver.add_argument("--port", type=int, default=8051)
    runserver.set_defaults(func=cmd_runserver)

    reg = sub.add_parser("register-source", help="Create data source")
    reg.add_argument("--name", required=True)
    reg.add_argument("--modality", required=True)
    reg.add_argument("--reliability", type=float, default=0.5)
    reg.add_argument("--priority", type=int, default=5)
    reg.add_argument("--description")
    reg.add_argument("--base-dir")
    reg.set_defaults(func=cmd_register_source)

    ingest = sub.add_parser("ingest-signal", help="Ingest signal")
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--signal-type", required=True)
    ingest.add_argument("--payload-ref", required=True)
    ingest.add_argument("--intensity", type=float, default=0.6)
    ingest.add_argument("--base-dir")
    ingest.set_defaults(func=cmd_ingest_signal)

    fuse = sub.add_parser("fuse-window", help="Fuse recent window")
    fuse.add_argument("--window", type=int, default=5)
    fuse.add_argument("--base-dir")
    fuse.set_defaults(func=cmd_fuse)

    stats = sub.add_parser("stats", help="Show stats")
    stats.add_argument("--base-dir")
    stats.set_defaults(func=cmd_stats)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
