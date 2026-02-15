"""CLI for Contextual Analytics SPPR."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .bootstrap import build_context


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def with_context(func):
    def wrapper(args):
        ctx = build_context(Path(args.base_dir) if getattr(args, "base_dir", None) else None)
        try:
            return func(ctx, args)
        finally:
            pass

    return wrapper


@with_context
def cmd_register(ctx, args):
    payload = {
        "name": args.name,
        "modality": args.modality,
        "description": args.description,
        "relevance_bias": args.relevance,
        "latency_expectation": args.latency,
    }
    result = ctx.engine.register_source(payload)
    _print(result.payload if result.ok else {"error": result.error})


@with_context
def cmd_ingest(ctx, args):
    payload = {
        "source": args.source,
        "modality": args.modality,
        "content_ref": args.content,
        "annotation": args.annotation,
        "recorded_at": (datetime.utcnow()).isoformat(),
        "intensity": args.intensity,
        "confidence": args.confidence,
    }
    result = ctx.engine.ingest_observation(payload)
    _print(result.payload if result.ok else {"error": result.error})


@with_context
def cmd_analyse(ctx, args):
    result = ctx.engine.analyse(limit=args.limit)
    _print(result.payload if result.ok else {"error": result.error})


@with_context
def cmd_stats(ctx, args):
    result = ctx.engine.stats()
    _print(result.payload)


@with_context
def cmd_runserver(ctx, args):
    from .app import create_app

    app = create_app()
    app.run(host=args.host, port=args.port)


def build_parser():
    parser = argparse.ArgumentParser(description="Contextual Analytics CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    register = sub.add_parser("register-source", help="Register new source channel")
    register.add_argument("--name", required=True)
    register.add_argument("--modality", required=True)
    register.add_argument("--description")
    register.add_argument("--relevance", type=float, default=0.5)
    register.add_argument("--latency", type=int, default=5)
    register.add_argument("--base-dir")
    register.set_defaults(func=cmd_register)

    ingest = sub.add_parser("ingest", help="Push observation")
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--modality", required=True)
    ingest.add_argument("--content", required=True)
    ingest.add_argument("--annotation")
    ingest.add_argument("--intensity", type=float, default=0.6)
    ingest.add_argument("--confidence", type=float, default=0.6)
    ingest.add_argument("--base-dir")
    ingest.set_defaults(func=cmd_ingest)

    analyse = sub.add_parser("analyse", help="Run analytical window")
    analyse.add_argument("--limit", type=int, default=64)
    analyse.add_argument("--base-dir")
    analyse.set_defaults(func=cmd_analyse)

    stats = sub.add_parser("stats", help="Show operational stats")
    stats.add_argument("--base-dir")
    stats.set_defaults(func=cmd_stats)

    runserver = sub.add_parser("runserver", help="Start HTTP server")
    runserver.add_argument("--host", default="0.0.0.0")
    runserver.add_argument("--port", type=int, default=8151)
    runserver.set_defaults(func=cmd_runserver)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
