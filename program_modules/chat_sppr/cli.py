"""CLI for the chat module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import build_context


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Chat SPPR CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    user = sub.add_parser("ensure-user", help="Create or fetch user")
    user.add_argument("--username", required=True)
    user.add_argument("--display-name")
    user.add_argument("--role", default="operator")
    user.add_argument("--base-dir")
    user.set_defaults(cmd="ensure-user")

    start = sub.add_parser("start-session", help="Start chat session")
    start.add_argument("--user-id", type=int, required=True)
    start.add_argument("--title", required=True)
    start.add_argument("--base-dir")
    start.set_defaults(cmd="start-session")

    ask = sub.add_parser("ask", help="Send question")
    ask.add_argument("--session-id", type=int, required=True)
    ask.add_argument("--user-message", required=True)
    ask.add_argument("--base-dir")
    ask.set_defaults(cmd="ask")

    history = sub.add_parser("history", help="Show history")
    history.add_argument("--session-id", type=int, required=True)
    history.add_argument("--limit", type=int, default=20)
    history.add_argument("--base-dir")
    history.set_defaults(cmd="history")

    close = sub.add_parser("close-session", help="Close session")
    close.add_argument("--session-id", type=int, required=True)
    close.add_argument("--base-dir")
    close.set_defaults(cmd="close-session")

    report = sub.add_parser("report", help="Show telemetry report")
    report.add_argument("--base-dir")
    report.set_defaults(cmd="report")

    watchdog = sub.add_parser("watchdog", help="Run diagnostics")
    watchdog.add_argument("--base-dir")
    watchdog.set_defaults(cmd="watchdog")

    serve = sub.add_parser("runserver", help="Start HTTP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8451)
    serve.add_argument("--base-dir")
    serve.set_defaults(cmd="runserver")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    ctx = build_context(Path(args.base_dir) if getattr(args, "base_dir", None) else None)

    if args.command == "ensure-user":
        payload = {
            "username": args.username,
            "display_name": args.display_name or args.username,
            "role": args.role,
        }
        result = ctx.engine.ensure_user(payload)
        _print(result.payload)
    elif args.command == "start-session":
        result = ctx.engine.start_session({"user_id": args.user_id, "title": args.title})
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "ask":
        result = ctx.engine.ask({"session_id": args.session_id, "user_message": args.user_message})
        _print(result.payload if result.ok else {"error": result.error})
    elif args.command == "history":
        result = ctx.engine.history({"session_id": args.session_id, "limit": args.limit})
        _print(result.payload)
    elif args.command == "close-session":
        result = ctx.engine.close_session({"session_id": args.session_id})
        _print(result.payload)
    elif args.command == "report":
        result = ctx.engine.report()
        _print(result.payload)
    elif args.command == "watchdog":
        result = ctx.engine.watchdog()
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
