#!/usr/bin/env python3
"""Utility script to prune expired AI search snippet cache entries."""

from app import app, setup_app, _prune_expired_snippet_cache


setup_app()


def prune_expired() -> int:
    with app.app_context():
        return _prune_expired_snippet_cache()


if __name__ == "__main__":
    removed = prune_expired()
    print(f"Removed {removed} expired snippet cache rows")
