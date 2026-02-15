"""Integrator SPPR standalone module."""

from .bootstrap import build_context
from .app import create_app

__all__ = ["build_context", "create_app"]
