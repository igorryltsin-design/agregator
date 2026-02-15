"""Aggregator SPPR module package."""

from .app import create_app
from .bootstrap import bootstrap_pipeline

__all__ = ["create_app", "bootstrap_pipeline"]
