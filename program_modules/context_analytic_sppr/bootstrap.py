"""Bootstrap utilities for Contextual Analytics module."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig, load_config
from .database import Database
from .engine import ContextEngine
from .logging_setup import setup_logging


@dataclass
class ContextApp:
    config: AppConfig
    database: Database
    engine: ContextEngine


def build_context(base_dir=None) -> ContextApp:
    setup_logging()
    config = load_config(base_dir)
    database = Database(config.database_url)
    database.create_all()
    engine = ContextEngine(config, database)
    return ContextApp(config=config, database=database, engine=engine)
