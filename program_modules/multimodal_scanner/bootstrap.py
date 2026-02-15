"""Bootstrapper for the scanner module."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig, load_config
from .database import Database
from .engine import ScannerEngine
from .logging_setup import setup_logging


@dataclass
class ScannerContext:
    config: AppConfig
    database: Database
    engine: ScannerEngine


def build_context(base_dir=None) -> ScannerContext:
    setup_logging()
    config = load_config(base_dir)
    database = Database(config.storage.database_url)
    database.create_all()
    engine = ScannerEngine(config, database)
    return ScannerContext(config=config, database=database, engine=engine)
