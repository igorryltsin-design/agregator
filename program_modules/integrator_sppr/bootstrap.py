"""Bootstrap context for Integrator SPPR."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig, load_config
from .database import Database
from .engine import IntegratorEngine
from .logging_setup import setup_logging


@dataclass
class IntegratorContext:
    config: AppConfig
    database: Database
    engine: IntegratorEngine


def build_context(base_dir=None) -> IntegratorContext:
    setup_logging()
    config = load_config(base_dir)
    database = Database(config.database_url)
    database.create_all()
    engine = IntegratorEngine(config, database)
    return IntegratorContext(config=config, database=database, engine=engine)
