"""Bootstrap helper for chat module."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig, load_config
from .database import Database
from .engine import ChatEngine
from .logging_setup import setup_logging


@dataclass
class ChatContext:
    config: AppConfig
    database: Database
    engine: ChatEngine


def build_context(base_dir=None) -> ChatContext:
    setup_logging()
    config = load_config(base_dir)
    database = Database(config.database_url)
    database.create_all()
    engine = ChatEngine(config, database)
    return ChatContext(config=config, database=database, engine=engine)
