"""Database helpers for the Aggregator SPPR program."""

from __future__ import annotations

import contextlib
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Database:
    """Database wrapper that hides SQLAlchemy boilerplate."""

    def __init__(self, url: str, echo: bool = False) -> None:
        self._engine: Engine = create_engine(url, echo=echo, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    @property
    def engine(self) -> Engine:
        return self._engine

    def create_all(self) -> None:
        from . import models  # noqa: F401

        Base.metadata.create_all(self._engine)

    @contextlib.contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
