"""Shared pytest fixtures for Agregator integration tests.

Provides a Flask test application, in-memory SQLite database, test client,
pre-created users, and mock LLM helpers so that test modules can focus on
behaviour rather than boilerplate.
"""

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Environment overrides (must be set BEFORE app import)
# ---------------------------------------------------------------------------

os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-do-not-use-in-production")
os.environ.setdefault("DEFAULT_ADMIN_USER", "admin")
os.environ.setdefault("DEFAULT_ADMIN_PASSWORD", "admin123")
os.environ.setdefault("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
os.environ.setdefault("LMSTUDIO_MODEL", "test-model")
os.environ.setdefault("SCAN_ROOT", tempfile.gettempdir())
os.environ.setdefault("EXTRACT_TEXT", "0")
os.environ.setdefault("TRANSCRIBE_ENABLED", "0")


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

_MOCK_LLM_RESPONSE: dict[str, Any] = {
    "choices": [
        {
            "message": {
                "content": "Это тестовый ответ от LLM.",
                "role": "assistant",
            }
        }
    ]
}


class MockLLMResponse:
    """Minimal mock that quacks like a ``requests.Response``."""

    status_code = 200
    ok = True

    def __init__(self, content: str = "Это тестовый ответ от LLM."):
        self._content = content
        self._json = {
            "choices": [
                {"message": {"content": content, "role": "assistant"}}
            ]
        }

    def json(self) -> dict:
        return self._json

    @property
    def text(self) -> str:
        return json.dumps(self._json)


def mock_llm_post(*args, **kwargs) -> MockLLMResponse:
    """Drop-in replacement for ``requests.Session.post`` during tests."""
    return MockLLMResponse()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _tmp_dir() -> Generator[Path, None, None]:
    """Session-scoped temporary directory for test artefacts."""
    with tempfile.TemporaryDirectory(prefix="agregator_test_") as d:
        yield Path(d)


@pytest.fixture(scope="session")
def app(_tmp_dir: Path):
    """Create a Flask application configured for testing.

    Uses an in-memory SQLite database so tests are fast and isolated.
    """
    db_path = _tmp_dir / "test.db"

    os.environ["SCAN_ROOT"] = str(_tmp_dir)

    from models import db
    from app import app as flask_app, setup_app
    from agregator.config import load_app_config

    cfg = load_app_config()
    flask_app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
            "SQLALCHEMY_TRACK_MODIFICATIONS": False,
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret-key",
            "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,
        }
    )
    setup_app(cfg, ensure_database=True)

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()


@pytest.fixture()
def client(app):
    """Flask test client for issuing HTTP requests."""
    return app.test_client()


@pytest.fixture()
def db_session(app):
    """Provide a transactional database session that rolls back after each test."""
    from models import db

    with app.app_context():
        connection = db.engine.connect()
        transaction = connection.begin()
        session = db.session
        yield session
        session.rollback()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def admin_user(app):
    """Return an admin ``User`` object, creating one if necessary."""
    from models import db, User

    with app.app_context():
        user = User.query.filter_by(username="admin").first()
        if not user:
            user = User(username="admin", role="admin")
            user.set_password("admin123")
            db.session.add(user)
            db.session.commit()
        return user


@pytest.fixture()
def viewer_user(app):
    """Return a non-privileged ``User`` (viewer role)."""
    from models import db, User

    with app.app_context():
        user = User.query.filter_by(username="viewer_test").first()
        if not user:
            user = User(username="viewer_test", role="viewer")
            user.set_password("viewer123")
            db.session.add(user)
            db.session.commit()
        return user


@pytest.fixture()
def auth_client(client, admin_user):
    """Flask test client with an authenticated admin session."""
    with client.session_transaction() as sess:
        sess["user_id"] = admin_user.id
    return client


@pytest.fixture()
def sample_collection(app, admin_user):
    """Create and return a sample ``Collection``."""
    from models import db, Collection

    with app.app_context():
        col = Collection(
            name="Test Collection",
            slug="test-collection",
            owner_id=admin_user.id,
        )
        db.session.add(col)
        db.session.commit()
        db.session.refresh(col)
        return col


@pytest.fixture()
def sample_file(app, sample_collection, _tmp_dir: Path):
    """Create and return a sample ``File`` record with a physical file on disk."""
    from models import db, File

    file_path = _tmp_dir / "test_document.txt"
    file_path.write_text("This is a test document for Agregator.", encoding="utf-8")

    with app.app_context():
        f = File(
            collection_id=sample_collection.id,
            path=str(file_path),
            rel_path="test_document.txt",
            filename="test_document.txt",
            ext=".txt",
            size=file_path.stat().st_size,
            title="Test Document",
            author="Test Author",
            material_type="article",
            text_excerpt="This is a test document for Agregator.",
        )
        db.session.add(f)
        db.session.commit()
        db.session.refresh(f)
        return f


@pytest.fixture()
def mock_llm():
    """Patch LLM HTTP calls to return deterministic mock responses.

    Usage::

        def test_something(mock_llm, auth_client):
            resp = auth_client.post("/api/ai-search", json={"query": "test"})
            assert resp.status_code == 200
    """
    with patch("requests.Session.post", side_effect=mock_llm_post) as m:
        yield m
