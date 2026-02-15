"""Authentication and authorization helpers.

Extracted from ``app.py`` to isolate pipeline-key checking, role normalization
and role constants.  Route-level decorators remain in ``app.py`` because they
depend on the ``app`` instance directly, but the underlying logic lives here.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from flask import Response, jsonify, request

from models import User

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_ADMIN = "admin"
ROLE_EDITOR = "editor"
ROLE_VIEWER = "viewer"

_ROLE_ALIASES: dict[str, str] = {
    "user": ROLE_EDITOR,
    "editor": ROLE_EDITOR,
    "viewer": ROLE_VIEWER,
    "admin": ROLE_ADMIN,
}

PIPELINE_API_KEY: str = (os.getenv("PIPELINE_API_KEY") or "").strip()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_user_role(value: str | None, *, default: str = ROLE_VIEWER) -> str:
    """Normalize a raw role string to one of the canonical role constants."""
    if not value:
        return default
    token = str(value).strip().lower()
    return _ROLE_ALIASES.get(token, default if token not in _ROLE_ALIASES else token)


def user_role(user: User | None) -> str:
    """Return the canonical role for *user*."""
    if not user:
        return ROLE_VIEWER
    return normalize_user_role(getattr(user, "role", None), default=ROLE_EDITOR)


def extract_pipeline_token() -> Optional[str]:
    """Extract an API token from the current request headers."""
    header = request.headers.get("Authorization")
    token: Optional[str] = None
    if header:
        lowered = header.lower()
        if lowered.startswith("bearer "):
            token = header[7:].strip()
        else:
            token = header.strip()
    if not token:
        token = request.headers.get("X-Agregator-Key")
        if token:
            token = token.strip()
    if not token:
        token = request.headers.get("X-Agregator-Token")
        if token:
            token = token.strip()
    return token or None


def check_pipeline_access(load_current_user_fn) -> tuple[bool, Optional[Response]]:
    """Check whether the current request has pipeline-level access.

    Parameters
    ----------
    load_current_user_fn:
        Callable that returns the current ``User`` (or ``None``).

    Returns
    -------
    tuple:
        ``(True, None)`` if access is granted, otherwise
        ``(False, <403 Response>)``.
    """
    user = load_current_user_fn()
    if user_role(user) == ROLE_ADMIN:
        return True, None
    if PIPELINE_API_KEY:
        token = extract_pipeline_token()
        if token and hmac.compare_digest(token, PIPELINE_API_KEY):
            return True, None
        resp = jsonify({"ok": False, "error": "Forbidden"})
        resp.status_code = 403
        return False, resp
    return True, None
