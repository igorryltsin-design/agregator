"""Flask Blueprint registry.

Import and register all API blueprints here.  The ``register_blueprints``
helper is called from ``setup_app()`` in ``app.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask


def register_blueprints(app: Flask) -> None:
    """Register all application blueprints that are not yet registered.

    Blueprints that were previously registered directly inside ``app.py``
    (admin_bp, users_bp, osint_bp) are still imported from there for now.
    New blueprints created during the decomposition are registered here.
    """
    # The following blueprints are still defined in app.py for now and
    # registered in setup_app().  This function serves as the central
    # registry for *new* blueprints extracted during the refactor.
    pass
