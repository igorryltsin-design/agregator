"""Middleware utilities for Agregator Flask application."""

from .auth import (
    PIPELINE_API_KEY,
    ROLE_ADMIN,
    ROLE_EDITOR,
    ROLE_VIEWER,
    extract_pipeline_token,
    check_pipeline_access,
    normalize_user_role,
    user_role,
)
from .errors import json_error, html_page, add_pipeline_cors_headers, pipeline_cors_preflight

__all__ = [
    "PIPELINE_API_KEY",
    "ROLE_ADMIN",
    "ROLE_EDITOR",
    "ROLE_VIEWER",
    "extract_pipeline_token",
    "check_pipeline_access",
    "normalize_user_role",
    "user_role",
    "json_error",
    "html_page",
    "add_pipeline_cors_headers",
    "pipeline_cors_preflight",
]
