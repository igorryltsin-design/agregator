"""Error and response helpers.

Extracted from ``app.py`` so that any Blueprint or service can produce
consistent JSON/HTML error responses without importing the monolith.
"""

from __future__ import annotations

from html import escape

from flask import Response, jsonify, make_response, request


def json_error(message: str, status: int = 400):
    """Return a JSON error tuple suitable as a Flask view return value."""
    return jsonify({"error": str(message)}), int(status)


def html_page(
    title: str,
    body: str,
    *,
    extra_head: str = "",
    status: int = 200,
) -> Response:
    """Return a minimal standalone HTML page."""
    head = f"""
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(title)}</title>
    {extra_head}
    """
    html = f'<!doctype html><html lang="ru"><head>{head}</head><body>{body}</body></html>'
    resp = make_response(html, status)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


def add_pipeline_cors_headers(response: Response) -> Response:
    """Add CORS headers suitable for the pipeline integration endpoint."""
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        vary = response.headers.get("Vary")
        response.headers["Vary"] = f"{vary}, Origin" if vary else "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    else:
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault(
        "Access-Control-Allow-Headers", "Content-Type, Authorization"
    )
    response.headers.setdefault("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response


def pipeline_cors_preflight() -> Response:
    """Handle an OPTIONS pre-flight for the pipeline endpoint."""
    response = make_response("", 204)
    return add_pipeline_cors_headers(response)
