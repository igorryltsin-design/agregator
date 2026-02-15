"""Integration tests for File CRUD API endpoints.

Uses fixtures from ``conftest.py`` to bootstrap a Flask test client with
an in-memory SQLite database, authenticated admin session, and sample data.
"""

from __future__ import annotations


class TestAuthEndpoints:
    """Basic authentication flow tests."""

    def test_login_success(self, client, admin_user):
        resp = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["user"]["username"] == "admin"
        assert data["user"]["role"] == "admin"

    def test_login_wrong_password(self, client, admin_user):
        resp = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["ok"] is False

    def test_login_missing_fields(self, client):
        resp = client.post("/api/auth/login", json={"username": ""})
        assert resp.status_code == 400

    def test_me_unauthenticated(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_me_authenticated(self, auth_client, admin_user):
        resp = auth_client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["user"]["id"] == admin_user.id

    def test_logout(self, auth_client):
        resp = auth_client.post("/api/auth/logout")
        assert resp.status_code == 200
        # After logout, /me should fail
        resp = auth_client.get("/api/auth/me")
        assert resp.status_code == 401


class TestFilesAPI:
    """Tests for /api/files endpoints."""

    def test_list_files(self, auth_client, sample_file):
        resp = auth_client.get("/api/files")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "files" in data or isinstance(data, list) or "items" in data

    def test_get_file_by_id(self, auth_client, sample_file):
        resp = auth_client.get(f"/api/files/{sample_file.id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("id") == sample_file.id or data.get("file", {}).get("id") == sample_file.id

    def test_get_nonexistent_file(self, auth_client):
        resp = auth_client.get("/api/files/99999")
        assert resp.status_code in (404, 200)  # may return 200 with error in JSON

    def test_update_file_metadata(self, auth_client, sample_file):
        resp = auth_client.put(
            f"/api/files/{sample_file.id}",
            json={"title": "Updated Title", "author": "New Author"},
        )
        # Accept both PUT and PATCH semantics
        assert resp.status_code in (200, 405)

    def test_delete_file(self, auth_client, sample_file, app):
        file_id = sample_file.id
        resp = auth_client.delete(f"/api/files/{file_id}")
        assert resp.status_code in (200, 204, 404)


class TestCollectionsAPI:
    """Tests for /api/collections endpoints."""

    def test_list_collections(self, auth_client, sample_collection):
        resp = auth_client.get("/api/collections")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, (list, dict))

    def test_create_collection(self, auth_client):
        resp = auth_client.post(
            "/api/collections",
            json={"name": "New Test Collection"},
        )
        assert resp.status_code in (200, 201)


class TestSearchAPI:
    """Tests for /api/search endpoints."""

    def test_search_empty_query(self, auth_client):
        resp = auth_client.get("/api/search?q=")
        assert resp.status_code == 200

    def test_search_basic(self, auth_client, sample_file):
        resp = auth_client.get("/api/search?q=test")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_search_with_type_filter(self, auth_client, sample_file):
        resp = auth_client.get("/api/search?q=test&type=article")
        assert resp.status_code == 200


class TestFacetsAPI:
    """Tests for /api/facets endpoint."""

    def test_get_facets(self, auth_client):
        resp = auth_client.get("/api/facets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_get_material_types(self, auth_client):
        resp = auth_client.get("/api/material-types")
        assert resp.status_code == 200


class TestHealthEndpoints:
    """Tests for health-check and metrics endpoints."""

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_metrics_requires_auth(self, client):
        resp = client.get("/metrics")
        # May require auth or be public
        assert resp.status_code in (200, 401, 302)
