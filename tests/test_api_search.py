"""Integration tests for AI search and document chat API endpoints."""

from __future__ import annotations

import json


class TestAiSearchAPI:
    """Tests for /api/ai-search endpoint."""

    def test_ai_search_requires_auth(self, client):
        resp = client.post("/api/ai-search", json={"query": "test"})
        assert resp.status_code in (401, 302)

    def test_ai_search_empty_query(self, auth_client):
        resp = auth_client.post("/api/ai-search", json={"query": ""})
        # Should return a validation error or empty results
        assert resp.status_code in (200, 400)

    def test_ai_search_basic(self, auth_client, sample_file, mock_llm):
        resp = auth_client.post(
            "/api/ai-search",
            json={"query": "тестовый документ", "stream": False},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_ai_search_streaming(self, auth_client, sample_file, mock_llm):
        resp = auth_client.post(
            "/api/ai-search",
            json={"query": "тестовый документ", "stream": True},
        )
        assert resp.status_code == 200
        # Streaming returns NDJSON lines
        content_type = resp.content_type or ""
        assert "ndjson" in content_type or "json" in content_type


class TestSearchFeedbackAPI:
    """Tests for /api/ai-search/feedback endpoint."""

    def test_feedback_requires_auth(self, client):
        resp = client.post(
            "/api/ai-search/feedback",
            json={"file_id": 1, "query_hash": "abc", "relevant": True},
        )
        assert resp.status_code in (401, 302)

    def test_feedback_submit(self, auth_client, sample_file):
        resp = auth_client.post(
            "/api/ai-search/feedback",
            json={
                "file_id": sample_file.id,
                "query_hash": "test_hash_123",
                "relevant": True,
            },
        )
        assert resp.status_code in (200, 400)

    def test_classic_search_feedback(self, auth_client, sample_file):
        resp = auth_client.post(
            "/api/search/feedback",
            json={
                "file_id": sample_file.id,
                "query": "test query",
                "relevant": True,
            },
        )
        assert resp.status_code in (200, 400)


class TestDocChatAPI:
    """Tests for /api/doc-chat/* endpoints."""

    def test_doc_chat_documents_requires_auth(self, client):
        resp = client.get("/api/doc-chat/documents")
        assert resp.status_code in (401, 302)

    def test_doc_chat_documents_list(self, auth_client):
        resp = auth_client.get("/api/doc-chat/documents")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_doc_chat_prepare_no_files(self, auth_client):
        resp = auth_client.post(
            "/api/doc-chat/prepare",
            json={"file_ids": []},
        )
        assert resp.status_code in (200, 400)

    def test_doc_chat_prepare_with_file(self, auth_client, sample_file, mock_llm):
        resp = auth_client.post(
            "/api/doc-chat/prepare",
            json={"file_ids": [sample_file.id]},
        )
        assert resp.status_code in (200, 202)

    def test_doc_chat_ask_no_session(self, auth_client):
        resp = auth_client.post(
            "/api/doc-chat/ask",
            json={"question": "test question"},
        )
        # Should fail or return error without a valid session
        assert resp.status_code in (200, 400)

    def test_doc_chat_preferences_get(self, auth_client):
        resp = auth_client.get("/api/doc-chat/preferences")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_doc_chat_preferences_update(self, auth_client):
        resp = auth_client.post(
            "/api/doc-chat/preferences",
            json={"tone": "academic", "detail": "deep", "language": "ru"},
        )
        assert resp.status_code == 200

    def test_doc_chat_clear(self, auth_client):
        resp = auth_client.post("/api/doc-chat/clear", json={})
        assert resp.status_code == 200
