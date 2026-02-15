from __future__ import annotations


def test_settings_get_exposes_schema_metadata(auth_client):
    resp = auth_client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data.get("settings_hub_v2_enabled"), bool)
    runtime_fields = data.get("runtime_fields") or []
    assert runtime_fields, "runtime_fields must be present"
    sample = runtime_fields[0]
    for key in (
        "name",
        "api_key",
        "group",
        "type",
        "description",
        "runtime_mutable",
        "visibility",
        "restart_required",
        "constraints",
        "risk_level",
    ):
        assert key in sample, f"Missing schema key: {key}"


def test_settings_rejects_env_only_fields(auth_client):
    payload = {"runtime_fields": {"database_url": "postgresql://example"}}
    resp = auth_client.post("/api/settings", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data.get("error") == "validation_error"
    assert "database_url" in (data.get("field_errors") or {})


def test_settings_runtime_patch_with_schema_key(auth_client):
    payload = {"runtime_fields": {"rag_embedding_batch_size": 7}}
    resp = auth_client.post("/api/settings", json=payload)
    assert resp.status_code == 200
    again = auth_client.get("/api/settings").get_json()
    runtime_fields = again.get("runtime_fields") or []
    by_name = {field.get("name"): field for field in runtime_fields}
    assert by_name["rag_embedding_batch_size"]["value"] == 7
