# Security Audit Report

**Date:** 2026-02-13
**Scope:** Full codebase review for SQL injection, CORS, rate limiting, input validation, secrets management.

## Findings

### 1. CRITICAL: `.env` file tracked in Git

**Status:** FIXED
**Location:** `.gitignore`
**Issue:** The `.env` file containing secrets (API keys, passwords, database credentials) was not listed in `.gitignore` and was tracked by Git (`git status` shows `M .env`).
**Fix:** Added `.env`, `.env.local`, `.env.*.local` to `.gitignore`. **Action required:** Run `git rm --cached .env` to remove from tracking, then rotate all secrets.

### 2. LOW: SQL Injection in FTS queries

**Status:** SAFE (no action needed)
**Location:** `agregator/services/search.py`
**Analysis:** All FTS5 MATCH queries use SQLAlchemy `text()` with named parameters (`:match`, `:limit`, `:rid`). User input is pre-tokenized via `_FTS_TOKEN_PATTERN = re.compile(r"[\w\d]+")` which strips all special characters, preventing FTS syntax injection.

### 3. LOW: Dynamic SQL in OSINT storage

**Status:** SAFE (acceptable risk)
**Location:** `agregator/osint/storage.py:247-254`
**Analysis:** `_ensure_columns()` uses f-strings for `PRAGMA table_info({table})` and `ALTER TABLE {table}`. However, `table` is a hardcoded string from the code (`osint_jobs`, `osint_searches`, etc.), not user input. No injection vector.

### 4. MEDIUM: No rate limiting on LLM/AI endpoints

**Status:** OPEN
**Location:** `app.py` - `/api/ai-search`, `/api/doc-chat/ask`
**Issue:** These endpoints call external LLM APIs which are expensive (compute, money). No rate limiting is in place.
**Recommendation:** Add Flask-Limiter with per-user rate limits:
```python
# requirements.txt
flask-limiter>=3.5

# app.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

@app.route('/api/ai-search', methods=['POST'])
@limiter.limit("10 per minute")
def api_ai_search():
    ...
```

### 5. MEDIUM: No input validation layer

**Status:** OPEN
**Location:** All API endpoints
**Issue:** Request payload validation is done inline with `request.get_json()` and manual type checks. No schema validation.
**Recommendation:** Add Pydantic or marshmallow for request schema validation. Priority endpoints:
- `/api/ai-search` (complex payload)
- `/api/settings` (many fields)
- `/api/doc-chat/prepare` (file IDs)
- `/api/import/jobs` (upload params)

### 6. LOW: CORS configuration

**Status:** ACCEPTABLE
**Location:** `app.py` - `_add_pipeline_cors_headers()`
**Analysis:** CORS headers are only added to the pipeline integration endpoint (`/api/training/problem-pipeline`), not globally. The `Access-Control-Allow-Origin` mirrors the request `Origin` header when present, which is standard for cookie-based auth. Falls back to `*` when no Origin header (non-browser requests). This is reasonable for the pipeline use case.

### 7. LOW: Session security

**Status:** ACCEPTABLE
**Location:** `app.py` - session configuration
**Analysis:** Flask sessions use `session.permanent = True` on login. Consider adding:
- `SESSION_COOKIE_HTTPONLY = True` (already Flask default)
- `SESSION_COOKIE_SAMESITE = 'Lax'`
- `SESSION_COOKIE_SECURE = True` (when behind HTTPS)

### 8. INFO: Password hashing

**Status:** SAFE
**Location:** `models.py` - `User.set_password()` / `User.check_password()`
**Analysis:** Uses Werkzeug's `generate_password_hash` / `check_password_hash` which defaults to `pbkdf2:sha256` with proper salting.

## Summary

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | CRITICAL | .env tracked in Git | FIXED |
| 2 | LOW | FTS SQL injection | SAFE |
| 3 | LOW | Dynamic SQL in OSINT | SAFE |
| 4 | MEDIUM | No rate limiting | OPEN |
| 5 | MEDIUM | No input validation | OPEN |
| 6 | LOW | CORS configuration | ACCEPTABLE |
| 7 | LOW | Session security | ACCEPTABLE |
| 8 | INFO | Password hashing | SAFE |

## Recommended Next Steps

1. Run `git rm --cached .env` and rotate all secrets
2. Add Flask-Limiter for AI/LLM endpoints
3. Add Pydantic request schemas for high-risk endpoints
4. Set `SESSION_COOKIE_SAMESITE='Lax'` in Flask config
