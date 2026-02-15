# Agregator UI — E2E Smoke Test Report

**Date:** 2026-02-14  
**App URL:** http://localhost:5050/app  
**Tester:** Automated (Playwright + API checks)

---

## Overall result: **PARTIAL**

---

## Passed checks

- **App root load:** `GET http://localhost:5050/app` returns HTTP 200 (verified via curl; ~1.7s response time).
- **Login API:** `POST /api/auth/login` with `admin` / `admin123` returns 200 and valid user payload:
  ```json
  {"ok":true,"user":{"username":"admin","role":"admin","permissions":{"can_admin":true,...}}}
  ```
- **Auth flow (browser):** Login form accepts credentials; redirects away from `/login` after successful auth (observed in earlier Playwright run).
- **Admin logs page:** Loads and URL contains `logs` (observed in earlier run).
- **Admin tasks page:** Loads and URL contains `tasks` (observed in earlier run).
- **Route structure:** All target routes exist in `main.tsx`: `/`, `/settings`, `/admin/status`, `/admin/logs`, `/admin/tasks`.

---

## Failed checks

| Area | Observed behavior | Error text |
|------|-------------------|------------|
| **Catalogue page** | Catalogue content check failed | "Catalogue elements not found" — body did not match expected text (Поиск/Каталог/Agregator) |
| **Settings DB panel** | DB management UI not found | "DB UI not found" — SQLite/PostgreSQL selector or "Управление базой данных" not present in body |
| **Settings migration wizard** | Migration block not found | "Migration block not found" — host/port/db/user/password or host presets not visible |
| **Admin status page** | Status content not found | "Status content not found" — or navigation timed out before content loaded |
| **Browser connectivity** | Intermittent failures | `net::ERR_CONNECTION_RESET`, `page.goto: Timeout 15000ms exceeded` |

---

## Blockers and assumptions

1. **Browser environment:** Playwright tests hit `ERR_CONNECTION_RESET` and timeouts when loading localhost. Likely causes: sandbox/network limits, server closing connections to headless clients, or IPv4/IPv6 resolution.
2. **Settings Hub V2:** DB management and migration wizard are behind `expertVisible` (`!hubV2Enabled || activeTab === 'expert'`). If `settings_hub_v2_enabled` is true, the user must open the **Expert** tab before the DB panel is visible.
3. **Admin status API:** Page calls `/api/admin/status/overview`. Slow or failing responses produce "Не удалось загрузить состояние сервиса"; tests did not confirm a successful load.
4. **Credentials assumed:** `admin` / `admin123` from test fixtures; no other accounts were tried.

---

## Code-based verification (no live UI)

Verified from source:

- **Settings DB management** (`SettingsPage.tsx` ~2046–2090):
  - DB type selector: SQLite (legacy .db) / PostgreSQL
  - Backup, import, clear-db controls
- **Settings migration wizard** (`SettingsPage.tsx` ~2093–2170):
  - SQLite path, PostgreSQL URL
  - Host (with presets: localhost, host.docker.internal, postgres), port, db, user, password
  - Mode: dry-run / run
  - "Запустить миграцию" button

---

## Recommended next steps (prioritized)

1. **Browser connectivity:** Run smoke tests on the same host as the server (e.g. CI or dev machine), or use `browser.newContext({ ignoreHTTPSErrors: true })` and bypass proxy, to avoid `ERR_CONNECTION_RESET`.
2. **Settings Expert tab:** When testing DB panel, add explicit click on the "Expert" tab if the Settings Hub V2 UI is enabled.
3. **Admin status API:** Inspect `/api/admin/status/overview` latency and errors; add timeouts or retries in the status page.
4. **Stabilize Playwright:** Use `waitUntil: 'commit'` or `'domcontentloaded'`, avoid `networkidle` on pages with background polling.
5. **Manual smoke pass:** Use an interactive browser to confirm Catalogue, Settings (including Expert tab), and Admin pages visually and functionally.
