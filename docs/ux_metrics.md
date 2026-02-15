# UX Metrics Baseline

## Purpose

This document tracks UX improvements introduced in the interface upgrade and provides a repeatable checklist for phase-by-phase verification.

## Core Metrics

| Metric | Baseline Method | Target |
|---|---|---|
| Clicks to key admin flows | Count click path from `/` to `admin/status`, `admin/tasks`, `admin/collections` | -20% vs current baseline |
| Coverage of unified states | Audit key routes for `empty/loading/error` consistency | 100% on key routes |
| UI regressions per release | Count UX-related bugs in release notes/issues | Downward trend |
| Token consistency | Search for hardcoded color values in UI modules | No new hardcoded tokens in shared UI |

## Baseline Snapshot Procedure

1. Use a standard admin account and standard editor account.
2. Record click count for:
   - catalog -> service status
   - catalog -> tasks
   - catalog -> collections
3. Visit key routes and record whether each has:
   - empty state
   - loading state
   - error boundary fallback
4. Run code search for hardcoded color literals in `frontend/src` and log count.

## Key Route Coverage Checklist

- `/`
- `/doc-chat`
- `/osint`
- `/stats`
- `/ingest`
- `/profile`
- `/settings`
- `/admin/status`
- `/admin/tasks`
- `/admin/logs`
- `/admin/llm`
- `/admin/collections`
- `/admin/ai-metrics`
- `/admin/facets`

## Release Validation Checklist

For each UX release phase:

- [ ] Navigation path click counts measured and compared to baseline
- [ ] Key route coverage checklist re-validated
- [ ] No critical visual regressions in dark/light themes
- [ ] Toast, offline, and error fallback behavior manually verified
- [ ] Result summary added to this file under "History"

## History

### 2026-02-13

- Created baseline template and validation checklist.
- Added measurable target definitions for navigation efficiency and UX state coverage.
