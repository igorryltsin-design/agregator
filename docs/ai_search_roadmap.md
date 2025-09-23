# AI Search Enhancement Roadmap

## Objectives
- Improve relevance and transparency of keyword expansion.
- Increase quality and reuse of snippets via LLM and caching.
- Provide richer, real-time feedback in the UI about progress and result quality.
- Collect user feedback to continuously refine ranking.
- Instrument performance metrics to guide optimization.

## Workstreams

### 1. Keyword Intelligence
- **Adaptive stoplist:** persist `idf` stats and automatically ban tokens with low discriminative power over time.
- **User feedback loop:** record clicks / "Нет результата" events; store in `keyword_feedback` table.
- **UI exposure:** display generated keywords as editable chips; let user remove/add before running search.

### 2. Snippet Excellence
- **Dual-mode caching:** store raw and LLM-refined snippets keyed by `(file_id, query_hash)`.
- **LLM budget control:** throttle snippet generation by time/usage; surface ETA in progress stream.
- **Snippet quality audit:** add admin view to compare cache hit ratios and freshness.

### 3. Deep Search Performance
- **Parallel chunk scanning:** use thread pool within `_deep_scan_file` (configurable worker count).
- **Incremental streaming:** send SSE events on every chunk processed; include running timer.
- **Configurable depth profiles:** presets (Fast, Balanced, Thorough) adjusting `chunk_chars`, `max_chunks`, `llm_snippets`.

### 4. UX & Transparency
- **Timeline UI:** convert progress list to vertical timeline with icons (keywords, candidates, deep scan, LLM steps).
- **Live counters:** show number of files scanned / remaining, ETA, LLM usage indicator.
- **Result annotations:** badges for source type (cache/full-text/LLM), highlight matched terms frequency.

### 5. Feedback & Analytics
- **Relevance feedback widget:** allow marking sources as "точно подходит" / "мимо"; log to backend.
- **Dashboard:** grafana-ready endpoint `/api/admin/ai-search-metrics` with per-stage durations.
- **Alerting:** emit warnings when LLM latency crosses threshold or stream fallback usage spikes.

### 6. Infrastructure & Ops
- **Cache storage:** introduce `redis` (or sqlite table) namespace for snippet + keyword caches with TTL.
- **Background cleaner:** nightly job refreshing hot documents, pruning stale cache entries.
- **Config management:** expose new knobs in `/api/settings` + UI (LLM snippet default, worker count, cache TTLs).

## Sequencing
1. Implement logging/metrics (Workstreams 3 & 5) to baseline current performance.
2. Add keyword feedback + UI chips (Workstream 1) to address noisy terms quickly.
3. Integrate snippet caching and optional LLM refinement with throttling (Workstream 2 & 6).
4. Enhance UI timeline + counters (Workstream 4) based on new progress events.
5. Iterate on parallel deep scan and presets once metrics highlight bottlenecks.

## Risks & Mitigations
- **LLM cost/time:** add per-request budget caps and graceful degradation to cached snippets.
- **Concurrency issues:** ensure thread pool respects Python GIL by using `concurrent.futures.ThreadPoolExecutor` with IO-bound tasks only.
- **User privacy:** anonymize feedback logs; allow opt-out.

## Next Steps
- Define DB migrations for feedback tables & cache metadata.
- Select cache backend (Redis vs SQLite) and update deployment docs.
- Prototype timeline UI with mock SSE events.
