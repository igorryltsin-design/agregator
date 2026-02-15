# Aggregator SPPR

Aggregator SPPR is a standalone early-fusion component for multimodal signals. The module registers data sources, normalizes weighted streams, builds relation graphs, and forms aligned selections for downstream analytics.

## Features
- Flask REST API for registering sources, ingesting signals, and launching fusion windows
- Reliability/intensity scoring with timestamp normalization
- Relation graph construction with provenance-aware edges and event history
- CLI for batch loading and diagnostics
- Built-in telemetry plus lightweight background scheduler

## Run
```bash
python -m program_modules.aggregator_sppp.cli runserver --host 0.0.0.0 --port 8051
```

CLI commands: `register-source`, `ingest-signal`, `fuse-window`, `stats`.
