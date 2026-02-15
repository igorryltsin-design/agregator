"""Bootstrap helpers that assemble all runtime components."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import AppConfig, load_config
from .database import Database
from .logging_setup import configure_logging
from .services.pipeline import AggregationPipeline
from .services.scheduler import SchedulerService
from .telemetry import Telemetry


class BootstrapContext:
    def __init__(
        self,
        config: AppConfig,
        database: Database,
        telemetry: Telemetry,
        pipeline: AggregationPipeline,
        scheduler: SchedulerService,
    ) -> None:
        self.config = config
        self.database = database
        self.telemetry = telemetry
        self.pipeline = pipeline
        self.scheduler = scheduler


def bootstrap_pipeline(base_dir: Path | None = None) -> BootstrapContext:
    configure_logging()
    config = load_config(base_dir)
    logging.getLogger("aggregator.bootstrap").info("Loaded config for %s", config.extra)
    database = Database(config.database_url)
    database.create_all()

    telemetry = Telemetry(config.telemetry)
    pipeline = AggregationPipeline(config, database, telemetry)
    scheduler = SchedulerService(config)
    scheduler.add_task(
        "telemetry_flush",
        interval=config.telemetry.export_interval.total_seconds(),
        handler=lambda: logging.getLogger("aggregator.telemetry").info(
            "Telemetry snapshot %s", telemetry.snapshot()
        ),
    )
    scheduler.start()
    telemetry.start()
    return BootstrapContext(config, database, telemetry, pipeline, scheduler)
