import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agregator.osint.service import OsintSearchService
from agregator.osint.storage import (
    OsintRepository,
    OsintRepositoryConfig,
    OsintSchedule,
)


def test_sanitize_schedule_config_requires_interval():
    config = {"interval_minutes": 15, "label": "Test", "notify": True}
    sanitized = OsintSearchService._sanitize_schedule_config(config)
    assert sanitized is not None
    assert sanitized["interval_minutes"] == 15
    assert sanitized["label"] == "Test"
    assert sanitized["notify"] is True

    with pytest.raises(ValueError):
        OsintSearchService._sanitize_schedule_config({"interval_minutes": "abc"})

    with pytest.raises(ValueError):
        OsintSearchService._sanitize_schedule_config({"active": True})


def test_repository_schedule_lifecycle():
    repo = OsintRepository(OsintRepositoryConfig(url="sqlite:///:memory:"))
    job = repo.create_job(
        query="test query",
        locale="ru-RU",
        region=None,
        safe=False,
        sources=[{"type": "engine", "engine": "google"}],
        params={},
        user_id=None,
    )
    template = {
        "query": "test query",
        "locale": "ru-RU",
        "region": None,
        "safe": False,
        "sources": [{"type": "engine", "engine": "google"}],
        "params": {},
        "user_id": None,
    }
    schedule = repo.upsert_schedule(
        job_id=job["id"],
        schedule_id=None,
        template=template,
        interval_minutes=30,
        start_at=None,
        notify=False,
        notify_channel=None,
        label="Каждые 30 минут",
    )
    assert schedule["active"] is True
    assert schedule["interval_minutes"] == 30
    assert schedule["next_run_at"] is not None

    with repo.session() as session:
        record = session.get(OsintSchedule, schedule["id"])
        assert record is not None
        record.next_run_at = datetime.utcnow() - timedelta(minutes=1)
        session.commit()

    claimed = repo.claim_next_schedule()
    assert claimed is not None
    assert claimed["id"] == schedule["id"]
    assert claimed["template"]["query"] == "test query"

    # Mark successful run and ensure next run is scheduled
    repo.complete_schedule_run(schedule["id"], job_id=job["id"])
    refreshed = repo.claim_next_schedule()
    assert refreshed is None  # next_run_at is still in the future

    # Force next run to be due and ensure it can be claimed again
    with repo.session() as session:
        schedule_row = session.get(OsintSchedule, schedule["id"])
        assert schedule_row is not None
        schedule_row.next_run_at = datetime.utcnow() - timedelta(minutes=1)
        schedule_row.running = False
        session.commit()
    claimed_again = repo.claim_next_schedule()
    assert claimed_again is not None
    assert claimed_again["id"] == schedule["id"]
    repo.mark_schedule_failure(schedule["id"], delay_minutes=5, error="boom")
    assert repo.claim_next_schedule() is None
