"""Timeline aggregation logic."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import EventRepository, TimelineRepository


class TimelineService:
    def __init__(self, config: AppConfig):
        self.config = config

    def refresh(self, session: Session) -> List[dict]:
        event_repo = EventRepository(session)
        timeline_repo = TimelineRepository(session)
        events = event_repo.latest(limit=self.config.timeline.max_events)
        markers: List[dict] = []
        if not events:
            return markers
        window = self.config.timeline.aggregation_window
        bucket = []
        bucket_start = events[0].start_at
        for event in events:
            if (bucket_start - event.start_at) > window:
                markers.append(self._flush_bucket(timeline_repo, bucket))
                bucket = []
                bucket_start = event.start_at
            bucket.append(event)
        if bucket:
            markers.append(self._flush_bucket(timeline_repo, bucket))
        return markers

    def _flush_bucket(self, repo: TimelineRepository, bucket):
        bucket = sorted(bucket, key=lambda event: event.start_at)
        summary = "; ".join(event.title for event in bucket[:3])
        marker = repo.upsert_marker(bucket[-1].start_at, summary, [event.id for event in bucket])
        return {
            "date": marker.date.isoformat(),
            "summary": marker.summary,
            "event_ids": marker.event_ids,
        }
