"""Database storage for OSINT search jobs and results."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker, selectinload

from agregator.osint.markdown import render_markdown

class Base(DeclarativeBase):
    pass


class OsintJob(Base):
    __tablename__ = "osint_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(String, nullable=False, index=True)
    locale: Mapped[str] = mapped_column(String, nullable=False, default="ru-RU")
    region: Mapped[str | None] = mapped_column(String, nullable=True)
    safe: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued", index=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_error: Mapped[str | None] = mapped_column(String, nullable=True)
    ontology: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ontology_error: Mapped[str | None] = mapped_column(String, nullable=True)

    searches: Mapped[list["OsintSearch"]] = relationship(
        "OsintSearch",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    schedule: Mapped[Optional["OsintSchedule"]] = relationship(
        "OsintSchedule",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class OsintSearch(Base):
    __tablename__ = "osint_searches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("osint_jobs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    query: Mapped[str] = mapped_column(String, nullable=False, index=True)
    engine: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False, index=True, default="")
    locale: Mapped[str] = mapped_column(String, nullable=False, default="ru-RU")
    requested_url: Mapped[str | None] = mapped_column(String, nullable=True)
    final_url: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued", index=True)
    blocked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    from_cache: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    html_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_error: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    job: Mapped[Optional[OsintJob]] = relationship("OsintJob", back_populates="searches")
    results: Mapped[list["OsintResult"]] = relationship(
        "OsintResult",
        back_populates="search",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class OsintResult(Base):
    __tablename__ = "osint_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    search_id: Mapped[int] = mapped_column(
        ForeignKey("osint_searches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(String, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    search: Mapped[OsintSearch] = relationship("OsintSearch", back_populates="results")


class OsintSchedule(Base):
    __tablename__ = "osint_schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("osint_jobs.id", ondelete="CASCADE"),
        nullable=True,
        unique=True,
        index=True,
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    running: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    interval_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cron: Mapped[str | None] = mapped_column(String, nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notify: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notify_channel: Mapped[str | None] = mapped_column(String, nullable=True)
    template: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    job: Mapped[Optional[OsintJob]] = relationship("OsintJob", back_populates="schedule")


def _resolve_db_url(db_url: str | None) -> str:
    if db_url:
        return db_url
    env_url = os.getenv("OSINT_DATABASE_URL")
    if env_url:
        return env_url
    default_path = os.getenv("OSINT_DB_PATH")
    if not default_path:
        default_path = os.path.join(_resolve_instance_dir(), "osint_cache_v2.db")
    default_path = os.path.abspath(default_path)
    os.makedirs(os.path.dirname(default_path), exist_ok=True)
    return f"sqlite:///{default_path}"


def _resolve_instance_dir() -> str:
    candidates = [
        os.getenv("AGREGATOR_INSTANCE_PATH"),
        os.getenv("INSTANCE_PATH"),
        os.path.join(os.getcwd(), "instance"),
        os.path.join(os.path.expanduser("~"), ".agregator", "instance"),
        os.path.join(tempfile.gettempdir(), "agregator-instance"),
    ]
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(str(candidate))
        if path in checked:
            continue
        checked.add(path)
        if _is_writable_dir(path):
            return path
    raise OSError("Unable to find writable instance directory for OSINT storage")


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


@dataclass(slots=True)
class OsintRepositoryConfig:
    url: str | None = None
    echo: bool = False


class OsintRepository:
    """Repository for persisting OSINT jobs, sources and results."""

    def __init__(self, config: OsintRepositoryConfig | None = None) -> None:
        self.config = config or OsintRepositoryConfig()
        self.db_url = _resolve_db_url(self.config.url)
        self._engine = create_engine(self.db_url, echo=self.config.echo, future=True)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False, future=True)
        self._schema_checked = False

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        if self._schema_checked:
            return
        Base.metadata.create_all(self._engine)
        with self._engine.begin() as conn:
            self._ensure_columns(
                conn,
                "osint_jobs",
                {
                    "region": "TEXT",
                    "progress": "TEXT",
                    "params": "TEXT",
                    "source_total": "INTEGER DEFAULT 0",
                    "source_completed": "INTEGER DEFAULT 0",
                    "user_id": "INTEGER",
                    "started_at": "DATETIME",
                    "completed_at": "DATETIME",
                    "analysis": "TEXT",
                    "analysis_error": "TEXT",
                    "ontology": "TEXT",
                    "ontology_error": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "osint_searches",
                {
                    "job_id": "INTEGER",
                    "source": "TEXT DEFAULT ''",
                    "params": "TEXT",
                    "meta": "TEXT",
                    "error": "TEXT",
                    "started_at": "DATETIME",
                    "completed_at": "DATETIME",
                    "text_content": "TEXT",
                    "screenshot_path": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "osint_results",
                {
                    "snippet": "TEXT",
                    "score": "REAL",
                    "meta": "TEXT",
                    "text_content": "TEXT",
                    "screenshot_path": "TEXT",
                },
            )
        self._schema_checked = True

    @staticmethod
    def _ensure_columns(conn, table: str, columns: Dict[str, str]) -> None:
        if conn.dialect.name != "sqlite":
            # For PostgreSQL/MySQL we rely on metadata-based create_all.
            return
        existing: set[str] = set()
        for row in conn.execute(text(f"PRAGMA table_info({table})")):
            if hasattr(row, "_mapping") and "name" in row._mapping:
                existing.add(str(row._mapping["name"]))
            elif isinstance(row, (tuple, list)) and len(row) > 1:
                existing.add(str(row[1]))
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

    # ------------------------------------------------------------------
    # Session helper
    # ------------------------------------------------------------------
    def session(self) -> Session:
        return self._session_factory()

    # ------------------------------------------------------------------
    # Job orchestration
    # ------------------------------------------------------------------
    def create_job(
        self,
        *,
        query: str,
        locale: str,
        region: str | None,
        safe: bool,
        sources: List[dict],
        params: dict | None = None,
        user_id: int | None = None,
    ) -> dict:
        self.ensure_schema()
        created = datetime.utcnow()
        progress = {
            str(spec.get("id") or spec.get("source", f"src{idx}")): {
                "status": "queued",
                "label": spec.get("label") or spec.get("id") or spec.get("source") or f"Source {idx + 1}",
                "type": spec.get("type"),
                "created_at": created.isoformat(),
            }
            for idx, spec in enumerate(sources)
        }
        with self.session() as session:
            job = OsintJob(
                query=query,
                locale=locale,
                region=region,
                safe=safe,
                status="queued",
                sources=sources,
                progress=progress,
                params=params,
                source_total=len(sources),
                source_completed=0,
                user_id=user_id,
                created_at=created,
                started_at=None,
                completed_at=None,
                analysis=None,
                analysis_error=None,
                ontology=None,
                ontology_error=None,
            )
            session.add(job)
            session.flush()
            for idx, spec in enumerate(sources):
                source_id = str(spec.get("id") or spec.get("source") or f"src{idx}")
                entry = OsintSearch(
                    job_id=job.id,
                    query=query,
                    engine=spec.get("engine"),
                    source=source_id,
                    locale=locale,
                    status="queued",
                    blocked=False,
                    from_cache=False,
                    params=spec,
                    html_snapshot="",
                    created_at=created,
                )
                session.add(entry)
            session.commit()
            session.refresh(job)
            return self._serialize_job(job)

    def update_job_status(
        self,
        job_id: int,
        *,
        status: str | None = None,
        error: str | None = None,
        started: bool | None = None,
        completed: bool | None = None,
    ) -> None:
        self.ensure_schema()
        with self.session() as session:
            job = session.get(OsintJob, job_id)
            if not job:
                return
            now = datetime.utcnow()
            if started:
                if job.started_at is None:
                    job.started_at = now
                if job.status == "queued":
                    job.status = "running"
            if status:
                job.status = status
            if error:
                job.error = error
            if completed:
                job.completed_at = job.completed_at or now
            session.commit()

    def mark_source_status(
        self,
        job_id: int,
        source_id: str,
        *,
        status: str,
        error: str | None = None,
        extra: dict | None = None,
    ) -> None:
        self.ensure_schema()
        now = datetime.utcnow()
        with self.session() as session:
            search = (
                session.query(OsintSearch)
                .filter(OsintSearch.job_id == job_id, OsintSearch.source == source_id)
                .one_or_none()
            )
            job = session.get(OsintJob, job_id)
            if not job or not search:
                return
            search.status = status
            if status == "running":
                search.started_at = search.started_at or now
                if job.started_at is None:
                    job.started_at = now
                if job.status in {"queued", "pending"}:
                    job.status = "running"
            if status in {"completed", "error"}:
                search.completed_at = now
            if error:
                search.error = error
            progress = dict(job.progress or {})
            entry = dict(progress.get(source_id) or {})
            entry.update({
                "status": status,
                "updated_at": now.isoformat(),
            })
            if error:
                entry["error"] = error
            if extra:
                entry.update(extra)
            progress[source_id] = entry
            job.progress = progress
            job.source_completed = sum(
                1 for value in progress.values() if value.get("status") in {"completed", "error"}
            )
            if status == "error":
                job.status = "error"
                job.error = error
                job.completed_at = job.completed_at or now
            elif job.source_completed >= job.source_total and job.status != "error":
                job.status = "completed"
                job.completed_at = job.completed_at or now
            session.commit()

    def persist_source_result(
        self,
        *,
        job_id: int,
        source_id: str,
        engine: str | None,
        blocked: bool,
        from_cache: bool,
        html_snapshot: str | None,
        text_content: str | None,
        screenshot_path: str | None,
        llm_payload: str | None,
        llm_model: str | None,
        llm_error: str | None,
        status: str,
        error: str | None,
        metadata: dict | None,
        results: List[dict],
        requested_url: str | None = None,
        final_url: str | None = None,
    ) -> dict:
        self.ensure_schema()
        now = datetime.utcnow()
        with self.session() as session:
            search = (
                session.query(OsintSearch)
                .filter(OsintSearch.job_id == job_id, OsintSearch.source == source_id)
                .one_or_none()
            )
            if search is None:
                return {}
            session.query(OsintResult).filter(OsintResult.search_id == search.id).delete()
            rank = 1
            for item in results:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip() or str(item.get("url") or "")
                url = str(item.get("url") or "").strip()
                if not title or not url:
                    continue
                snippet = item.get("snippet")
                score_value = item.get("score")
                try:
                    score = float(score_value) if isinstance(score_value, (int, float)) else None
                except Exception:
                    score = None
                result_row = OsintResult(
                    search_id=search.id,
                    rank=rank,
                    title=title,
                    url=url,
                    snippet=str(snippet) if snippet is not None else None,
                    text_content=item.get("text_content") if isinstance(item.get("text_content"), str) else None,
                    screenshot_path=item.get("screenshot_path") if isinstance(item.get("screenshot_path"), str) else None,
                    score=score,
                    meta=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
                )
                session.add(result_row)
                rank += 1
            search.engine = engine
            search.blocked = blocked
            search.from_cache = from_cache
            if requested_url is not None:
                search.requested_url = requested_url
            if final_url is not None:
                search.final_url = final_url
            if html_snapshot is not None:
                search.html_snapshot = html_snapshot
            if text_content is not None:
                search.text_content = text_content
            if screenshot_path is not None:
                search.screenshot_path = screenshot_path
            search.llm_payload = llm_payload
            search.llm_model = llm_model
            search.llm_error = llm_error
            search.status = status
            search.error = error
            search.meta = metadata
            search.completed_at = now
            session.flush()
            session.commit()
        result_count = sum(1 for item in results if isinstance(item, dict))
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        extra_payload: dict[str, Any] = {
            "completed": True,
            "fallback": bool(metadata_dict.get("fallback")),
            "fallback_url": metadata_dict.get("fallback_url"),
            "final_url": metadata_dict.get("final_url") or metadata_dict.get("fallback_url"),
            "links_collected": result_count,
        }
        for key in (
            "results_parsed",
            "results_on_page",
            "results_forwarded",
            "pages_estimated",
            "requested_results",
            "keywords",
            "refined_query",
            "original_query",
        ):
            value = metadata_dict.get(key)
            if value is not None:
                extra_payload[key] = value
        if metadata_dict.get("pages_estimated") is not None and "pages_processed" not in extra_payload:
            extra_payload["pages_processed"] = metadata_dict.get("pages_estimated")
        if metadata_dict.get("fallback"):
            extra_payload["retry_available"] = True
        self.mark_source_status(
            job_id,
            source_id,
            status=status,
            error=error,
            extra=extra_payload,
        )
        return self.get_job(job_id)

    def record_source_error(
        self,
        *,
        job_id: int,
        source_id: str,
        message: str,
    ) -> dict:
        self.ensure_schema()
        with self.session() as session:
            search = (
                session.query(OsintSearch)
                .filter(OsintSearch.job_id == job_id, OsintSearch.source == source_id)
                .one_or_none()
            )
            if search:
                search.status = "error"
                search.error = message
                search.completed_at = datetime.utcnow()
                session.commit()
        self.mark_source_status(job_id, source_id, status="error", error=message)
        return self.get_job(job_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_job(self, job_id: int) -> dict:
        self.ensure_schema()
        with self.session() as session:
            stmt = (
                select(OsintJob)
                .options(
                    selectinload(OsintJob.searches).selectinload(OsintSearch.results),
                    selectinload(OsintJob.schedule),
                )
                .where(OsintJob.id == job_id)
            )
            job = session.execute(stmt).scalar_one_or_none()
            if not job:
                return {}
            return self._serialize_job(job)

    def list_jobs(self, limit: int = 10) -> List[dict]:
        self.ensure_schema()
        limit = max(1, min(int(limit or 10), 100))
        with self.session() as session:
            stmt = (
                select(OsintJob)
                .options(
                    selectinload(OsintJob.schedule),
                )
                .order_by(OsintJob.created_at.desc())
                .limit(limit)
            )
            jobs = session.execute(stmt).scalars().all()
            return [self._serialize_job(job, include_sources=False) for job in jobs]

    def delete_job(self, job_id: int) -> bool:
        self.ensure_schema()
        with self.session() as session:
            job = session.get(OsintJob, job_id)
            if not job:
                return False
            session.delete(job)
            session.commit()
            return True

    def set_job_analysis(self, job_id: int, summary: str | None, error: str | None) -> None:
        self.ensure_schema()
        with self.session() as session:
            job = session.get(OsintJob, job_id)
            if not job:
                return
            job.analysis = summary
            job.analysis_error = error
            session.commit()

    def set_job_ontology(self, job_id: int, payload: dict | None, error: str | None) -> None:
        self.ensure_schema()
        with self.session() as session:
            job = session.get(OsintJob, job_id)
            if not job:
                return
            job.ontology = payload if payload else None
            job.ontology_error = error
            session.commit()

    # ------------------------------------------------------------------
    # Schedule management
    # ------------------------------------------------------------------
    def upsert_schedule(
        self,
        *,
        job_id: int,
        schedule_id: int | None,
        template: dict,
        interval_minutes: int,
        start_at: datetime | None,
        notify: bool,
        notify_channel: str | None,
        label: str | None,
    ) -> dict:
        self.ensure_schema()
        now = datetime.utcnow()
        with self.session() as session:
            schedule: OsintSchedule | None = None
            if schedule_id:
                schedule = session.get(OsintSchedule, schedule_id)
            if schedule is None:
                schedule = (
                    session.query(OsintSchedule)
                    .filter(OsintSchedule.job_id == job_id)
                    .one_or_none()
                )
            if schedule is None:
                schedule = OsintSchedule()
                session.add(schedule)
            schedule.job_id = job_id
            schedule.interval_minutes = max(5, int(interval_minutes))
            schedule.label = label
            schedule.notify = notify
            schedule.notify_channel = notify_channel
            schedule.template = dict(template or {})
            schedule.active = True
            schedule.running = False
            schedule.last_error = None
            schedule.cron = None
            next_run_at = self._calculate_next_run(
                schedule.interval_minutes,
                start_at=start_at,
                fallback_reference=schedule.last_run_at or now,
            )
            schedule.next_run_at = next_run_at
            schedule.updated_at = now
            session.commit()
            session.refresh(schedule)
            return self._serialize_schedule_full(schedule)

    def disable_schedule(self, *, schedule_id: int | None = None, job_id: int | None = None) -> None:
        if schedule_id is None and job_id is None:
            return
        self.ensure_schema()
        with self.session() as session:
            schedule: OsintSchedule | None = None
            if schedule_id is not None:
                schedule = session.get(OsintSchedule, schedule_id)
            if schedule is None and job_id is not None:
                schedule = (
                    session.query(OsintSchedule)
                    .filter(OsintSchedule.job_id == job_id)
                    .one_or_none()
                )
            if not schedule:
                return
            schedule.active = False
            schedule.running = False
            schedule.next_run_at = None
            schedule.last_error = None
            schedule.updated_at = datetime.utcnow()
            session.commit()

    def claim_next_schedule(self) -> dict | None:
        self.ensure_schema()
        with self.session() as session:
            now = datetime.utcnow()
            schedule = (
                session.query(OsintSchedule)
                .filter(
                    OsintSchedule.active.is_(True),
                    OsintSchedule.running.is_(False),
                    OsintSchedule.next_run_at.isnot(None),
                    OsintSchedule.next_run_at <= now,
                )
                .order_by(OsintSchedule.next_run_at.asc(), OsintSchedule.id.asc())
                .first()
            )
            if not schedule:
                return None
            schedule.running = True
            schedule.updated_at = now
            session.commit()
            session.refresh(schedule)
            return self._serialize_schedule_full(schedule)

    def complete_schedule_run(self, schedule_id: int, *, job_id: int | None = None) -> dict | None:
        self.ensure_schema()
        with self.session() as session:
            schedule = session.get(OsintSchedule, schedule_id)
            if not schedule:
                return None
            now = datetime.utcnow()
            if job_id is not None:
                schedule.job_id = job_id
            schedule.last_run_at = now
            schedule.running = False
            schedule.last_error = None
            schedule.next_run_at = self._calculate_next_run(
                schedule.interval_minutes,
                start_at=None,
                fallback_reference=now,
            )
            schedule.updated_at = now
            session.commit()
            session.refresh(schedule)
            return self._serialize_schedule_full(schedule)

    def mark_schedule_failure(self, schedule_id: int, *, delay_minutes: int, error: str | None = None) -> dict | None:
        self.ensure_schema()
        with self.session() as session:
            schedule = session.get(OsintSchedule, schedule_id)
            if not schedule:
                return None
            now = datetime.utcnow()
            schedule.running = False
            schedule.last_error = error
            effective_delay = max(5, int(delay_minutes or 0))
            schedule.next_run_at = now + timedelta(minutes=effective_delay)
            schedule.updated_at = now
            session.commit()
            session.refresh(schedule)
            return self._serialize_schedule_full(schedule)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _serialize_job(self, job: OsintJob, *, include_sources: bool = True) -> dict:
        sources_info = job.sources or []
        progress = job.progress or {}
        data: dict[str, Any] = {
            "id": job.id,
            "query": job.query,
            "locale": job.locale,
            "region": job.region,
            "safe": bool(job.safe),
            "status": job.status,
            "error": job.error,
            "progress": progress,
            "sources_total": job.source_total,
            "sources_completed": job.source_completed,
            "created_at": _iso(job.created_at),
            "started_at": _iso(job.started_at),
            "completed_at": _iso(job.completed_at),
            "sources": [],
            "params": job.params or {},
            "source_specs": sources_info,
            "user_id": job.user_id,
            "analysis": getattr(job, "analysis", None),
            "analysis_error": getattr(job, "analysis_error", None),
            "analysis_html": render_markdown(getattr(job, "analysis", None)),
            "ontology": getattr(job, "ontology", None),
            "ontology_error": getattr(job, "ontology_error", None),
            "schedule": self._serialize_schedule(job.schedule),
        }
        if include_sources:
            data["sources"] = [self._serialize_search(search) for search in job.searches]
        return data

    def _serialize_search(self, search: OsintSearch) -> dict:
        return {
            "id": search.id,
            "job_id": search.job_id,
            "source": search.source,
            "engine": search.engine,
            "status": search.status,
            "blocked": bool(search.blocked),
            "from_cache": bool(search.from_cache),
            "params": search.params or {},
            "metadata": search.meta or {},
            "requested_url": search.requested_url,
            "final_url": search.final_url,
            "html_snapshot": search.html_snapshot,
             "text_content": search.text_content,
             "screenshot_path": search.screenshot_path,
            "llm_payload": search.llm_payload,
            "llm_model": search.llm_model,
            "llm_error": search.llm_error,
            "error": search.error,
            "created_at": _iso(search.created_at),
            "started_at": _iso(search.started_at),
            "completed_at": _iso(search.completed_at),
            "results": [self._serialize_result(item) for item in sorted(search.results, key=lambda x: x.rank)],
        }

    def _serialize_schedule(self, schedule: OsintSchedule | None) -> dict | None:
        if schedule is None:
            return None
        return {
            "id": schedule.id,
            "label": schedule.label,
            "active": bool(schedule.active),
            "running": bool(schedule.running),
            "interval_minutes": schedule.interval_minutes,
            "cron": schedule.cron,
            "next_run_at": _iso(schedule.next_run_at),
            "last_run_at": _iso(schedule.last_run_at),
            "notify": bool(schedule.notify),
            "notify_channel": schedule.notify_channel,
            "last_error": schedule.last_error,
        }

    def _serialize_schedule_full(self, schedule: OsintSchedule) -> dict:
        data = self._serialize_schedule(schedule) or {}
        data["template"] = dict(schedule.template or {})
        data["job_id"] = schedule.job_id
        return data

    @staticmethod
    def _calculate_next_run(
        interval_minutes: int | None,
        *,
        start_at: datetime | None,
        fallback_reference: datetime | None,
    ) -> datetime | None:
        if interval_minutes is None or interval_minutes <= 0:
            if start_at and start_at > datetime.utcnow():
                return start_at
            return None
        interval = max(5, int(interval_minutes))
        now = datetime.utcnow()
        if start_at and start_at > now:
            return start_at
        base = fallback_reference or now
        return base + timedelta(minutes=interval)

    @staticmethod
    def _serialize_result(item: OsintResult) -> dict:
        return {
            "id": item.id,
            "rank": item.rank,
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet,
            "text_content": item.text_content,
            "screenshot_path": item.screenshot_path,
            "score": item.score,
            "metadata": item.meta or {},
            "created_at": _iso(item.created_at),
        }


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return None
