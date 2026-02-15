"""High-level OSINT search orchestration with multi-source support."""

from __future__ import annotations

import fnmatch
import hashlib
import math
import random
import json
import logging
import os
import re
import threading
import socket
import time
import html
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

from flask import current_app

from agregator.services import SearchService, get_task_queue
from agregator.osint.analysis import (
    build_analysis_context,
    analysis_source_alerts,
    analysis_llm_notes,
    build_structured_payload,
    build_analysis_messages,
    build_analysis_fallback,
)
from agregator.osint.parser import ParsedSerpPayload, SerpParser
from agregator.osint.serp import SerpFetcher, SerpRequest
from agregator.osint.storage import OsintRepository, OsintRepositoryConfig
from models import File, Tag, db


@dataclass(slots=True)
class LocalSourceOptions:
    mode: str = "catalog"  # catalog | filesystem
    path: str | None = None
    limit: int = 20
    recursive: bool = True
    ocr: bool = False
    exclude_patterns: tuple[str, ...] = ()


_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")

_OSINT_STOP_WORDS = {
    "и","или","а","но","же","то","ли","не","ни","да","уж","как","так","что","кто","где","когда","зачем","почему",
    "какой","какая","какие","каков","это","эта","этот","эти","того","тому","этом","этих","тех","там","тут","здесь",
    "бы","либо","пусть","дабы","быть","есть","нет","между","через","после","перед","около","возле","у","к","ко",
    "от","до","для","по","под","над","о","об","обо","при","без","из","изза","с","со","ну","в","во","на","ещё","также",
    "тоже","сам","сама","сами","само","свой","своя","свои","моё","мой","моя","мои","твой","твоя","твои","наш","наша",
    "наши","ваш","ваша","ваши","тот","та","то","те","прочее","другой","иная","иное","иные",
    "the","a","an","and","or","but","if","then","else","when","where","why","how","what","who","whom","whose",
    "this","that","these","those","is","are","was","were","be","been","being","to","of","in","on","for","with","at",
    "by","from","as","about","into","through","after","over","between","out","against","during","without","before",
    "under","around","among","it","its","we","you","they","he","she","them","his","her","their","our","your","my",
    "me","us","do","does","did","not","no","yes"
}


_TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".html",
    ".htm",
    ".xml",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".conf",
    ".log",
    ".sql",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".go",
    ".rb",
    ".php",
    ".sh",
    ".bat",
    ".ps1",
    ".toml",
    ".tex",
    ".mdx",
}

_DOCUMENT_FILE_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".rtf",
    ".epub",
    ".djvu",
}

_SERP_RESULTS_PER_PAGE = 10
SUPPORTED_OSINT_LOCALES = ("ru-RU", "en-US")


class OsintSearchService:
    """Coordinates multi-source OSINT searches, including remote engines and local files."""

    def __init__(
        self,
        *,
        fetcher: SerpFetcher | None = None,
        parser: SerpParser | None = None,
        repository: OsintRepository | None = None,
        repository_config: OsintRepositoryConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.fetcher = fetcher or SerpFetcher()
        self.parser = parser or SerpParser()
        self.repository = repository or OsintRepository(repository_config)
        self.logger = logger or logging.getLogger("agregator.osint.service")
        self.queue = get_task_queue()
        self._schedule_lock = threading.Lock()
        self._schedule_thread: threading.Thread | None = None
        self._schedule_thread_started = False
        self._schedule_event = threading.Event()
        self._network_probe_lock = threading.Lock()
        self._network_probe_cache: tuple[float, bool] | None = None

    # ------------------------------------------------------------------
    # LLM analysis helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _analysis_context(snapshot: dict, *, max_items: int = 3, snippet_limit: int = 220) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        sources = snapshot.get("sources") or []
        for source in sources:
            metadata = source.get("metadata") or {}
            label = (
                metadata.get("label")
                or source.get("source")
                or source.get("engine")
                or "Источник"
            )
            entries: list[dict[str, str]] = []
            for item in (source.get("results") or [])[:max_items]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("url") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet_raw = str(item.get("snippet") or "").strip()
                if not title:
                    continue
                snippet = snippet_raw
                if snippet_limit and len(snippet) > snippet_limit:
                    snippet = snippet[: snippet_limit - 1].rstrip() + "…"
                entries.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
            )
            contexts.append(
                {
                    "label": label,
                    "fallback": bool(metadata.get("fallback")),
                    "alerts": OsintSearchService._analysis_source_alerts(source),
                    "notes": OsintSearchService._analysis_llm_notes(source),
                    "entries": entries,
                }
            )
        return contexts

    @staticmethod
    def _analysis_source_alerts(source: dict) -> list[str]:
        alerts: list[str] = []
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        if metadata.get("fallback"):
            alerts.append("Режим fallback — требуется ручная проверка.")
        if metadata.get("from_cache") or source.get("from_cache"):
            alerts.append("Ответ из кэша — информация может быть устаревшей.")
        if metadata.get("blocked") or source.get("blocked"):
            alerts.append("Источник заблокирован или обрезан — проверьте оригинал.")
        error_message = source.get("error")
        if error_message:
            alerts.append(f"Ошибка источника: {str(error_message).strip()}")
        llm_error = source.get("llm_error")
        if llm_error:
            alerts.append(f"LLM-разбор не выполнен: {str(llm_error).strip()}")
        return alerts

    @staticmethod
    def _analysis_llm_notes(source: dict, *, max_items: int = 3) -> list[str]:
        payload = source.get("llm_payload")
        notes: list[str] = []
        if not isinstance(payload, str):
            return notes
        raw = payload.strip()
        if not raw:
            return notes
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        entries: Iterable[dict[str, Any]] = []
        if isinstance(parsed, dict):
            maybe_items = parsed.get("items")
            if isinstance(maybe_items, list):
                entries = maybe_items
            else:
                entries = []
        elif isinstance(parsed, list):
            entries = parsed
        else:
            entries = []
        count = 0
        for item in entries:
            if count >= max_items:
                break
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title and not snippet:
                continue
            parts = []
            if title:
                parts.append(title)
            if snippet:
                parts.append(snippet)
            note = " — ".join(parts)
            if url:
                note += f" (URL: {url})"
            notes.append(note)
            count += 1
        if notes:
            return notes
        # Fallback: use first non-empty lines from raw payload
        fallback_lines = [
            line.strip("•*- \t")
            for line in raw.splitlines()
            if line.strip()
        ]
        for line in fallback_lines:
            if len(notes) >= max_items:
                break
            notes.append(line[:220])
        return notes

    @staticmethod
    def _analysis_structured_payload(contexts: list[dict[str, Any]]) -> str | None:
        if not contexts:
            return None
        payload: list[dict[str, Any]] = []
        for ctx in contexts:
            entries = []
            for entry in ctx.get("entries") or []:
                if len(entries) >= 4:
                    break
                if not isinstance(entry, dict):
                    continue
                entries.append(
                    {
                        "title": entry.get("title"),
                        "snippet": entry.get("snippet"),
                        "url": entry.get("url"),
                    }
                )
            payload.append(
                {
                    "label": ctx.get("label"),
                    "fallback": bool(ctx.get("fallback")),
                    "alerts": ctx.get("alerts") or [],
                    "notes": ctx.get("notes") or [],
                    "entries": entries,
                }
            )
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return None

    @staticmethod
    def _analysis_messages(query: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not contexts:
            return []
        lines: list[str] = []
        for ctx in contexts:
            label = ctx.get("label") or "Источник"
            fallback = bool(ctx.get("fallback"))
            header_suffix = " [проверка вручную]" if fallback else ""
            lines.append(f"{label}{header_suffix}:")
            alerts = ctx.get("alerts") or []
            for alert in alerts:
                lines.append(f"  ! {alert}")
            entries = ctx.get("entries") or []
            if not entries:
                lines.append("  - нет результатов")
            for entry in entries:
                title = entry.get("title") or ""
                snippet = entry.get("snippet") or ""
                url = entry.get("url") or ""
                fallback_note = " [проверка вручную]" if fallback else ""
                snippet_suffix = f" — {snippet}" if snippet else ""
                url_suffix = f" (URL: {url})" if url else ""
                lines.append(f"  - {title}{snippet_suffix}{url_suffix}{fallback_note}".rstrip())
            notes = ctx.get("notes") or []
            if notes:
                for idx, note in enumerate(notes[:3], start=1):
                    lines.append(f"  * Наблюдение {idx}: {note}")
            if not entries and not notes:
                lines.append("  - сведений нет")
        user_content = (
            f"Запрос: {query or 'нет запроса'}\n\n"
            "Контекст по источникам:\n"
            + "\n".join(lines)
        )
        structured_payload = OsintSearchService._analysis_structured_payload(contexts)
        if structured_payload:
            user_content += "\n\nСтруктурированные данные (JSON):\n" + structured_payload
        system_prompt = (
            "Ты аналитик OSINT. Используй переданные данные, чтобы подготовить структурированный отчёт на русском языке.\n"
            "Формат ответа строго следующий:\n"
            "Наблюдения:\n"
            "- …\n"
            "Риски:\n"
            "- …\n"
            "Следующие шаги:\n"
            "- …\n"
            "Если раздел пуст, укажи «- нет данных». Не выдумывай факты — только из предоставленного контекста."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _analysis_fallback(query: str, contexts: list[dict[str, Any]], *, reason: str | None = None) -> str:
        lines: list[str] = []
        header_query = query.strip() if isinstance(query, str) else ""
        if header_query:
            lines.append(f"Краткий обзор по запросу «{header_query}».")
        else:
            lines.append("Краткий обзор результатов поиска.")
        if not contexts:
            lines.append("Источники не вернули релевантных результатов.")
        else:
            for ctx in contexts:
                label = ctx.get("label") or "Источник"
                entries = ctx.get("entries") or []
                if not entries:
                    detail = "нет результатов."
                else:
                    top = entries[0]
                    title = top.get("title") or "результат"
                    snippet = top.get("snippet") or ""
                    url = top.get("url") or ""
                    parts = [title]
                    if snippet:
                        parts.append(snippet)
                    detail = " — ".join(part for part in parts if part)
                    if url:
                        detail += f" ({url})"
                    if ctx.get("fallback"):
                        detail += " [проверка вручную]"
                lines.append(f"- {label}: {detail}")
                alerts = ctx.get("alerts") or []
                for alert in alerts:
                    lines.append(f"  ! {alert}")
                notes = ctx.get("notes") or []
                for note in notes[:2]:
                    lines.append(f"  * Дополнение: {note}")
        if reason:
            lines.append(f"⚠️ Итог готов без LLM: {reason}.")
        lines.append("Изучите карточки источников для подробностей.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _json_clone(payload: Any) -> Any:
        try:
            return json.loads(json.dumps(payload, ensure_ascii=False))
        except Exception:
            return payload

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            dt = value
        elif value is None:
            return None
        else:
            text = str(value).strip()
            if not text:
                return None
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.replace(tzinfo=None)

    @staticmethod
    def _sanitize_schedule_config(config: Any) -> dict | None:
        if not isinstance(config, dict):
            return None
        cleaned: dict[str, Any] = {}
        if "id" in config:
            try:
                cleaned["id"] = int(config["id"])
            except (TypeError, ValueError):
                pass
        if "active" in config:
            cleaned["active"] = bool(config.get("active"))
        else:
            cleaned["active"] = True
        if "interval_minutes" in config and config["interval_minutes"] not in (None, ""):
            try:
                minutes = int(config["interval_minutes"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Некорректный интервал расписания") from exc
            if minutes <= 0:
                raise ValueError("Интервал расписания должен быть больше нуля")
            cleaned["interval_minutes"] = max(5, minutes)
        start_at = OsintSearchService._parse_iso_datetime(config.get("start_at"))
        if start_at:
            cleaned["start_at"] = start_at
        label = str(config.get("label") or "").strip()
        if label:
            cleaned["label"] = label[:120]
        if "notify" in config:
            cleaned["notify"] = bool(config.get("notify"))
        notify_channel = str(config.get("notify_channel") or "").strip()
        if notify_channel:
            cleaned["notify_channel"] = notify_channel[:120]
        if cleaned.get("active", True) and "interval_minutes" not in cleaned and cleaned.get("id") is None:
            raise ValueError("Не указан интервал расписания")
        return cleaned

    @staticmethod
    def _build_schedule_template(
        *,
        query: str,
        locale: str,
        region: str | None,
        safe: bool,
        sources: list[dict],
        params: dict | None,
        user_id: int | None,
    ) -> dict:
        base_params = dict(params or {})
        base_params.pop("schedule", None)
        return {
            "query": query,
            "locale": locale,
            "region": region,
            "safe": bool(safe),
            "sources": OsintSearchService._json_clone(sources),
            "params": OsintSearchService._json_clone(base_params),
            "user_id": user_id,
        }

    def _configure_schedule(
        self,
        job_snapshot: dict,
        schedule_raw: Any,
        *,
        template: dict,
        sanitized: dict | None = None,
    ) -> None:
        if schedule_raw is None and sanitized is None:
            return
        schedule_config = sanitized or self._sanitize_schedule_config(schedule_raw)
        if schedule_config is None:
            return
        is_active = schedule_config.get("active", True)
        schedule_id = schedule_config.get("id")
        job_id = int(job_snapshot.get("id"))
        if not is_active:
            self.repository.disable_schedule(schedule_id=schedule_id, job_id=job_id)
            job_snapshot["schedule"] = None
            params = job_snapshot.get("params")
            if isinstance(params, dict):
                params.pop("schedule", None)
            return
        interval = schedule_config.get("interval_minutes")
        if interval is None:
            raise ValueError("Не указан интервал расписания")
        schedule_record = self.repository.upsert_schedule(
            job_id=job_id,
            schedule_id=schedule_id,
            template=template,
            interval_minutes=interval,
            start_at=schedule_config.get("start_at"),
            notify=schedule_config.get("notify", False),
            notify_channel=schedule_config.get("notify_channel"),
            label=schedule_config.get("label"),
        )
        job_snapshot["schedule"] = schedule_record
        params = job_snapshot.get("params")
        if isinstance(params, dict):
            params_schedule = params.setdefault("schedule", {})
            params_schedule.update(
                {
                    "id": schedule_record.get("id"),
                    "interval_minutes": schedule_record.get("interval_minutes"),
                    "notify": schedule_record.get("notify"),
                    "label": schedule_record.get("label"),
                }
            )
        self._schedule_event.set()
        self.ensure_schedule_worker()

    def update_schedule(self, job_id: int, schedule_raw: Any) -> dict:
        schedule_config = self._sanitize_schedule_config(schedule_raw)
        if schedule_config is None:
            raise ValueError("Некорректные параметры расписания")
        job_snapshot = self.repository.get_job(job_id)
        if not job_snapshot:
            raise ValueError("osint_job_not_found")
        template = self._build_schedule_template(
            query=str(job_snapshot.get("query") or ""),
            locale=str(job_snapshot.get("locale") or "ru-RU"),
            region=job_snapshot.get("region"),
            safe=bool(job_snapshot.get("safe")),
            sources=job_snapshot.get("source_specs") or [],
            params=job_snapshot.get("params") or {},
            user_id=job_snapshot.get("user_id"),
        )
        if not schedule_config.get("active", True):
            self.repository.disable_schedule(schedule_id=schedule_config.get("id"), job_id=job_id)
            updated = self.repository.get_job(job_id)
            return self._enrich_job(updated)
        interval = schedule_config.get("interval_minutes")
        if interval is None:
            raise ValueError("Не указан интервал расписания")
        schedule_record = self.repository.upsert_schedule(
            job_id=job_id,
            schedule_id=schedule_config.get("id"),
            template=template,
            interval_minutes=interval,
            start_at=schedule_config.get("start_at"),
            notify=schedule_config.get("notify", False),
            notify_channel=schedule_config.get("notify_channel"),
            label=schedule_config.get("label"),
        )
        updated = self.repository.get_job(job_id)
        if updated is not None:
            updated["schedule"] = schedule_record
        self._schedule_event.set()
        self.ensure_schedule_worker()
        return self._enrich_job(updated)

    def _refine_query(self, raw_query: str) -> tuple[str, list[str]]:
        query = (raw_query or "").strip()
        if not query:
            return "", []
        try:
            from app import (
                STOP_WORDS,
                _extract_quoted_phrases,
                _normalize_keyword_candidate,
                _tokenize_query,
                call_lmstudio_keywords,
            )
            stop_words = STOP_WORDS
        except Exception:
            stop_words = _OSINT_STOP_WORDS
            call_lmstudio_keywords = None

            def _normalize_keyword_candidate(value: str) -> str:
                cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" \"'«»“”[]{}")
                return cleaned.lower().strip()

            def _extract_quoted_phrases(_: str) -> list[str]:  # type: ignore
                return []

            def _tokenize_query(text: str) -> list[str]:  # type: ignore
                parts = re.split(r"[^\w\-]+", text.lower())
                return [p for p in parts if p]
        else:
            stop_words = STOP_WORDS

        phrases: list[str] = []
        try:
            phrases = _extract_quoted_phrases(query)
        except Exception:
            phrases = []

        keyword_candidates: list[str] = []
        if call_lmstudio_keywords is not None:
            try:
                keyword_candidates = call_lmstudio_keywords(query, "ai-search") or []
            except Exception:
                keyword_candidates = []
        try:
            baseline_tokens = _tokenize_query(query)
        except Exception:
            baseline_tokens = []
        if not keyword_candidates:
            keyword_candidates = baseline_tokens
            baseline_tokens = []

        seen: set[str] = set()
        normalized_terms: list[str] = []

        def _add_term(term: str, *, prepend: bool = False) -> None:
            normalized = _normalize_keyword_candidate(term)
            if not normalized:
                return
            if normalized in stop_words:
                return
            if normalized in seen:
                return
            seen.add(normalized)
            if prepend:
                normalized_terms.insert(0, normalized)
            else:
                normalized_terms.append(normalized)

        for phrase in phrases:
            _add_term(phrase, prepend=True)
        for candidate in keyword_candidates:
            _add_term(candidate)
        for fallback_term in baseline_tokens:
            _add_term(fallback_term)

        if not normalized_terms:
            normalized_terms = [query]

        formatted_terms: list[str] = []
        for term in normalized_terms:
            formatted_terms.append(f'"{term}"' if " " in term and not term.startswith('"') else term)

        refined = " ".join(formatted_terms).strip()
        return refined, normalized_terms

    def ensure_schedule_worker(self) -> None:
        with self._schedule_lock:
            if self._schedule_thread_started:
                return
            try:
                app = current_app._get_current_object()
            except RuntimeError:
                self.logger.warning("OSINT scheduler требуется вызвать из контекста приложения")
                return
            thread = threading.Thread(
                target=self._schedule_loop,
                args=(app,),
                name="osint-schedule-worker",
                daemon=True,
            )
            thread.start()
            self._schedule_thread = thread
            self._schedule_thread_started = True

    def _schedule_loop(self, app) -> None:
        self.logger.info("OSINT schedule worker started")
        idle_timeout = 60.0
        while True:
            try:
                with app.app_context():
                    claimed = self.repository.claim_next_schedule()
                if not claimed:
                    self._schedule_event.wait(timeout=idle_timeout)
                    self._schedule_event.clear()
                    continue
                with app.app_context():
                    self._execute_scheduled_job(claimed)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("OSINT schedule loop error: %s", exc)
                time.sleep(30)

    def _execute_scheduled_job(self, schedule: dict) -> None:
        schedule_id = schedule.get("id")
        template = schedule.get("template") or {}
        interval = schedule.get("interval_minutes")
        params = self._json_clone(template.get("params") or {})
        params.setdefault("schedule", {})
        params["schedule"].update(
            {
                "id": schedule_id,
                "interval_minutes": interval,
                "label": schedule.get("label") or template.get("label"),
                "notify": schedule.get("notify"),
            }
        )
        try:
            job = self.start_job(
                query=str(template.get("query") or ""),
                locale=str(template.get("locale") or "ru-RU"),
                region=template.get("region"),
                safe=bool(template.get("safe")),
                sources=self._json_clone(template.get("sources") or []),
                params=params,
                user_id=template.get("user_id"),
            )
            self.logger.info(
                "OSINT scheduled job #%s launched from schedule #%s",
                job.get("id"),
                schedule_id,
            )
        except Exception as exc:  # noqa: BLE001
            interval_fallback = max(15, int(interval or 15))
            self.repository.mark_schedule_failure(
                schedule_id,
                delay_minutes=interval_fallback,
                error=str(exc),
            )
            self._schedule_event.set()
            self.logger.exception("OSINT scheduled job failed (schedule=%s): %s", schedule_id, exc)

    # ------------------------------------------------------------------
    # Job orchestration
    # ------------------------------------------------------------------
    def start_job(
        self,
        *,
        query: str,
        locale: str,
        region: str | None,
        safe: bool,
        sources: list[dict],
        params: dict | None = None,
        user_id: int | None = None,
    ) -> dict:
        """Create a multi-source search job and enqueue tasks."""
        if not sources:
            raise ValueError("sources_required")
        prepared_sources = self._prepare_sources(sources)
        schedule_raw = None
        schedule_config: dict[str, Any] | None = None
        if isinstance(params, dict):
            schedule_raw = params.get("schedule")
            if schedule_raw:
                schedule_config = self._sanitize_schedule_config(schedule_raw)
        refined_query, keyword_terms = self._refine_query(query)
        search_query = refined_query or query
        base_params_payload = params.copy() if isinstance(params, dict) else {}
        base_params_payload.setdefault("refined_query", search_query)
        if keyword_terms:
            base_params_payload.setdefault("keywords", keyword_terms)
        requested_locales = self._normalize_requested_locales(base_params_payload.get("locales"), locale)
        base_params_payload["locales"] = requested_locales
        primary_locale = requested_locales[0]
        job_sources = self._expand_sources_for_locales(prepared_sources, requested_locales)
        job = self.repository.create_job(
            query=query,
            locale=primary_locale,
            region=region,
            safe=safe,
            sources=job_sources,
            params=base_params_payload,
            user_id=user_id,
        )
        template = self._build_schedule_template(
            query=query,
            locale=primary_locale,
            region=region,
            safe=safe,
            sources=job_sources,
            params=base_params_payload,
            user_id=user_id,
        )
        app = current_app._get_current_object()
        for spec in job_sources:
            self.queue.submit(
                self._execute_source_task,
                app,
                job["id"],
                {
                    "query": search_query,
                    "original_query": query,
                    "keywords": keyword_terms,
                    "locale": spec.get("locale") or primary_locale,
                    "region": region,
                    "safe": safe,
                    "params": params or {},
                },
                spec,
                description=f"osint:{spec['id']}",
            )
        if schedule_raw or schedule_config:
            self._configure_schedule(job, schedule_raw, template=template, sanitized=schedule_config)
        return self._enrich_job(job)

    def get_job(self, job_id: int) -> dict:
        job = self.repository.get_job(job_id)
        return self._enrich_job(job)

    def list_jobs(self, limit: int = 10) -> list[dict]:
        return self.repository.list_jobs(limit)

    def delete_job(self, job_id: int) -> bool:
        return self.repository.delete_job(job_id)

    # ------------------------------------------------------------------
    # Legacy single-source execution (kept for backwards compatibility)
    # ------------------------------------------------------------------
    def run(self, request: SerpRequest) -> dict:
        """Synchronous execution for single-engine requests."""
        self.logger.debug("Running legacy single-source OSINT search (%s)", request.engine)
        serp_result = self.fetcher.fetch(request)
        payload = self.parser.parse(serp_result)
        job = self.repository.create_job(
            query=request.query,
            locale=request.locale,
            region=request.region,
            safe=request.safe,
            sources=[{"type": "engine", "engine": request.engine}],
            params={"max_results": request.max_results, "force_refresh": request.force_refresh},
        )
        job_id = job["id"]
        self.repository.mark_source_status(job_id, request.engine, status="running")
        snapshot = self.repository.persist_source_result(
            job_id=job_id,
            source_id=request.engine,
            engine=request.engine,
            blocked=serp_result.blocked,
            from_cache=bool(serp_result.metadata.get("from_cache")),
            html_snapshot=serp_result.html,
            text_content=serp_result.text,
            screenshot_path=serp_result.screenshot_path,
            llm_payload=payload.raw_response,
            llm_model=payload.llm_model,
            llm_error=payload.llm_error,
            status="completed",
            error=None,
            metadata=serp_result.metadata,
            results=[
                {
                    "rank": item.rank,
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "metadata": item.metadata,
                    "score": None,
                }
                for item in payload.items
            ],
        )
        self.repository.update_job_status(job_id, status="completed", completed=True)
        updated = self.repository.get_job(job_id)
        self._maybe_trigger_analysis(updated)
        self._maybe_trigger_ontology(updated)
        return self._enrich_job(updated)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_sources(raw_sources: Iterable[dict]) -> list[dict]:
        prepared: list[dict] = []
        for idx, src in enumerate(raw_sources):
            entry = dict(src or {})
            source_id = str(entry.get("id") or entry.get("engine") or entry.get("type") or f"src{idx}")
            entry["id"] = source_id
            entry.setdefault("label", entry.get("label") or source_id.capitalize())
            if entry.get("type") == "local":
                options = entry.get("options") or {}
                exclude_raw = options.get("exclude_patterns")
                exclude_patterns: tuple[str, ...] = ()
                if isinstance(exclude_raw, str):
                    parts = [part.strip() for part in exclude_raw.split(",") if part and part.strip()]
                    exclude_patterns = tuple(parts)
                elif isinstance(exclude_raw, (list, tuple)):
                    parts = [str(part).strip() for part in exclude_raw if str(part).strip()]
                    exclude_patterns = tuple(parts)
                entry["options"] = {
                    "mode": options.get("mode", "catalog"),
                    "path": options.get("path"),
                    "limit": max(1, min(int(options.get("limit", 20) or 20), 200)),
                    "recursive": bool(options.get("recursive", True)),
                    "ocr": bool(options.get("ocr", False)),
                    "exclude_patterns": exclude_patterns,
                }
            else:
                entry["type"] = "engine"
                entry["engine"] = str(entry.get("engine") or entry.get("id")).lower()
            prepared.append(entry)
        return prepared

    @staticmethod
    def _normalize_requested_locales(value: Any, primary_locale: str) -> list[str]:
        candidates: list[str] = []
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, Sequence):
            candidates.extend(str(item) for item in value)
        cleaned: list[str] = []
        seen: set[str] = set()
        primary = primary_locale if primary_locale in SUPPORTED_OSINT_LOCALES else SUPPORTED_OSINT_LOCALES[0]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text or text not in SUPPORTED_OSINT_LOCALES or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        if not cleaned:
            return [primary]
        filtered = [item for item in cleaned if item != primary]
        return [primary] + filtered

    def _expand_sources_for_locales(self, sources: list[dict], locales: list[str]) -> list[dict]:
        if not locales:
            return [copy.deepcopy(spec) for spec in sources]
        expanded: list[dict] = []
        for idx, spec in enumerate(sources):
            if spec.get("locale_expanded"):
                expanded.append(copy.deepcopy(spec))
                continue
            spec_locale = locales[0]
            if spec.get("type") != "engine":
                clone = copy.deepcopy(spec)
                clone.setdefault("locale", spec_locale)
                clone["locale_expanded"] = True
                expanded.append(clone)
                continue
            base_id = str(spec.get("id") or spec.get("engine") or spec.get("source") or f"src{idx}")
            base_label = str(spec.get("label") or base_id).strip()
            for locale in locales:
                clone = copy.deepcopy(spec)
                clone["locale"] = locale
                clone_id = f"{base_id}:{locale}"
                clone["id"] = clone_id
                clone["source"] = clone_id
                clone["label"] = f"{base_label} ({locale})" if base_label else locale
                clone["locale_expanded"] = True
                expanded.append(clone)
        return expanded

    def _execute_source_task(
        self,
        app,
        job_id: int,
        base_params: dict,
        source_spec: dict,
    ) -> None:
        with app.app_context():
            source_id = str(source_spec.get("id"))
            self.repository.update_job_status(job_id, started=True)
            self.repository.mark_source_status(job_id, source_id, status="running")
            try:
                keywords = base_params.get("keywords") or []
                if source_spec.get("type") == "local":
                    result = self._run_local_source(
                        query=base_params["query"],
                        keywords=keywords if isinstance(keywords, list) else [],
                        locale=base_params["locale"],
                        source_spec=source_spec,
                    )
                    snapshot = self.repository.persist_source_result(
                        job_id=job_id,
                        source_id=source_id,
                        engine=source_spec.get("engine") or "local",
                        blocked=False,
                        from_cache=result.get("from_cache", False),
                        html_snapshot=None,
                        text_content=None,
                        screenshot_path=None,
                        llm_payload=None,
                        llm_model=None,
                        llm_error=None,
                        status="completed",
                        error=None,
                        metadata=result.get("metadata"),
                        results=result.get("items", []),
                    )
                    self._maybe_trigger_analysis(snapshot)
                    self._maybe_trigger_ontology(snapshot)
                else:
                    self._run_engine_source(
                        job_id=job_id,
                        source_id=source_id,
                        base_params=base_params,
                        source_spec=source_spec,
                    )
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                self.logger.exception("OSINT source %s failed: %s", source_id, exc)
                self.repository.record_source_error(job_id=job_id, source_id=source_id, message=message)

    def _run_engine_source(
        self,
        *,
        job_id: int,
        source_id: str,
        base_params: dict,
        source_spec: dict,
    ) -> None:
        engine = str(source_spec.get("engine") or source_id).lower()
        max_results = source_spec.get("max_results")
        force_refresh = source_spec.get("force_refresh")
        extra_params = source_spec.get("extra_params")
        raw_keywords = base_params.get("keywords") or []
        keywords = [
            str(term).strip()
            for term in raw_keywords
            if isinstance(term, str) and str(term).strip()
        ]
        query_for_terms = str(base_params.get("original_query") or base_params.get("query") or "")
        if not self._network_available():
            self.repository.mark_source_status(
                job_id,
                source_id,
                status="blocked",
                error="нет интернета - нет ответа",
                extra={
                    "fallback": True,
                    "fallback_url": None,
                    "from_fetcher": False,
                },
            )
            self.repository.record_source_error(
                job_id=job_id,
                source_id=source_id,
                message="нет интернета - нет ответа",
            )
            return
        retry_flag = bool(base_params.get("retry"))
        user_agent_override = base_params.get("user_agent_override")
        proxy_override = base_params.get("proxy_override")
        request = SerpRequest(
            query=base_params["query"],
            engine=engine,  # type: ignore[arg-type]
            locale=base_params["locale"],
            region=base_params.get("region"),
            safe=bool(base_params.get("safe")),
            max_results=max_results if isinstance(max_results, int) else None,
            force_refresh=True if retry_flag else bool(force_refresh),
            extra_params=extra_params if isinstance(extra_params, dict) else None,
            retry=retry_flag,
            user_agent_override=str(user_agent_override).strip() or None if isinstance(user_agent_override, str) else None,
            proxy_override=str(proxy_override).strip() or None if isinstance(proxy_override, str) else None,
        )
        serp_result = self.fetcher.fetch(request)
        payload = self.parser.parse(serp_result)
        metadata = dict(serp_result.metadata or {})
        metadata.setdefault("label", source_spec.get("label") or source_id)
        metadata.setdefault("requested_url", serp_result.requested_url)
        metadata.setdefault("final_url", serp_result.final_url)
        terms = [term.lower() for term in keywords]
        if not terms:
            terms = self._filesystem_terms(query_for_terms.lower())
        result_payload: list[dict[str, Any]] = []
        for item in payload.items:
            highlight = self._highlight_text(item.snippet, terms)
            item_metadata = dict(item.metadata or {})
            if highlight:
                item_metadata.setdefault("highlight", highlight)
            structured = self._extract_structured(item.snippet, item.url)
            if structured:
                item_metadata.setdefault("extracted", structured)
            result_payload.append(
                {
                    "rank": item.rank,
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "highlight": highlight,
                    "metadata": item_metadata,
                    "score": None,
                }
            )
        if serp_result.text:
            doc_structured = self._extract_structured(serp_result.text)
            if doc_structured:
                metadata.setdefault("extracted", doc_structured)
        requested_results = max_results if isinstance(max_results, int) and max_results > 0 else None
        results_captured = len(payload.items)
        results_forwarded = len(result_payload)
        estimated_pages = max(
            1,
            math.ceil(max(results_captured, results_forwarded or 1) / _SERP_RESULTS_PER_PAGE),
        )
        metadata["requested_results"] = requested_results or self.parser.max_items
        metadata["results_parsed"] = results_captured
        metadata["results_forwarded"] = results_forwarded
        metadata["results_on_page"] = len(result_payload)
        metadata["pages_estimated"] = estimated_pages
        metadata["refined_query"] = base_params.get("query")
        metadata["original_query"] = base_params.get("original_query")
        if keywords:
            metadata["keywords"] = keywords
        snapshot = self.repository.persist_source_result(
            job_id=job_id,
            source_id=source_id,
            engine=engine,
            blocked=serp_result.blocked,
            from_cache=bool(serp_result.metadata.get("from_cache")),
            html_snapshot=serp_result.html,
            text_content=serp_result.text,
            screenshot_path=serp_result.screenshot_path,
            llm_payload=payload.raw_response,
            llm_model=payload.llm_model,
            llm_error=payload.llm_error,
            status="completed",
            error=None,
            metadata=metadata,
            results=result_payload,
            requested_url=serp_result.requested_url,
            final_url=serp_result.final_url,
        )
        self._maybe_trigger_analysis(snapshot)
        self._maybe_trigger_ontology(snapshot)

    def _run_local_source(
        self,
        *,
        query: str,
        keywords: list[str],
        locale: str,
        source_spec: dict,
    ) -> dict:
        options_raw = source_spec.get("options") or {}
        options = LocalSourceOptions(
            mode=str(options_raw.get("mode") or "catalog"),
            path=str(options_raw.get("path") or "").strip() or None,
            limit=max(1, min(int(options_raw.get("limit") or 20), 200)),
            recursive=bool(options_raw.get("recursive", True)),
            ocr=bool(options_raw.get("ocr", False)),
            exclude_patterns=tuple(options_raw.get("exclude_patterns") or ()),
        )
        label = source_spec.get("label") or source_spec.get("id")
        if options.mode == "filesystem":
            return self._local_filesystem_search(query=query, keywords=keywords, options=options, label=label)
        return self._local_catalog_search(query=query, locale=locale, keywords=keywords, options=options, label=label)

    def _local_catalog_search(
        self,
        *,
        query: str,
        locale: str,
        keywords: list[str],
        options: LocalSourceOptions,
        label: str | None,
    ) -> dict:
        svc = SearchService(
            db=db,
            file_model=File,
            tag_model=Tag,
        )
        ids = svc.candidate_ids(query, limit=options.limit)
        items = []
        terms = [str(term).lower() for term in keywords if isinstance(term, str) and term]
        if not terms:
            terms = self._filesystem_terms(query.lower())
        if ids:
            q = File.query.filter(File.id.in_(ids))
            rows = q.order_by(File.id.desc()).limit(options.limit).all()
            for idx, file_obj in enumerate(rows, start=1):
                title = file_obj.title or file_obj.filename
                snippet = file_obj.text_excerpt or file_obj.abstract or ""
                highlight = self._highlight_text(snippet, terms)
                metadata = {
                    "filename": file_obj.filename,
                    "collection_id": file_obj.collection_id,
                    "collection": getattr(file_obj.collection, "name", None),
                    "size": file_obj.size,
                }
                if highlight:
                    metadata["highlight"] = highlight
                structured = self._extract_structured(snippet, url=f"/files/{file_obj.id}")
                if structured:
                    metadata["extracted"] = structured
                items.append(
                    {
                        "rank": idx,
                        "title": title or file_obj.filename,
                        "url": f"/files/{file_obj.id}",
                        "snippet": snippet,
                        "highlight": highlight,
                        "metadata": metadata,
                    }
                )
        normalized_keywords = [
            str(term).strip()
            for term in keywords
            if isinstance(term, str) and str(term).strip()
        ]
        metadata_payload: dict[str, Any] = {
            "source": "catalog",
            "locale": locale,
            "label": label,
            "results_parsed": len(items),
            "results_forwarded": len(items),
            "results_on_page": len(items),
            "pages_estimated": 1,
            "refined_query": query,
        }
        if normalized_keywords:
            metadata_payload["keywords"] = normalized_keywords
        elif terms:
            metadata_payload["keywords"] = terms
        return {
            "from_cache": False,
            "metadata": metadata_payload,
            "items": items[: options.limit],
        }

    def _local_filesystem_search(
        self,
        *,
        query: str,
        keywords: list[str],
        options: LocalSourceOptions,
        label: str | None,
    ) -> dict:
        path = options.path
        if not path:
            raise ValueError("filesystem_path_required")
        expanded = Path(os.path.expandvars(path)).expanduser()
        candidates = [expanded]
        if not expanded.is_absolute():
            candidates.append(Path.cwd() / expanded)
            candidates.append(Path.home() / path)
            try:
                scan_root = current_app.config.get('SCAN_ROOT')
            except RuntimeError:
                scan_root = None
            if scan_root:
                candidates.append(Path(scan_root).expanduser() / path)
        base: Path | None = None
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved.exists():
                base = resolved
                break
        if base is None or not base.exists():
            raise FileNotFoundError(f"Путь не найден: {expanded}")
        assert base is not None
        normalized_query = query.lower()
        terms = [str(term).lower() for term in keywords if isinstance(term, str) and term]
        if not terms:
            terms = self._filesystem_terms(normalized_query)
        results = []
        iterator: Iterable[Path]
        if base.is_file():
            iterator = [base]
        elif options.recursive:
            iterator = base.rglob("*")
        else:
            iterator = base.glob("*")
        excluded_patterns = options.exclude_patterns or ()
        for candidate in iterator:
            if len(results) >= options.limit:
                break
            try:
                if not candidate.is_file():
                    continue
                path_str = str(candidate)
                if excluded_patterns and any(
                    fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(candidate.name, pattern)
                    for pattern in excluded_patterns
                ):
                    continue
                stat = candidate.stat()
                name_lower = candidate.name.lower()
                match_kind: str | None
                snippet: str | None = None
                highlight: str | None = None
                if not terms:
                    match_kind = "any"
                    snippet = str(candidate.parent)
                elif all(term in name_lower for term in terms):
                    match_kind = "name"
                    snippet = candidate.name
                else:
                    snippet = self._filesystem_scan_file(
                        candidate,
                        terms,
                        stat.st_size,
                        max_bytes=500_000,
                        force_ocr=options.ocr,
                    )
                    match_kind = "content" if snippet else None
                if match_kind is None:
                    continue
                if snippet and not self._filesystem_is_relevant(snippet, terms):
                    continue
                if not snippet:
                    snippet = str(candidate.parent)
                if snippet:
                    highlight = self._highlight_text(snippet, terms)
                metadata = {
                    "path": str(candidate),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "match": match_kind,
                }
                if highlight:
                    metadata["highlight"] = highlight
                structured = self._extract_structured(snippet, url=candidate.as_uri()) if snippet else {}
                if structured:
                    metadata["extracted"] = structured
                results.append(
                    {
                        "rank": len(results) + 1,
                        "title": candidate.name,
                        "url": candidate.as_uri(),
                        "snippet": snippet,
                        "highlight": highlight,
                        "metadata": metadata,
                    }
                )
            except Exception:
                continue
        normalized_keywords = [
            str(term).strip()
            for term in keywords
            if isinstance(term, str) and str(term).strip()
        ]
        payload_meta: dict[str, Any] = {
            "source": "filesystem",
            "path": str(base),
            "recursive": options.recursive,
            "label": label,
            "results_parsed": len(results),
            "results_forwarded": len(results),
            "results_on_page": len(results),
            "pages_estimated": 1,
            "refined_query": query,
        }
        if normalized_keywords:
            payload_meta["keywords"] = normalized_keywords
        elif terms:
            payload_meta["keywords"] = terms
        if options.ocr:
            payload_meta["ocr"] = True
        if excluded_patterns:
            payload_meta["exclude_patterns"] = list(excluded_patterns)
        return {
            "from_cache": False,
            "metadata": payload_meta,
            "items": results,
        }

    def retry_source(
        self,
        job_id: int,
        source_id: str,
        *,
        force_refresh: bool = True,
    ) -> dict:
        job_snapshot = self.repository.get_job(job_id)
        if not job_snapshot:
            raise ValueError("osint_job_not_found")
        specs = job_snapshot.get("source_specs") or []
        selected_spec: dict | None = None
        for candidate in specs:
            candidate_id = str(candidate.get("id") or candidate.get("source") or candidate.get("engine") or "")
            if candidate_id == source_id:
                selected_spec = dict(candidate)
                break
        if selected_spec is None:
            raise ValueError("osint_source_not_found")
        params = job_snapshot.get("params") or {}
        refined_query = params.get("refined_query") or job_snapshot.get("query") or ""
        spec_locale = str(
            selected_spec.get("locale") or job_snapshot.get("locale") or "ru-RU"
        )
        keywords = params.get("keywords")
        if not isinstance(keywords, list):
            keywords = []
        base_params = {
            "query": refined_query,
            "original_query": job_snapshot.get("query") or "",
            "keywords": keywords,
            "locale": spec_locale,
            "region": job_snapshot.get("region"),
            "safe": bool(job_snapshot.get("safe")),
            "params": params,
        }
        source_type = selected_spec.get("type") or ("engine" if selected_spec.get("engine") else "local")
        base_params["retry"] = True
        if source_type == "engine":
            ua_override, proxy_override = self.fetcher.choose_retry_overrides()
            if ua_override:
                base_params["user_agent_override"] = ua_override
            if proxy_override:
                base_params["proxy_override"] = proxy_override
        if source_type == "engine" and force_refresh:
            selected_spec["force_refresh"] = True
        self.repository.mark_source_status(
            job_id,
            source_id,
            status="queued",
            error=None,
            extra={"retry_requested": True},
        )
        self.repository.update_job_status(job_id, started=True)
        app = current_app._get_current_object()
        self.queue.submit(
            self._execute_source_task,
            app,
            job_id,
            base_params,
            selected_spec,
            description=f"osint-retry:{source_id}",
        )
        return self.repository.get_job(job_id)

    @staticmethod
    def _filesystem_is_relevant(snippet: str | None, terms: list[str], *, min_hits: int = 1) -> bool:
        if not snippet:
            return False
        if not terms:
            return True
        snippet_lower = snippet.lower()
        hits = sum(1 for term in terms if term and term.lower() in snippet_lower)
        return hits >= min_hits

    def _maybe_trigger_analysis(self, snapshot: dict) -> None:
        if not snapshot or snapshot.get("status") != "completed":
            return
        if snapshot.get("analysis") or snapshot.get("analysis_error"):
            return
        try:
            self._run_llm_analysis(snapshot)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("OSINT analysis failed: %s", exc)
            try:
                self.repository.set_job_analysis(int(snapshot.get("id")), None, str(exc))
            except Exception:
                pass

    def _run_llm_analysis(self, job_snapshot: dict) -> None:
        job_id = int(job_snapshot.get("id"))
        query = str(job_snapshot.get("query") or "")
        contexts = self._analysis_context(job_snapshot)
        if not contexts:
            fallback = self._analysis_fallback(
                query,
                contexts,
                reason="источники не вернули результатов",
            )
            self.repository.set_job_analysis(job_id, fallback, "analysis_context_missing")
            return

        messages = self._analysis_messages(query, contexts)
        summary: str | None = None
        last_error: Exception | None = None

        if messages:
            from app import (  # type: ignore
                _llm_extract_content,
                _llm_iter_choices,
                _llm_response_indicates_busy,
                _llm_send_chat,
            )

            for choice in _llm_iter_choices("osint-analysis"):
                try:
                    provider, response = _llm_send_chat(
                        choice,
                        messages,
                        temperature=0.2,
                        max_tokens=600,
                        timeout=180,
                        cache_bucket="osint-analysis",
                    )
                    if _llm_response_indicates_busy(response):
                        last_error = RuntimeError("llm_busy")
                        continue
                    response.raise_for_status()
                    data = response.json()
                    content = _llm_extract_content(provider, data)
                    if content:
                        summary = content.strip()
                        break
                    last_error = RuntimeError("empty_llm_response")
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue

        if summary:
            self.repository.set_job_analysis(job_id, summary, None)
            return

        if last_error:
            self.logger.info("OSINT LLM analysis fallback: %s", last_error)
        fallback = self._analysis_fallback(
            query,
            contexts,
            reason="LLM недоступен, использован эвристический обзор" if last_error else None,
        )
        error_text = str(last_error) if last_error else None
        self.repository.set_job_analysis(job_id, fallback, error_text)

    # ------------------------------------------------------------------
    # Ontology graph helpers
    # ------------------------------------------------------------------
    def _maybe_trigger_ontology(self, snapshot: dict) -> None:
        if not snapshot:
            return
        params = snapshot.get("params") or {}
        if not isinstance(params, dict):
            return
        build_requested = bool(params.get("build_ontology") or params.get("ontology_enabled"))
        if not build_requested:
            return
        status = str(snapshot.get("status") or "").lower()
        if status not in {"running", "completed"}:
            return
        combined = snapshot.get("combined_results")
        if combined is None:
            combined = self._combine_results(snapshot)
            snapshot["combined_results"] = combined
        if not combined and status != "completed":
            return
        signature = self._ontology_signature(combined)
        existing_graph = snapshot.get("ontology") if isinstance(snapshot.get("ontology"), dict) else None
        previous_signature = ""
        if existing_graph and isinstance(existing_graph.get("meta"), dict):
            previous_signature = str(existing_graph["meta"].get("signature") or "")
        existing_method = ""
        existing_meta = existing_graph.get("meta") if isinstance(existing_graph, dict) else {}
        if isinstance(existing_meta, dict):
            existing_method = str(existing_meta.get("method") or "").lower()

        processed_items = set()
        if isinstance(existing_meta, dict):
            processed_raw = existing_meta.get("processed_items")
            if isinstance(processed_raw, list):
                processed_items = {str(item) for item in processed_raw}

        pending_items = [
            item for item in (combined or [])
            if self._ontology_item_key(item) not in processed_items
        ]

        if (
            signature
            and previous_signature
            and signature == previous_signature
            and not snapshot.get("ontology_error")
            and existing_method == "llm"
            and not pending_items
        ):
            return
        chunk_size = max(1, min(int(params.get("ontology_chunk_size") or 6), 12))
        if not pending_items:
            pending_items = (combined or [])[:chunk_size]
        else:
            pending_items = pending_items[:chunk_size]
        self.logger.info(
            "OSINT ontology chunk queued job=%s signature=%s size=%s processed=%s total=%s",
            snapshot.get("id"),
            signature,
            len(pending_items),
            len(processed_items),
            len(combined or []),
        )
        try:
            self._run_llm_ontology(
                snapshot,
                signature=signature,
                previous_graph=existing_graph,
                chunk=pending_items,
                processed=processed_items,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("OSINT ontology failed: %s", exc)
            try:
                job_id = int(snapshot.get("id"))
                placeholder = {
                    "nodes": [],
                    "edges": [],
                    "meta": {
                        "method": "error",
                        "signature": signature,
                        "updated_at": datetime.utcnow().isoformat(),
                    },
                }
                self.repository.set_job_ontology(job_id, placeholder, str(exc))
            except Exception:
                pass

    def _run_llm_ontology(
        self,
        job_snapshot: dict,
        *,
        signature: str | None,
        previous_graph: dict | None,
        chunk: list[dict[str, Any]],
        processed: set[str],
    ) -> None:
        job_id = int(job_snapshot.get("id"))
        query = str(job_snapshot.get("query") or "")
        combined = job_snapshot.get("combined_results")
        if combined is None:
            combined = self._combine_results(job_snapshot)
            job_snapshot["combined_results"] = combined
        if not combined:
            payload = {"nodes": [], "edges": [], "meta": {"method": "empty"}}
            meta = payload.setdefault("meta", {})
            meta["signature"] = signature or "empty"
            meta["updated_at"] = datetime.utcnow().isoformat()
            meta["sources_seen"] = []
            meta["result_count"] = 0
            self.repository.set_job_ontology(job_id, payload, None)
            return

        messages = self._ontology_messages(query, chunk, previous_graph, total=len(combined))
        graph: dict | None = None
        last_error: Exception | None = None

        if messages:
            from app import (  # type: ignore
                _llm_extract_content,
                _llm_iter_choices,
                _llm_response_indicates_busy,
                _llm_send_chat,
            )

            for choice in _llm_iter_choices("osint-ontology"):
                try:
                    provider, response = _llm_send_chat(
                        choice,
                        messages,
                        temperature=0.1,
                        max_tokens=800,
                        timeout=210,
                        cache_bucket="osint-ontology",
                    )
                    if _llm_response_indicates_busy(response):
                        last_error = RuntimeError("llm_busy")
                        continue
                    response.raise_for_status()
                    data = response.json()
                    content = _llm_extract_content(provider, data)
                    if not content:
                        last_error = RuntimeError("empty_llm_response")
                        continue
                    parsed = self._ontology_parse_response(content)
                    if parsed:
                        graph = parsed
                        break
                    last_error = RuntimeError("invalid_ontology_payload")
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue

        if graph:
            graph = self._ontology_merge_graphs(previous_graph, graph)
            meta = graph.setdefault("meta", {})
            meta["method"] = "llm"
            meta["signature"] = signature or meta.get("signature") or ""
            meta["updated_at"] = datetime.utcnow().isoformat()
            meta["sources_seen"] = self._ontology_sources_seen(combined)
            meta["result_count"] = len(combined)
            processed_list = list(processed)
            processed_list.extend(self._ontology_item_key(item) for item in chunk)
            meta["processed_items"] = processed_list[-80:]
            meta.pop("retry_count", None)
            self.repository.set_job_ontology(job_id, graph, None)
            remaining = [
                item
                for item in combined
                if self._ontology_item_key(item) not in set(meta.get("processed_items") or [])
            ]
            if remaining:
                self._schedule_ontology_retry(job_id, signature, 1)
            self.logger.info(
                "OSINT ontology updated via LLM job=%s nodes=%s edges=%s remaining=%s",
                job_id,
                len(graph.get("nodes") or []),
                len(graph.get("edges") or []),
                len(remaining),
            )
            return

        fallback_graph = self._ontology_fallback(
            query,
            combined,
            note="LLM недоступен, использован эвристический обзор" if last_error else None,
        )
        fallback_graph = self._ontology_merge_graphs(previous_graph, fallback_graph)
        meta = fallback_graph.setdefault("meta", {})
        meta["method"] = "fallback"
        meta["signature"] = signature or meta.get("signature") or ""
        meta["updated_at"] = datetime.utcnow().isoformat()
        meta["sources_seen"] = self._ontology_sources_seen(combined)
        meta["result_count"] = len(combined)
        meta.pop("processed_items", None)
        retry_count = int(meta.get("retry_count") or 0) + 1
        meta["retry_count"] = retry_count
        error_text = str(last_error) if last_error else "ontology_unavailable"
        self.repository.set_job_ontology(job_id, fallback_graph, error_text if last_error else None)
        if retry_count <= 3:
            self._schedule_ontology_retry(job_id, signature, retry_count)
        self.logger.warning(
            "OSINT ontology fallback job=%s retry=%s error=%s",
            job_id,
            retry_count,
            error_text,
        )

    @staticmethod
    def _ontology_messages(
        query: str,
        items: list[dict[str, Any]],
        previous: dict | None,
        *,
        total: int,
        limit: int = 14,
    ) -> list[dict[str, str]]:
        if not items:
            return []
        lines: list[str] = []
        for idx, item in enumerate(items[:limit], start=1):
            title = (item.get("title") or item.get("url") or f"Результат {idx}").strip()
            snippet = str(item.get("snippet") or "").strip()
            url = str(item.get("url") or "").strip()
            sources = item.get("sources") or []
            origin_parts: list[str] = []
            for src in sources:
                label = str(src.get("label") or src.get("id") or "").strip()
                rank = src.get("rank")
                if label and rank:
                    origin_parts.append(f"{label}#{rank}")
                elif label:
                    origin_parts.append(label)
            origin = ", ".join(origin_parts)
            entry_lines = [f"{idx}. {title}"]
            if origin:
                entry_lines[-1] += f" [{origin}]"
            if url:
                entry_lines.append(f"URL: {url}")
            if snippet:
                entry_lines.append(snippet)
            lines.append("\n".join(entry_lines))
        system_prompt = (
            "Ты аналитик OSINT. Построй онтологию найденных сущностей и их связей. "
            "Всегда отвечай на русском языке. Если название, компания или термин изначально дан на английском, сохрани его без изменения, "
            "а русский эквивалент (если уместно) добавь в summary или aliases. "
            "Группируй схожие названия в один узел (например, 'ООО Ромашка' и 'Ромашка LLC'). "
            "Определи тип узла из списка: person, organization, location, event, object, concept, other. "
            "Для каждой связи добавь краткое описание сути отношения (по-русски). "
            "Если передан текущий граф, обнови и расширь его: повторно используй идентификаторы существующих узлов, "
            "дополни описания, добавь новые вершины и связи. "
            "Верни только JSON с полями 'nodes' и 'edges' без пояснений, держи итоговый граф не более чем из 40 узлов и 80 рёбер."
        )
        previous_summary = OsintSearchService._ontology_existing_summary(previous) if previous else ""
        user_prompt = (
            f"Запрос: {query or 'нет запроса'}\n"
            "Выпиши сущности и связи на основе фактов ниже. "
            "Источники поиска:\n"
            + "\n\n".join(lines)
            + "\n\nТребуемый JSON-формат:\n"
            "{\n"
            '  "nodes": [\n'
            '    {"id": "string", "label": "string", "type": "person|organization|location|event|object|concept|other", '
            '"summary": "string", "aliases": ["string"], "sources": ["string"], "score": number}\n'
            "  ],\n"
            '  "edges": [\n'
            '    {"from": "node_id", "to": "node_id", "relation": "string", "description": "string", '
            '"sources": ["string"], "confidence": number}\n'
            "  ]\n"
            "}\n"
            "Не придумывай фактов и не включай связи без опоры на найденные данные."
        )
        if previous_summary:
            user_prompt += (
                "\n\nТекущий граф (сохрани существующие узлы и дополни новыми данными):\n"
                f"{previous_summary}"
            )
        user_prompt += (
            f"\n\nВ этом чанке передано {len(items)} результатов из {total}."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _ontology_existing_summary(
        graph: dict | None,
        *,
        max_nodes: int = 16,
        max_edges: int = 20,
    ) -> str:
        if not graph:
            return ""
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) and not isinstance(edges, list):
            return ""
        lines: list[str] = []
        if isinstance(nodes, list) and nodes:
            lines.append("Текущие узлы:")
            for node in nodes[:max_nodes]:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id") or "").strip()
                label = str(node.get("label") or "").strip()
                node_type = str(node.get("type") or "").strip()
                summary = OsintSearchService._ontology_clean_text(node.get("summary"))
                alias_list = ", ".join(node.get("aliases", [])[:4]) if isinstance(node.get("aliases"), list) else ""
                parts = [f"- {node_id or label or 'node'}"]
                if label and node_id and label.lower() != node_id.lower():
                    parts.append(f"«{label}»")
                if node_type:
                    parts.append(f"[{node_type}]")
                if alias_list:
                    parts.append(f"алиасы: {alias_list}")
                if summary:
                    parts.append(f"описание: {summary}")
                lines.append(" ".join(parts))
        if isinstance(edges, list) and edges:
            lines.append("\nТекущие связи:")
            for edge in edges[:max_edges]:
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("from") or "").strip()
                dst = str(edge.get("to") or "").strip()
                relation = str(edge.get("label") or edge.get("relation") or "").strip()
                description = OsintSearchService._ontology_clean_text(edge.get("description"))
                parts = [f"- {src or '?'} -> {dst or '?'}"]
                if relation:
                    parts.append(f"({relation})")
                if description:
                    parts.append(f"описание: {description}")
                lines.append(" ".join(parts))
        return "\n".join(lines).strip()

    @staticmethod
    def _ontology_parse_response(content: str) -> dict | None:
        if not content:
            return None
        text = content.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = text.strip()
        try:
            data = json.loads(text)
        except Exception:
            return None
        return OsintSearchService._ontology_normalize_graph(data)

    @staticmethod
    def _ontology_normalize_graph(data: Any) -> dict | None:
        if not isinstance(data, dict):
            return None
        raw_nodes = data.get("nodes")
        raw_edges = data.get("edges")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            return None
        nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        for idx, item in enumerate(raw_nodes, start=1):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or "").strip()
            if not label:
                continue
            node_id_raw = item.get("id") or item.get("key") or item.get("slug")
            node_id = OsintSearchService._ontology_clean_id(str(node_id_raw)) if node_id_raw else ""
            if not node_id:
                node_id = OsintSearchService._ontology_make_id(label, idx)
            while node_id in node_ids:
                node_id = f"{node_id}-{idx}"
                idx += 1
            node_ids.add(node_id)
            node = {
                "id": node_id,
                "label": label,
                "type": OsintSearchService._ontology_clean_type(item.get("type") or item.get("category") or item.get("kind")),
                "summary": OsintSearchService._ontology_clean_text(item.get("summary") or item.get("description")),
                "aliases": OsintSearchService._ontology_clean_list(item.get("aliases")),
                "sources": OsintSearchService._ontology_clean_list(item.get("sources")),
                "score": OsintSearchService._ontology_clean_score(item.get("score") or item.get("confidence")),
            }
            nodes.append(node)
            if len(nodes) >= 20:
                break
        node_id_set = {node["id"] for node in nodes}
        edges: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str, str]] = set()
        edge_index = 1
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            src_raw = item.get("from") or item.get("source") or item.get("start")
            dst_raw = item.get("to") or item.get("target") or item.get("end")
            if not src_raw or not dst_raw:
                continue
            src = OsintSearchService._ontology_clean_id(str(src_raw))
            dst = OsintSearchService._ontology_clean_id(str(dst_raw))
            relation = OsintSearchService._ontology_clean_text(item.get("relation") or item.get("type") or item.get("label"))
            key = (src, dst, relation or "")
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edge = {
                "id": f"edge-{edge_index}",
                "from": src,
                "to": dst,
                "label": relation or None,
                "description": OsintSearchService._ontology_clean_text(item.get("description") or item.get("summary")),
                "sources": OsintSearchService._ontology_clean_list(item.get("sources")),
                "confidence": OsintSearchService._ontology_clean_score(item.get("confidence") or item.get("score")),
            }
            edge_index += 1
            edges.append(edge)
            if len(edges) >= 30:
                break
        # Filter edges referencing missing nodes
        nodes_by_id = {node["id"]: node for node in nodes}
        cleaned_edges = [
            edge for edge in edges if edge["from"] in nodes_by_id and edge["to"] in nodes_by_id
        ]
        # Ensure node IDs referenced in edges use normalized forms
        referenced_ids = {edge["from"] for edge in cleaned_edges} | {edge["to"] for edge in cleaned_edges}
        # If LLM referred to nodes via original IDs, attempt to map using aliases/labels
        if referenced_ids - node_id_set:
            label_map = {OsintSearchService._ontology_clean_id(node["label"]): node["id"] for node in nodes}
            alias_map: dict[str, str] = {}
            for node in nodes:
                for alias in node.get("aliases") or []:
                    normalized_alias = OsintSearchService._ontology_clean_id(alias)
                    if normalized_alias:
                        alias_map.setdefault(normalized_alias, node["id"])
            remapped_edges: list[dict[str, Any]] = []
            for edge in cleaned_edges:
                src = edge["from"]
                dst = edge["to"]
                mapped_src = src
                mapped_dst = dst
                if src not in node_id_set:
                    mapped_src = alias_map.get(src) or label_map.get(src) or src
                if dst not in node_id_set:
                    mapped_dst = alias_map.get(dst) or label_map.get(dst) or dst
                if mapped_src in node_id_set and mapped_dst in node_id_set:
                    edge["from"] = mapped_src
                    edge["to"] = mapped_dst
                    remapped_edges.append(edge)
            cleaned_edges = remapped_edges
        # Compact nodes list to those referenced or first N
        referenced = {edge["from"] for edge in cleaned_edges} | {edge["to"] for edge in cleaned_edges}
        if referenced:
            nodes = [node for node in nodes if node["id"] in referenced or node["score"]]
        return {"nodes": nodes, "edges": cleaned_edges}

    @staticmethod
    def _ontology_merge_graphs(
        base: dict | None,
        update: dict | None,
    ) -> dict:
        if not base:
            return update or {"nodes": [], "edges": []}
        if not update:
            return base

        def merge_lists(values: Iterable[Any]) -> list[str]:
            deduped: list[str] = []
            seen: set[str] = set()
            for value in values:
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(text)
            return deduped

        nodes_map: dict[str, dict[str, Any]] = {}
        for source in (base, update):
            for node in (source.get("nodes") or []) if isinstance(source, dict) else []:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id") or "").strip()
                if not node_id:
                    continue
                existing = nodes_map.get(node_id, {})
                merged = dict(existing)
                for key in ("id", "label", "type"):
                    value = node.get(key)
                    if value:
                        merged[key] = value
                summary = OsintSearchService._ontology_clean_text(
                    node.get("summary") or existing.get("summary")
                )
                if summary:
                    merged["summary"] = summary
                score = OsintSearchService._ontology_clean_score(
                    node.get("score") if node.get("score") is not None else existing.get("score")
                )
                if score is not None:
                    merged["score"] = score
                aliases = merge_lists(
                    [
                        *(existing.get("aliases") or []),
                        *(node.get("aliases") or []),
                    ]
                )
                merged["aliases"] = aliases
                sources = merge_lists(
                    [
                        *(existing.get("sources") or []),
                        *(node.get("sources") or []),
                    ]
                )
                merged["sources"] = sources
                nodes_map[node_id] = merged

        edges_map: dict[tuple[str, str, str], dict[str, Any]] = {}
        edge_counter = 1
        for source in (base, update):
            for edge in (source.get("edges") or []) if isinstance(source, dict) else []:
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("from") or edge.get("source") or "").strip()
                dst = str(edge.get("to") or edge.get("target") or "").strip()
                relation = str(edge.get("label") or edge.get("relation") or "").strip()
                if not src or not dst:
                    continue
                key = (src, dst, relation.lower())
                existing = edges_map.get(key)
                description = OsintSearchService._ontology_clean_text(
                    edge.get("description") or edge.get("summary")
                )
                confidence = OsintSearchService._ontology_clean_score(
                    edge.get("confidence") if edge.get("confidence") is not None else edge.get("score")
                )
                if existing:
                    if description:
                        existing["description"] = description
                    if confidence is not None:
                        existing["confidence"] = confidence
                    merged_sources = merge_lists(
                        [
                            *(existing.get("sources") or []),
                            *(edge.get("sources") or []),
                        ]
                    )
                    existing["sources"] = merged_sources
                    continue
                edge_id = str(edge.get("id") or "").strip()
                if not edge_id:
                    edge_id = f"edge-{edge_counter}"
                edge_counter += 1
                merged = {
                    "id": edge_id,
                    "from": src,
                    "to": dst,
                    "label": relation or None,
                    "description": description,
                    "sources": merge_lists(edge.get("sources") or []),
                    "confidence": confidence,
                }
                edges_map[key] = merged

        merged_graph = {
            "nodes": list(nodes_map.values()),
            "edges": list(edges_map.values()),
        }
        meta = dict(base.get("meta") or {})
        meta.update(update.get("meta") or {})
        if meta:
            merged_graph["meta"] = meta
        return OsintSearchService._ontology_prune_graph(merged_graph)

    @staticmethod
    def _ontology_sources_seen(combined: list[dict[str, Any]]) -> list[str]:
        sources: set[str] = set()
        for item in combined or []:
            for ref in item.get("sources") or []:
                if not isinstance(ref, dict):
                    continue
                label = str(ref.get("id") or ref.get("label") or "").strip()
                if label:
                    sources.add(label)
        return sorted(sources)

    @staticmethod
    def _ontology_prune_graph(graph: dict, *, max_nodes: int = 40, max_edges: int = 80) -> dict:
        if not isinstance(graph, dict):
            return graph
        nodes_raw = list(graph.get("nodes") or [])
        edges_raw = list(graph.get("edges") or [])
        if len(nodes_raw) > max_nodes:
            nodes_raw = nodes_raw[:max_nodes]
        node_ids = {node.get("id") for node in nodes_raw if isinstance(node, dict) and node.get("id")}
        edges_filtered = [
            edge
            for edge in edges_raw
            if isinstance(edge, dict)
            and edge.get("from") in node_ids
            and edge.get("to") in node_ids
        ]
        if len(edges_filtered) > max_edges:
            edges_filtered = edges_filtered[:max_edges]
        pruned = dict(graph)
        pruned["nodes"] = nodes_raw
        pruned["edges"] = edges_filtered
        return pruned

    @staticmethod
    def _ontology_signature(combined: list[dict[str, Any]], *, limit: int = 40) -> str:
        if not combined:
            return "empty"
        payload: list[dict[str, Any]] = []
        for item in combined[:limit]:
            if not isinstance(item, dict):
                continue
            entry = {
                "id": item.get("id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
                "sources": [
                    {
                        "id": ref.get("id"),
                        "label": ref.get("label"),
                        "rank": ref.get("rank"),
                    }
                    for ref in (item.get("sources") or [])
                    if isinstance(ref, dict)
                ],
            }
            payload.append(entry)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _ontology_clean_id(value: str) -> str:
        if not value:
            return ""
        normalized = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ_\-:.]+", "-", value.strip())
        normalized = normalized.strip("-:.")
        return normalized[:64]

    @staticmethod
    def _ontology_make_id(label: str, idx: int) -> str:
        base = OsintSearchService._ontology_clean_id(label.lower())
        if not base:
            base = f"node-{idx}"
        if len(base) < 4:
            base = f"{base}-{idx}"
        return base[:64]

    @staticmethod
    def _ontology_clean_type(raw: Any) -> str | None:
        if not raw:
            return None
        text = str(raw).strip().lower()
        allowed = {"person", "organization", "location", "event", "object", "concept", "other", "company", "entity"}
        if text in allowed:
            if text == "company":
                return "organization"
            if text == "entity":
                return "concept"
            return text
        if any(key in text for key in ("org", "company", "firm", "corp")):
            return "organization"
        if any(key in text for key in ("person", "individual", "human", "персона")):
            return "person"
        if any(key in text for key in ("city", "country", "location", "place", "регион", "город")):
            return "location"
        return "other"

    @staticmethod
    def _ontology_clean_text(raw: Any, *, limit: int = 320) -> str | None:
        if not raw:
            return None
        text = str(raw).strip()
        if not text:
            return None
        text = re.sub(r"\s+", " ", text)
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return text

    @staticmethod
    def _ontology_clean_list(raw: Any, *, limit: int = 6) -> list[str]:
        if raw is None:
            return []
        items: list[str] = []
        if isinstance(raw, str):
            # split on commas or semicolons
            parts = re.split(r"[;,]", raw)
            items = [part.strip() for part in parts if part.strip()]
        elif isinstance(raw, (list, tuple, set)):
            for value in raw:
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    items.append(text)
        else:
            text = str(raw).strip()
            if text:
                items = [text]
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _ontology_clean_score(raw: Any) -> float | None:
        if raw is None:
            return None
        try:
            value = float(raw)
        except Exception:
            return None
        if not value and value != 0.0:
            return None
        if value > 1:
            value = min(value, 100.0)
        if value < 0:
            value = 0.0
        return round(value, 4)

    @staticmethod
    def _ontology_fallback(
        query: str,
        combined: list[dict[str, Any]],
        *,
        limit: int = 8,
        note: str | None = None,
    ) -> dict:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        if query:
            nodes.append(
                {
                    "id": "query",
                    "label": query,
                    "type": "concept",
                    "summary": None,
                    "aliases": [],
                    "sources": [],
                    "score": None,
                }
            )
        for idx, item in enumerate(combined[:limit], start=1):
            label = (item.get("title") or item.get("url") or f"Результат {idx}").strip()
            node_id = f"result-{idx}"
            sources = [str(src.get("label") or src.get("id") or "").strip() for src in item.get("sources") or [] if str(src.get("label") or src.get("id") or "").strip()]
            snippet = OsintSearchService._ontology_clean_text(item.get("snippet"))
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "type": "object",
                    "summary": snippet,
                    "aliases": [],
                    "sources": sources,
                    "score": None,
                }
            )
            if query:
                edges.append(
                    {
                        "id": f"edge-{idx}",
                        "from": "query",
                        "to": node_id,
                        "label": "related",
                        "description": None,
                        "sources": sources,
                        "confidence": None,
                    }
                )
        return {
            "nodes": nodes,
            "edges": edges,
            "meta": (
                {"source_count": len(combined), "note": note}
                if note
                else {"source_count": len(combined)}
            ),
        }

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_job_markdown(self, job_id: int) -> str:
        job_snapshot = self.repository.get_job(job_id)
        if not job_snapshot:
            raise ValueError("osint_job_not_found")
        return self._build_job_markdown(job_snapshot)

    @staticmethod
    def _build_job_markdown(job_snapshot: dict) -> str:
        def _clean(value: str | None, *, fallback: str = "") -> str:
            text = str(value or fallback)
            return " ".join(text.split())

        lines: list[str] = []
        job_id = job_snapshot.get("id")
        header = f"# OSINT-отчёт по задаче #{job_id}" if job_id else "# OSINT-отчёт"
        lines.append(header)
        lines.append("")
        query = _clean(job_snapshot.get("query"), fallback="(не указан)")
        lines.append(f"**Запрос:** {query}")
        locale = job_snapshot.get("locale") or "ru-RU"
        region = job_snapshot.get("region") or "по умолчанию"
        lines.append(f"**Локаль:** {locale}, **Регион:** {region}")
        created = job_snapshot.get("created_at") or "(нет данных)"
        lines.append(f"**Создано:** {created}")
        status = job_snapshot.get("status") or "unknown"
        lines.append(f"**Статус:** {status}")
        params = job_snapshot.get("params") or {}
        keywords = params.get("keywords")
        if isinstance(keywords, list) and keywords:
            formatted_keywords = ", ".join(str(term) for term in keywords if term)
            if formatted_keywords:
                lines.append(f"**Ключевые слова:** {formatted_keywords}")
        lines.append("")

        schedule = job_snapshot.get("schedule")
        if schedule:
            lines.append("## Расписание")
            interval = schedule.get("interval_minutes")
            next_run = schedule.get("next_run_at") or "—"
            last_run = schedule.get("last_run_at") or "—"
            lines.append(f"- Интервал: {interval} мин.")
            lines.append(f"- Следующий запуск: {next_run}")
            lines.append(f"- Последний запуск: {last_run}")
            lines.append("")

        sources = job_snapshot.get("sources") or []
        if sources:
            lines.append("## Источники")
            for source in sources:
                label = (
                    source.get("metadata", {}).get("label")
                    or source.get("source")
                    or "Источник"
                )
                engine = source.get("engine") or source.get("metadata", {}).get("source")
                status_text = source.get("status") or "unknown"
                lines.append(f"### {label}")
                if engine:
                    lines.append(f"- Тип: {engine}")
                lines.append(f"- Статус: {status_text}")
                if source.get("error"):
                    lines.append(f"- Ошибка: {source['error']}")
                if source.get("metadata", {}).get("fallback"):
                    lines.append("- Требуется ручная проверка (капча)")
                results = source.get("results") or []
                if results:
                    lines.append("#### Топ результатов")
                    for item in results[:5]:
                        title = _clean(item.get("title"), fallback="(без названия)")
                        url = item.get("url") or ""
                        snippet = _clean(item.get("snippet"))
                        entry = f"- [{title}]({url})" if url else f"- {title}"
                        if snippet:
                            entry += f" — {snippet}"
                        lines.append(entry)
                else:
                    lines.append("- Результаты недоступны")
                lines.append("")

        analysis = job_snapshot.get("analysis")
        if analysis:
            lines.append("## Автоматический анализ")
            lines.append(analysis.strip())
            lines.append("")
        elif job_snapshot.get("analysis_error"):
            lines.append("## Автоматический анализ")
            lines.append(f"Не выполнен: {job_snapshot['analysis_error']}")
            lines.append("")

        combined = job_snapshot.get("combined_results") or []
        if combined:
            lines.append("## Сводные результаты")
            for item in combined:
                title = _clean(item.get("title"), fallback="(без названия)")
                url = item.get("url") or ""
                snippet = _clean(item.get("snippet"))
                entry = f"- [{title}]({url})" if url else f"- {title}"
                if snippet:
                    entry += f" — {snippet}"
                lines.append(entry)
            lines.append("")

        ontology_error = job_snapshot.get("ontology_error")
        if job_snapshot.get("ontology"):
            lines.append("## Онтология")
            lines.append("Построена автоматически (см. граф в интерфейсе).")
            lines.append("")
        elif ontology_error:
            lines.append("## Онтология")
            lines.append(f"Не построена: {ontology_error}")
            lines.append("")

        if not analysis and not combined:
            lines.append("*(Отчёт сформирован без LLM из карточек источников.)*")

        return "\n".join(lines).strip() + "\n"

    # ------------------------------------------------------------------
    # Combined results helper
    # ------------------------------------------------------------------
    def _network_available(self, *, ttl: float = 30.0) -> bool:
        now = time.time()
        with self._network_probe_lock:
            if self._network_probe_cache and (now - self._network_probe_cache[0]) < ttl:
                return self._network_probe_cache[1]
        status = self._probe_network()
        with self._network_probe_lock:
            self._network_probe_cache = (time.time(), status)
        return status

    @staticmethod
    def _probe_network(timeout: float = 3.0) -> bool:
        try:
            sock = socket.create_connection(("1.1.1.1", 53), timeout=timeout)
        except OSError:
            return False
        else:
            try:
                sock.close()
            except Exception:
                pass
            return True

    @staticmethod
    def _filesystem_terms(query: str) -> list[str]:
        if not query:
            return []
        return [part for part in re.split(r"\s+", query.strip()) if part]

    @staticmethod
    def _highlight_text(text: str | None, terms: Iterable[str]) -> str | None:
        if not text:
            return None
        normalized_terms = [term for term in terms if term]
        if not normalized_terms:
            return None
        escaped = html.escape(text)
        unique_terms = sorted({term.lower() for term in normalized_terms if term.strip()}, key=len, reverse=True)
        if not unique_terms:
            return None
        for term in unique_terms:
            try:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
            except re.error:
                continue
            escaped = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", escaped)
        return escaped

    @staticmethod
    def _extract_structured(text: str | None, url: str | None = None) -> dict[str, Any]:
        if not text:
            return {}
        emails = sorted({match.group(0) for match in _EMAIL_RE.finditer(text)})
        phones = sorted({re.sub(r"\s+", " ", match.group(0).strip()) for match in _PHONE_RE.finditer(text)})
        domain = None
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
            except Exception:
                domain = None
        payload: dict[str, Any] = {}
        if emails:
            payload["emails"] = emails
        if phones:
            payload["phones"] = phones
        if domain:
            payload.setdefault("domains", []).append(domain)
        return payload

    @staticmethod
    def _filesystem_should_scan(path: Path, file_size: int | None) -> bool:
        if file_size is None:
            return True
        suffix = path.suffix.lower()
        suffix = path.suffix.lower()
        if suffix in _TEXT_FILE_EXTENSIONS:
            if file_size is None:
                return True
            return file_size <= 8_000_000  # 8 MB
        if suffix in _DOCUMENT_FILE_EXTENSIONS:
            if file_size is None:
                return True
            return file_size <= 32_000_000  # 32 MB
        if file_size is None:
            return True
        return file_size <= 512_000

    @staticmethod
    def _filesystem_scan_file(
        path: Path,
        terms: list[str],
        file_size: int | None,
        *,
        max_bytes: int = 500_000,
        force_ocr: bool = False,
    ) -> str | None:
        if not terms:
            return None
        if not OsintSearchService._filesystem_should_scan(path, file_size):
            return None
        suffix = path.suffix.lower()
        if suffix in _TEXT_FILE_EXTENSIONS:
            try:
                with path.open("rb") as handle:
                    data = handle.read(max_bytes)
            except Exception:
                return None
            if not data:
                return None
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                try:
                    text = data.decode("cp1251", errors="ignore")
                except Exception:
                    text = data.decode("latin-1", errors="ignore")
            return OsintSearchService._snippet_from_text(text, terms)
        if suffix in _DOCUMENT_FILE_EXTENSIONS:
            text = OsintSearchService._filesystem_extract_document(path, suffix, limit=20000, force_ocr=force_ocr)
            if not text:
                return None
            return OsintSearchService._snippet_from_text(text, terms)
        return None

    @staticmethod
    def _filesystem_extract_document(path: Path, suffix: str, *, limit: int, force_ocr: bool = False) -> str:
        try:
            from app import (  # type: ignore
                extract_text_djvu,
                extract_text_docx,
                extract_text_epub,
                extract_text_pdf,
                extract_text_rtf,
            )
        except Exception:
            return ""
        try:
            if suffix == ".pdf":
                return extract_text_pdf(path, limit_chars=limit, force_ocr_first_page=force_ocr)
            if suffix == ".docx":
                return extract_text_docx(path, limit_chars=limit)
            if suffix == ".rtf":
                return extract_text_rtf(path, limit_chars=limit)
            if suffix == ".epub":
                return extract_text_epub(path, limit_chars=limit)
            if suffix == ".djvu":
                return extract_text_djvu(path, limit_chars=limit)
        except Exception:
            return ""
        return ""

    @staticmethod
    def _snippet_from_text(text: str, terms: list[str], *, window: int = 160) -> str | None:
        if not text:
            return None
        normalized = re.sub(r"\s+", " ", text)
        if not normalized:
            return None
        lower = normalized.lower()
        match_index = -1
        matched_term = ""
        for term in terms:
            idx = lower.find(term)
            if idx != -1:
                match_index = idx
                matched_term = term
                break
        if match_index == -1:
            start = 0
            end = min(len(normalized), window * 2)
        else:
            start = max(0, match_index - window)
            end = min(len(normalized), match_index + len(matched_term) + window)
        snippet = normalized[start:end].strip()
        return snippet or None

    @staticmethod
    def _enrich_job(snapshot: dict | None) -> dict | None:
        if not snapshot:
            return snapshot
        snapshot["combined_results"] = OsintSearchService._combine_results(snapshot)
        return snapshot

    @staticmethod
    def _normalize_url(url: str | None) -> str | None:
        if not url or not isinstance(url, str):
            return None
        trimmed = url.strip()
        if not trimmed:
            return None
        try:
            parsed = urlparse(trimmed)
        except Exception:
            return trimmed.lower()
        host = parsed.netloc or ""
        path = parsed.path or ""
        if not host and not path:
            return None
        normalized_host = host.lower()
        normalized_path = path or "/"
        if normalized_path.endswith("/") and len(normalized_path) > 1:
            normalized_path = normalized_path.rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{normalized_host}{normalized_path}{query}"

    @staticmethod
    def _extract_domain(url: str | None) -> str | None:
        normalized = OsintSearchService._normalize_url(url)
        if not normalized:
            return None
        if "/" in normalized:
            host = normalized.split("/", 1)[0]
        else:
            host = normalized
        if not host:
            return None
        try:
            parsed = urlparse(f"https://{host}" if "://" not in host else host)
        except Exception:
            parsed = None
        hostname = (parsed.hostname if parsed else host) or ""
        hostname = hostname.lower()
        if not hostname:
            return None
        parts = hostname.split(".")
        if len(parts) >= 3 and parts[-2] in {"co", "com", "net", "org", "gov"}:
            return ".".join(parts[-3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname

    @staticmethod
    def _result_rank_score(rank: Any) -> float:
        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            return 0.65
        if rank_value <= 0:
            return 1.2
        if rank_value == 1:
            return 1.1
        return max(0.35, 1.0 / math.log2(rank_value + 1.2))

    @staticmethod
    def _source_weight(metadata: dict[str, Any]) -> float:
        if not isinstance(metadata, dict):
            metadata = {}
        weight = 1.0
        source_name = str(metadata.get("source") or metadata.get("engine") or "").lower()
        if source_name in {"catalog", "filesystem"}:
            weight *= 1.25
        elif source_name in {"google", "yandex"}:
            weight *= 1.0
        else:
            weight *= 1.05 if source_name else 1.0
        if metadata.get("fallback"):
            weight *= 0.7
        if metadata.get("blocked"):
            weight *= 0.6
        if metadata.get("from_cache"):
            weight *= 0.9
        return weight

    @staticmethod
    def _result_freshness_weight(metadata: dict[str, Any]) -> float:
        if not isinstance(metadata, dict):
            metadata = {}
        fetched_at_raw = metadata.get("fetched_at")
        if not fetched_at_raw:
            return 0.85
        try:
            fetched_at = datetime.fromisoformat(str(fetched_at_raw))
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            timestamp = fetched_at.astimezone(timezone.utc)
            age_hours = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600
        except Exception:
            return 0.85
        if age_hours <= 6:
            return 1.15
        if age_hours <= 24:
            return 1.0
        if age_hours <= 72:
            return 0.8
        return 0.6

    @staticmethod
    def _result_score(item: dict[str, Any], metadata: dict[str, Any]) -> float:
        if not isinstance(item, dict):
            return 0.0
        if not isinstance(metadata, dict):
            metadata = {}
        rank_score = OsintSearchService._result_rank_score(item.get("rank"))
        source_weight = OsintSearchService._source_weight(metadata)
        freshness_weight = OsintSearchService._result_freshness_weight(metadata)
        return round(rank_score * source_weight * freshness_weight, 6)

    @staticmethod
    def _combine_results(snapshot: dict) -> list[dict[str, Any]]:
        sources = snapshot.get("sources") or []
        if not sources:
            return []
        combined: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        preferred_meta_keys = ("path", "filename", "collection", "collection_id", "size", "match")

        for source in sources:
            if not isinstance(source, dict):
                continue
            raw_results = source.get("results") or []
            if not raw_results:
                continue
            metadata_raw = source.get("metadata") or {}
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            label = (
                metadata.get("label")
                or source.get("source")
                or source.get("engine")
                or "Источник"
            )
            source_id = str(source.get("source") or source.get("id") or label)
            engine = source.get("engine")
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("url") or "").strip()
                snippet = str(item.get("snippet") or "").strip() or None
                url = str(item.get("url") or "").strip() or None
                normalized_url = OsintSearchService._normalize_url(url)
                domain = OsintSearchService._extract_domain(url)
                highlight_value = item.get("highlight") if isinstance(item.get("highlight"), str) else None
                raw_meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                filtered_meta = {
                    key: raw_meta[key]
                    for key in preferred_meta_keys
                    if key in raw_meta and raw_meta[key] not in (None, "", [])
                }
                key = None
                for candidate in (
                    normalized_url,
                    filtered_meta.get("path"),
                    filtered_meta.get("filename"),
                ):
                    if isinstance(candidate, str) and candidate.strip():
                        key = candidate.strip().lower()
                        break
                if key is None and title:
                    key = f"title:{title.lower()}"
                if key is None and snippet:
                    key = f"snippet:{snippet.lower()}"
                if key is None:
                    key = f"{source_id}:{len(order)}:{len(combined)}"
                if key not in combined:
                    entry = {
                        "id": key,
                        "title": title or url or filtered_meta.get("path") or label,
                        "snippet": snippet,
                        "url": url,
                        "metadata": filtered_meta.copy(),
                        "sources": [],
                        "highlight": highlight_value,
                        "score": 0.0,
                    }
                    if domain:
                        entry.setdefault("metadata", {})
                        entry["metadata"]["domain"] = domain
                    combined[key] = entry
                    order.append(key)
                entry = combined[key]
                if snippet and not entry.get("snippet"):
                    entry["snippet"] = snippet
                if url and not entry.get("url"):
                    entry["url"] = url
                entry_metadata = entry.setdefault("metadata", {}) or {}
                if filtered_meta:
                    for meta_key, meta_value in filtered_meta.items():
                        entry_metadata.setdefault(meta_key, meta_value)
                if domain and "domain" not in entry_metadata:
                    entry_metadata["domain"] = domain
                entry["metadata"] = entry_metadata
                if highlight_value and not entry.get("highlight"):
                    entry["highlight"] = highlight_value
                score = OsintSearchService._result_score(item, metadata)
                entry["score"] = round(float(entry.get("score", 0.0)) + score, 6)
                entry["sources"].append(
                    {
                        "id": source_id,
                        "label": label,
                        "engine": engine,
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "domain": domain,
                        "contribution": score,
                    }
                )
        results: list[dict[str, Any]] = []
        for idx, key in enumerate(order):
            entry = combined[key]
            entry["_order"] = idx
            if not entry.get("metadata"):
                entry["metadata"] = None
            else:
                entry["metadata"] = dict(entry["metadata"])
            results.append(entry)
        results.sort(key=lambda item: (-float(item.get("score") or 0.0), item["_order"]))
        for entry in results:
            entry.pop("_order", None)
            entry["score"] = round(float(entry.get("score") or 0.0), 4)
        return results

    @staticmethod
    def _ontology_item_key(item: dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        key = item.get("id") or item.get("url") or item.get("title")
        if not key:
            key = json.dumps(item, sort_keys=True)
        return hashlib.sha1(str(key).encode("utf-8")).hexdigest()

    def _schedule_ontology_retry(self, job_id: int, signature: str | None, attempt: int) -> None:
        try:
            app = current_app._get_current_object()
        except RuntimeError:
            return
        delay_seconds = min(300, max(10, attempt * 30))
        self.logger.info(
            "OSINT ontology retry scheduled job=%s signature=%s attempt=%s delay=%s",
            job_id,
            signature,
            attempt,
            delay_seconds,
        )
        self.queue.submit(
            self._ontology_retry_task,
            app,
            job_id,
            signature,
            attempt,
            delay_seconds,
            description=f"osint:ontology-retry:{job_id}:{attempt}",
        )

    def _ontology_retry_task(self, app, job_id: int, expected_signature: str | None, attempt: int, delay_seconds: int) -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        with app.app_context():
            snapshot = self.repository.get_job(job_id)
            if not snapshot:
                return
            params = snapshot.get("params") or {}
            if not isinstance(params, dict):
                return
            build_requested = bool(params.get("build_ontology") or params.get("ontology_enabled"))
            if not build_requested:
                return
            combined = self._combine_results(snapshot)
            snapshot["combined_results"] = combined
            signature = self._ontology_signature(combined)
            ontology_meta = {}
            if isinstance(snapshot.get("ontology"), dict):
                ontology_meta = snapshot["ontology"].get("meta") or {}
            retry_count = int((ontology_meta or {}).get("retry_count") or 0)
            if retry_count > 3 and signature == expected_signature:
                return
            # ensure any previous error flags remain to allow re-processing
            self.logger.info(
                "OSINT ontology retry fired job=%s signature=%s attempt=%s status=%s",
                job_id,
                signature,
                attempt,
                snapshot.get("status"),
            )
            self._maybe_trigger_ontology(snapshot)
