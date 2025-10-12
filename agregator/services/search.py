"""Search service encapsulating FTS maintenance and candidate lookup."""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, List, Optional

from sqlalchemy import and_, exists, or_, text


class SearchService:
    """Operate FTS-backed search helpers with lightweight caching."""

    _FTS_TOKEN_PATTERN = re.compile(r"[\w\d]+", re.UNICODE)

    def __init__(
        self,
        *,
        db,
        file_model,
        tag_model,
        cache_get: Optional[Callable[[str], Optional[Iterable[int]]]] = None,
        cache_set: Optional[Callable[[str, Iterable[int]], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db = db
        self.File = file_model
        self.Tag = tag_model
        self._cache_get = cache_get
        self._cache_set = cache_set
        self.logger = logger or logging.getLogger(__name__)
        self._cache_version = 0

    # ------------------------------------------------------------------
    # Вспомогательные методы работы с кешем
    # ------------------------------------------------------------------
    def invalidate_cache(self, reason: str | None = None) -> None:
        self._cache_version += 1
        if reason:
            self.logger.debug("Search cache invalidated: %s", reason)

    # ------------------------------------------------------------------
    # Обслуживание FTS
    # ------------------------------------------------------------------
    def ensure_support(self) -> None:
        """Ensure indexes, virtual tables and triggers exist for SQLite search."""
        from sqlalchemy import text as sql_text

        try:
            with self.db.engine.begin() as conn:
                statements = [
                    "CREATE INDEX IF NOT EXISTS idx_files_title_nocase ON files(title COLLATE NOCASE)",
                    "CREATE INDEX IF NOT EXISTS idx_files_author_nocase ON files(author COLLATE NOCASE)",
                    "CREATE INDEX IF NOT EXISTS idx_files_keywords_nocase ON files(keywords COLLATE NOCASE)",
                    "CREATE INDEX IF NOT EXISTS idx_files_filename_nocase ON files(filename COLLATE NOCASE)",
                    "CREATE INDEX IF NOT EXISTS idx_tags_key_nocase ON tags(key COLLATE NOCASE)",
                    "CREATE INDEX IF NOT EXISTS idx_tags_value_nocase ON tags(value COLLATE NOCASE)",
                ]
                for stmt in statements:
                    try:
                        conn.execute(sql_text(stmt))
                    except Exception as exc:  # pragma: no cover - только для диагностики
                        self.logger.debug('Index creation failed (%s): %s', stmt, exc)

                conn.execute(sql_text(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                        file_id UNINDEXED,
                        title,
                        author,
                        keywords,
                        abstract,
                        text_excerpt,
                        tokenize='unicode61'
                    )
                    """
                ))
                conn.execute(sql_text(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS tags_fts USING fts5(
                        tag_id UNINDEXED,
                        file_id UNINDEXED,
                        key,
                        value,
                        tokenize='unicode61'
                    )
                    """
                ))

                # Удаляем устаревшие триггеры (теперь их функции выполняет Python)
                try:
                    for legacy in (
                        'trg_files_ai', 'trg_files_au', 'trg_files_ad',
                        'trg_tags_ai', 'trg_tags_au', 'trg_tags_ad',
                    ):
                        conn.execute(sql_text(f"DROP TRIGGER IF EXISTS {legacy}"))
                except Exception as exc:  # pragma: no cover
                    self.logger.debug('Drop legacy trigger failed: %s', exc)

                self._rebuild_files_fts(connection=conn)
                self._rebuild_tags_fts(connection=conn)
            self.invalidate_cache('ensure_support')
        except Exception as exc:
            self.logger.warning('Failed to initialize search support: %s', exc)

    def _rebuild_files_fts(self, connection=None) -> None:
        sql = text(
            """
            INSERT INTO files_fts(rowid, file_id, title, author, keywords, abstract, text_excerpt)
            SELECT id, id,
                   coalesce(title,''),
                   coalesce(author,''),
                   coalesce(keywords,''),
                   coalesce(abstract,''),
                   coalesce(text_excerpt,'')
            FROM files
            """
        )
        try:
            if connection is None:
                with self.db.engine.begin() as conn:
                    conn.execute(text("DELETE FROM files_fts"))
                    conn.execute(sql)
            else:
                connection.execute(text("DELETE FROM files_fts"))
                connection.execute(sql)
        except Exception as exc:  # pragma: no cover
            self.logger.debug('files_fts rebuild failed: %s', exc)

    def rebuild_files(self, connection=None) -> None:
        self._rebuild_files_fts(connection=connection)

    def _rebuild_tags_fts(self, connection=None) -> None:
        sql = text(
            """
            INSERT INTO tags_fts(rowid, tag_id, file_id, key, value)
            SELECT id, id, file_id, coalesce(key,''), coalesce(value,'')
            FROM tags
            """
        )
        try:
            if connection is None:
                with self.db.engine.begin() as conn:
                    conn.execute(text("DELETE FROM tags_fts"))
                    conn.execute(sql)
            else:
                connection.execute(text("DELETE FROM tags_fts"))
                connection.execute(sql)
        except Exception as exc:  # pragma: no cover
            self.logger.debug('tags_fts rebuild failed: %s', exc)

    def rebuild_tags(self, connection=None) -> None:
        self._rebuild_tags_fts(connection=connection)

    def rebuild_all(self) -> None:
        self._rebuild_files_fts()
        self._rebuild_tags_fts()
        self.invalidate_cache('rebuild_all')

    def sync_file(self, file_obj) -> None:
        if not file_obj or not getattr(file_obj, 'id', None):
            return
        params = {
            'rid': file_obj.id,
            'title': file_obj.title or '',
            'author': file_obj.author or '',
            'keywords': file_obj.keywords or '',
            'abstract': getattr(file_obj, 'abstract', '') or '',
            'text_excerpt': file_obj.text_excerpt or '',
        }
        try:
            self.db.session.execute(text("DELETE FROM files_fts WHERE rowid = :rid"), {'rid': file_obj.id})
            self.db.session.execute(
                text(
                    """
                    INSERT INTO files_fts(rowid, file_id, title, author, keywords, abstract, text_excerpt)
                    VALUES (:rid, :rid, :title, :author, :keywords, :abstract, :text_excerpt)
                    """
                ),
                params,
            )
        except Exception as exc:  # pragma: no cover
            self.logger.debug('sync files_fts failed for id %s: %s', getattr(file_obj, 'id', None), exc)
        try:
            self.db.session.execute(text("DELETE FROM tags_fts WHERE file_id = :file_id"), {'file_id': file_obj.id})
            tag_rows = []
            for tag in list(getattr(file_obj, 'tags', []) or []):
                if not getattr(tag, 'id', None):
                    continue
                tag_rows.append({
                    'rowid': tag.id,
                    'tag_id': tag.id,
                    'file_id': file_obj.id,
                    'key': tag.key or '',
                    'value': tag.value or '',
                })
            if tag_rows:
                self.db.session.execute(
                    text(
                        """
                        INSERT INTO tags_fts(rowid, tag_id, file_id, key, value)
                        VALUES (:rowid, :tag_id, :file_id, :key, :value)
                        """
                    ),
                    tag_rows,
                )
        except Exception as exc:  # pragma: no cover
            self.logger.debug('sync tags_fts failed for file %s: %s', getattr(file_obj, 'id', None), exc)
        self.invalidate_cache(f'sync_file:{getattr(file_obj, "id", "?")}')

    def delete_file(self, file_id: int | None) -> None:
        if not file_id:
            return
        try:
            self.db.session.execute(text("DELETE FROM files_fts WHERE rowid = :rid"), {'rid': file_id})
        except Exception as exc:  # pragma: no cover
            self.logger.debug('delete from files_fts failed for %s: %s', file_id, exc)
        try:
            self.db.session.execute(text("DELETE FROM tags_fts WHERE file_id = :file_id"), {'file_id': file_id})
        except Exception as exc:  # pragma: no cover
            self.logger.debug('delete from tags_fts failed for %s: %s', file_id, exc)
        self.invalidate_cache(f'delete_file:{file_id}')

    # ------------------------------------------------------------------
    # Вспомогательные функции для построения запросов
    # ------------------------------------------------------------------
    def _fts_match_query(self, query: str) -> Optional[str]:
        tokens = self._FTS_TOKEN_PATTERN.findall((query or '').lower())
        cleaned = [token for token in tokens if token]
        if not cleaned:
            return None
        cleaned = cleaned[:8]
        return ' '.join(f'{token}*' for token in cleaned)

    def candidate_ids(self, query: str, limit: int = 4000) -> Optional[List[int]]:
        match_expr = self._fts_match_query(query)
        if match_expr is None:
            return []
        cache_key = f"fts::{self._cache_version}::{limit}::{match_expr}"
        if self._cache_get is not None:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return list(cached)

        ids: set[int] = set()
        try:
            rows = self.db.session.execute(
                text("SELECT rowid FROM files_fts WHERE files_fts MATCH :match LIMIT :limit"),
                {'match': match_expr, 'limit': limit},
            ).fetchall()
            ids.update(int(row[0]) for row in rows)
        except Exception as exc:
            self.logger.debug('files_fts MATCH failed: %s', exc)
            return None
        try:
            rows = self.db.session.execute(
                text("SELECT file_id FROM tags_fts WHERE tags_fts MATCH :match LIMIT :limit"),
                {'match': match_expr, 'limit': limit},
            ).fetchall()
            ids.update(int(row[0]) for row in rows)
        except Exception as exc:
            self.logger.debug('tags_fts MATCH failed: %s', exc)
        result = list(ids)
        if self._cache_set is not None:
            try:
                self._cache_set(cache_key, result)
            except Exception:  # pragma: no cover
                pass
        return result

    def apply_like_filter(self, base_query, query: str):
        like = f"%{query}%"
        File = self.File
        Tag = self.Tag
        filters = [
            File.title.ilike(like),
            File.author.ilike(like),
            File.keywords.ilike(like),
            File.filename.ilike(like),
            File.text_excerpt.ilike(like),
        ]
        if hasattr(File, 'abstract'):
            filters.append(File.abstract.ilike(like))
        tag_like = exists().where(and_(
            Tag.file_id == File.id,
            or_(Tag.value.ilike(like), Tag.key.ilike(like))
        ))
        filters.append(tag_like)
        return base_query.filter(or_(*filters))

    def apply_text_search_filter(self, base_query, query: str, limit: int = 4000):
        candidates = self.candidate_ids(query, limit=limit)
        if candidates is None:
            return self.apply_like_filter(base_query, query)
        if not candidates:
            return base_query.filter(self.File.id == -1)
        return base_query.filter(self.File.id.in_(candidates))


__all__ = ["SearchService"]
