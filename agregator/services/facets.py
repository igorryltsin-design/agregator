"""Facet service for aggregating search filters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from flask import current_app
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import aliased

from models import Collection, File, Tag, db


@dataclass(frozen=True)
class FacetQueryParams:
    query: str
    material_type: str
    context: str
    include_types: bool
    tag_filters: list[str]
    collection_filter: Optional[int]
    allowed_scope: Optional[set[int]]
    allowed_keys_list: Optional[list[str]]
    year_from: str
    year_to: str
    size_min: str
    size_max: str
    sources: dict
    request_args: tuple


class FacetService:
    def __init__(self, cache, logger: Optional[logging.Logger] = None) -> None:
        self.cache = cache
        self.logger = logger or logging.getLogger(__name__)
        self._cache_version = 0

    def invalidate(self, reason: Optional[str] = None) -> None:
        try:
            self._cache_version += 1
            self.cache.clear()
            if reason:
                self.logger.info("Facet cache invalidated: %s", reason)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to invalidate facet cache: %s", exc)

    def key_options(self, limit: int = 200) -> list[dict]:
        options: dict[str, dict] = {}
        try:
            rows = (
                db.session.query(Tag.key, func.count(Tag.id))
                .group_by(Tag.key)
                .order_by(func.count(Tag.id).desc())
                .limit(limit)
                .all()
            )
        except Exception:
            rows = []
        for key, count in rows:
            skey = str(key or '').strip()
            if not skey:
                continue
            options[skey] = {'key': skey, 'count': int(count or 0), 'samples': []}
        try:
            author_count = (
                db.session.query(func.count(File.id))
                .filter(File.author.isnot(None))
                .filter(File.author != '')
                .scalar()
            )
        except Exception:
            author_count = None
        if 'author' not in options:
            options['author'] = {'key': 'author', 'count': int(author_count or 0), 'samples': []}
        try:
            schema_keys = current_app.config.get('FACET_SCHEMA_KEYS') or []
            for key in schema_keys:
                skey = str(key or '').strip()
                if not skey or skey in options:
                    continue
                options[skey] = {'key': skey, 'count': 0, 'samples': []}
        except Exception:
            pass
        return sorted(options.values(), key=lambda x: x['key'])

    def get_facets(
        self,
        params: FacetQueryParams,
        *,
        search_candidate_fn,
        like_filter_fn,
    ) -> dict:
        if params.context == 'search':
            cache_key = (
                'facet-search',
                self._cache_version,
                params.request_args,
                None if params.allowed_scope is None else tuple(sorted(params.allowed_scope)),
                tuple(params.allowed_keys_list) if params.allowed_keys_list is not None else None,
                params.include_types,
            )
            try:
                return self.cache.get_or_set(cache_key, lambda: self._build_facets(params, search_candidate_fn, like_filter_fn))
            except AttributeError:
                # кеш не поддерживает get_or_set — используем запасной путь
                cached = getattr(self.cache, 'get', lambda key: None)(cache_key)
                if cached is not None:
                    return cached
                payload = self._build_facets(params, search_candidate_fn, like_filter_fn)
                getattr(self.cache, 'set', lambda key, value: None)(cache_key, payload)
                return payload
        return self._build_facets(params, search_candidate_fn, like_filter_fn)

    def _apply_scope(self, query, allowed_scope: Optional[set[int]]):
        if allowed_scope is None:
            return query
        if not allowed_scope:
            return query.filter(File.collection_id == -1)
        return query.filter(File.collection_id.in_(allowed_scope))

    def _build_facets(self, params: FacetQueryParams, search_candidate_fn, like_filter_fn) -> dict:
        base_query = self._apply_scope(File.query, params.allowed_scope)
        try:
            base_query = base_query.join(Collection, File.collection_id == Collection.id).filter(Collection.searchable == True)
        except Exception:
            pass
        if params.collection_filter is not None:
            base_query = base_query.filter(File.collection_id == params.collection_filter)
        if params.material_type:
            base_query = base_query.filter(File.material_type == params.material_type)

        filtered_query = base_query
        if params.query:
            candidates = search_candidate_fn(params.query, limit=6000)
            if candidates is None:
                filtered_query = like_filter_fn(filtered_query, params.query)
            elif not candidates:
                return {
                    'types': [] if params.include_types else [],
                    'tag_facets': {},
                    'include_types': params.include_types,
                    'allowed_keys': params.allowed_keys_list,
                    'context': params.context,
                }
            else:
                filtered_query = filtered_query.filter(File.id.in_(candidates))
        if params.year_from:
            filtered_query = filtered_query.filter(File.year >= params.year_from)
        if params.year_to:
            filtered_query = filtered_query.filter(File.year <= params.year_to)
        if params.size_min:
            try:
                filtered_query = filtered_query.filter(File.size >= int(params.size_min))
            except Exception:
                pass
        if params.size_max:
            try:
                filtered_query = filtered_query.filter(File.size <= int(params.size_max))
            except Exception:
                pass

        filtered_query = filtered_query.distinct()

        include_types = params.include_types
        types_facet: list[list] = []
        if include_types:
            ids_for_types = filtered_query.with_entities(File.id).subquery()
            types = (
                db.session.query(File.material_type, func.count(File.id))
                .filter(File.id.in_(ids_for_types))
                .group_by(File.material_type)
                .all()
            )
            types_facet = [[mt, cnt] for (mt, cnt) in types]

        tag_filters = params.tag_filters
        allowed_keys_set = set(params.allowed_keys_list or []) if params.allowed_keys_list is not None else None
        base_ids_subq = filtered_query.with_entities(File.id).subquery()
        base_keys = [row[0] for row in db.session.query(Tag.key).filter(Tag.file_id.in_(base_ids_subq)).distinct().all()]
        selected: dict[str, list[str]] = {}
        for tf in tag_filters:
            if '=' in tf:
                k, v = tf.split('=', 1)
                selected.setdefault(k, []).append(v)
                if k not in base_keys:
                    base_keys.append(k)

        if allowed_keys_set is not None:
            base_keys = [k for k in base_keys if k in allowed_keys_set or k in selected]
        for key in selected:
            if key not in base_keys:
                base_keys.append(key)

        tag_facets: dict[str, list[list]] = {}
        use_tags = params.sources.get('tags', True)
        if use_tags:
            for key in base_keys:
                if allowed_keys_set is not None and key not in allowed_keys_set and key not in selected:
                    continue
                qk = filtered_query
                for tf in tag_filters:
                    if '=' not in tf:
                        continue
                    k, v = tf.split('=', 1)
                    if k == key:
                        continue
                    tk = aliased(Tag)
                    qk = qk.join(tk, tk.file_id == File.id).filter(and_(tk.key == k, tk.value.ilike(f'%{v}%')))
                ids_subq = qk.with_entities(File.id).distinct().subquery()
                rows = (
                    db.session.query(Tag.value, func.count(Tag.id))
                    .filter(and_(Tag.file_id.in_(ids_subq), Tag.key == key))
                    .group_by(Tag.value)
                    .order_by(func.count(Tag.id).desc())
                    .all()
                )
                tag_facets[key] = [[val, cnt] for (val, cnt) in rows]
                if key in selected:
                    present_vals = {val for (val, _c) in tag_facets[key]}
                    for v in selected[key]:
                        if v not in present_vals:
                            tag_facets[key].append([v, 0])
        else:
            tag_facets = {}

        return {
            'types': types_facet,
            'tag_facets': tag_facets,
            'include_types': include_types,
            'allowed_keys': params.allowed_keys_list,
            'context': params.context,
        }


__all__ = ['FacetService', 'FacetQueryParams']
