#!/usr/bin/env python3
"""
Проверка, что ответы ИИ-поиска формулируются на русском языке.

Пример:
    python scripts/test_ai_search_language.py "submitted in partial fulfillment"

Можно указать несколько запросов:
    python scripts/test_ai_search_language.py -q "submitted in partial fulfillment" -q "deep neural networks"
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

from flask import g

from app import _ai_search_core, app


def contains_cyrillic(text: str) -> bool:
    return any('а' <= ch.lower() <= 'я' or ch.lower() == 'ё' for ch in text)


def has_en_dominance(text: str) -> bool:
    lat = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    cyr = sum(1 for ch in text if 'а' <= ch.lower() <= 'я' or ch.lower() == 'ё')
    return lat > 0 and cyr == 0


def run_query(query: str, top_k: int = 3) -> tuple[bool, str]:
    payload = {
        'query': query,
        'top_k': top_k,
        'deep_search': False,
        'llm_snippets': True,
        'max_candidates': 10,
    }
    with app.test_request_context('/ai-search-test'):
        g.current_user = SimpleNamespace(id=-1, username='ai-lang-test', role='admin')
        g.allowed_collection_ids = None
        result = _ai_search_core(payload)
    answer = (result.get('answer') or '').strip()
    if not answer:
        return False, "Пустой ответ"
    if has_en_dominance(answer) or not contains_cyrillic(answer):
        return False, f"Ответ без кириллицы: {answer[:160]}…"
    return True, answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Автоматическая проверка языка ответов ИИ-поиска.")
    parser.add_argument(
        '-q',
        '--query',
        action='append',
        dest='queries',
        help='Запрос для проверки (можно указать несколько). Если не задать, используется единичный тестовый запрос.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Количество документов в выдаче (по умолчанию 3).',
    )
    args = parser.parse_args(argv)

    queries = args.queries or ["submitted in partial fulfillment"]
    overall_ok = True
    for query in queries:
        ok, message = run_query(query, top_k=args.top_k)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {query!r} — {message}")
        overall_ok = overall_ok and ok
    return 0 if overall_ok else 1


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
