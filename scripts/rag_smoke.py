#!/usr/bin/env python3
"""
Smoke-тест для RAG: запускает серию запросов и выводит краткую статистику.

Пример:
    python scripts/rag_smoke.py --top-k 3 --deep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app import (
    app,
    ensure_collections_schema,
    ensure_default_admin,
    ensure_llm_schema,
    setup_app,
    _ai_search_core,
)


DEFAULT_QUERIES = [
    "обеспечение безопасности данных блокчейн",
    "машинное обучение для диагностики",
    "логистика open-pit mines truck dispatching",
    "распознавание речи нейросети",
    "влияние цифровизации на экономику",
]


def run_smoke(queries: list[str], *, top_k: int, deep_search: bool) -> list[dict]:
    results: list[dict] = []
    with app.app_context():
        for query in queries:
            payload = {
                "query": query,
                "top_k": top_k,
                "deep_search": deep_search,
                "use_rag": True,
            }
            start = time.perf_counter()
            response = _ai_search_core(payload)
            elapsed = (time.perf_counter() - start) * 1000.0
            rag_context = response.get("rag_context") or []
            rag_validation = response.get("rag_validation") or {}
            results.append(
                {
                    "query": query,
                    "elapsed_ms": round(elapsed, 2),
                    "items": len(response.get("items") or []),
                    "rag_context": len(rag_context),
                    "rag_fallback": bool(response.get("rag_fallback")),
                    "warnings": bool(rag_validation.get("hallucination_warning")),
                    "context_ids": [
                        (section.get("doc_id"), section.get("chunk_id"))
                        for section in rag_context
                        if isinstance(section, dict)
                    ],
                    "notes": response.get("rag_notes") or [],
                }
            )
    return results


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"queries": 0}
    total = len(results)
    elapsed = sum(item.get("elapsed_ms", 0.0) for item in results)
    fallbacks = sum(1 for item in results if item.get("rag_fallback"))
    warned = sum(1 for item in results if item.get("warnings"))
    total_context = sum(item.get("rag_context", 0) for item in results)
    avg_elapsed = elapsed / total if total else 0.0
    avg_context = total_context / total if total else 0.0
    return {
        "queries": total,
        "avg_elapsed_ms": round(avg_elapsed, 2),
        "avg_context": round(avg_context, 2),
        "fallback_ratio": round(fallbacks / total, 3),
        "warnings_ratio": round(warned / total, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Запускает smoke-тест RAG поиска.")
    parser.add_argument(
        "--queries",
        type=Path,
        help="Путь к файлу со списком запросов (по одному на строку).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Количество документов в ответе (по умолчанию 3).",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Включить deep_search для smoke-теста.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Сохранить результат в JSON-файл.",
    )
    args = parser.parse_args()

    if args.queries and args.queries.exists():
        with args.queries.open("r", encoding="utf-8") as fh:
            queries = [line.strip() for line in fh if line.strip()]
    else:
        queries = DEFAULT_QUERIES

    setup_app()
    with app.app_context():
        ensure_collections_schema()
        ensure_llm_schema()
        ensure_default_admin()

    results = run_smoke(queries, top_k=args.top_k, deep_search=args.deep)
    summary = summarize(results)

    print("=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Details ===")
    for item in results:
        print(
            f"- {item['query']!r}: {item['elapsed_ms']} ms, "
            f"docs={item['items']}, rag_context={item['rag_context']}, "
            f"fallback={item['rag_fallback']}, warnings={item['warnings']}"
        )
        if item["notes"]:
            for note in item["notes"]:
                print(f"    note: {note}")

    if args.json:
        payload = {"summary": summary, "results": results}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
