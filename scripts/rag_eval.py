#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from app import app, ensure_collections_schema, ensure_default_admin, ensure_llm_schema, setup_app, _ai_search_core
from agregator.rag import RetrievalCase, evaluate_retrieval


def _load_dataset(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: List[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    raise ValueError("Dataset must be JSON array or JSONL")


def _extract_context_ids(response: dict, mode: str) -> List[int]:
    rag_context = response.get("rag_context") or []
    ids: List[int] = []
    for section in rag_context:
        if not isinstance(section, dict):
            continue
        value = section.get("doc_id") if mode == "doc" else section.get("chunk_id")
        if value is None:
            continue
        try:
            ids.append(int(value))
        except Exception:
            continue
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality against a labeled dataset.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to JSON/JSONL dataset.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k for evaluation.")
    parser.add_argument(
        "--mode",
        choices=("doc", "chunk"),
        default="doc",
        help="Evaluate by expected_doc_ids or expected_chunk_ids.",
    )
    parser.add_argument("--deep", action="store_true", help="Enable deep_search for _ai_search_core.")
    parser.add_argument("--json-out", type=Path, help="Optional file path for full JSON report.")
    args = parser.parse_args()

    rows = _load_dataset(args.dataset)
    setup_app()
    with app.app_context():
        ensure_collections_schema()
        ensure_llm_schema()
        ensure_default_admin()

        cases: List[RetrievalCase] = []
        expected_key = "expected_doc_ids" if args.mode == "doc" else "expected_chunk_ids"
        for row in rows:
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            expected_ids_raw = row.get(expected_key) or []
            expected_ids = [int(x) for x in expected_ids_raw if str(x).strip().isdigit()]
            if not expected_ids:
                continue
            response = _ai_search_core(
                {
                    "query": query,
                    "top_k": max(1, int(args.top_k)),
                    "deep_search": bool(args.deep),
                    "use_rag": True,
                }
            )
            retrieved = _extract_context_ids(response or {}, args.mode)
            retrieved_citations = len(response.get("rag_context") or []) if isinstance(response, dict) else 0
            expected_citations = int(row.get("expected_citations") or len(expected_ids))
            cases.append(
                RetrievalCase(
                    query=query,
                    retrieved_ids=retrieved,
                    expected_ids=expected_ids,
                    k=max(1, int(args.top_k)),
                    retrieved_citations=retrieved_citations,
                    expected_citations=expected_citations,
                )
            )

    report = evaluate_retrieval(cases)
    print("RAG eval summary:")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {args.json_out}")


if __name__ == "__main__":
    main()
