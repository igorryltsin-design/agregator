from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(slots=True)
class RetrievalCase:
    query: str
    retrieved_ids: List[int]
    expected_ids: List[int]
    k: int = 5
    retrieved_citations: int = 0
    expected_citations: int = 0


def _recall_at_k(retrieved: Sequence[int], expected: Sequence[int], k: int) -> float:
    expected_set = {int(x) for x in expected}
    if not expected_set:
        return 0.0
    top = {int(x) for x in retrieved[: max(1, int(k))]}
    return len(expected_set & top) / float(len(expected_set))


def _mrr(retrieved: Sequence[int], expected: Sequence[int], k: int) -> float:
    expected_set = {int(x) for x in expected}
    if not expected_set:
        return 0.0
    for idx, item in enumerate(retrieved[: max(1, int(k))], start=1):
        if int(item) in expected_set:
            return 1.0 / float(idx)
    return 0.0


def _citation_coverage(retrieved_citations: int, expected_citations: int) -> float:
    expected = max(0, int(expected_citations))
    if expected == 0:
        return 1.0
    got = max(0, int(retrieved_citations))
    return min(1.0, got / float(expected))


def evaluate_retrieval(cases: Iterable[RetrievalCase]) -> Dict[str, object]:
    rows = list(cases)
    if not rows:
        return {
            "cases": 0,
            "avg_recall_at_k": 0.0,
            "avg_mrr": 0.0,
            "hit_rate": 0.0,
            "avg_citation_coverage": 0.0,
            "details": [],
        }

    details: List[Dict[str, object]] = []
    recall_values: List[float] = []
    mrr_values: List[float] = []
    hit_values: List[float] = []
    citation_values: List[float] = []

    for case in rows:
        recall = _recall_at_k(case.retrieved_ids, case.expected_ids, case.k)
        mrr = _mrr(case.retrieved_ids, case.expected_ids, case.k)
        hit = 1.0 if mrr > 0 else 0.0
        citation_cov = _citation_coverage(case.retrieved_citations, case.expected_citations)
        recall_values.append(recall)
        mrr_values.append(mrr)
        hit_values.append(hit)
        citation_values.append(citation_cov)
        details.append(
            {
                "query": case.query,
                "k": int(case.k),
                "expected_ids": [int(x) for x in case.expected_ids],
                "retrieved_ids": [int(x) for x in case.retrieved_ids],
                "recall_at_k": round(recall, 4),
                "mrr": round(mrr, 4),
                "hit": bool(hit),
                "citation_coverage": round(citation_cov, 4),
            }
        )

    total = float(len(rows))
    return {
        "cases": len(rows),
        "avg_recall_at_k": round(sum(recall_values) / total, 4),
        "avg_mrr": round(sum(mrr_values) / total, 4),
        "hit_rate": round(sum(hit_values) / total, 4),
        "avg_citation_coverage": round(sum(citation_values) / total, 4),
        "details": details,
    }


__all__ = ["RetrievalCase", "evaluate_retrieval"]
