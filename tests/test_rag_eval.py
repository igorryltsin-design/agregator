from agregator.rag.eval import RetrievalCase, evaluate_retrieval


def test_evaluate_retrieval_computes_expected_metrics():
    report = evaluate_retrieval(
        [
            RetrievalCase(
                query="q1",
                retrieved_ids=[10, 20, 30],
                expected_ids=[20],
                k=3,
                retrieved_citations=2,
                expected_citations=2,
            ),
            RetrievalCase(
                query="q2",
                retrieved_ids=[1, 2, 3],
                expected_ids=[7],
                k=3,
                retrieved_citations=0,
                expected_citations=1,
            ),
        ]
    )
    assert report["cases"] == 2
    assert report["avg_recall_at_k"] == 0.5
    assert report["avg_mrr"] == 0.25
    assert report["hit_rate"] == 0.5
    assert report["avg_citation_coverage"] == 0.5
