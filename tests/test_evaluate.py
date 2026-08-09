from rag.evaluate import (
    LabelledQuery, evaluate, precision_at_k, recall_at_k,
    reciprocal_rank, summarise,
)
from rag.retrieve import Result


class FakeSearcher:
    def __init__(self, ids): self.ids = ids
    def search(self, query, k=5):
        return [Result(rank=i, score=1.0, chunk_id=c, doc_id="d",
                       text="", metadata={}) for i, c in enumerate(self.ids[:k], 1)]


def test_recall_counts_found_over_relevant():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5


def test_recall_respects_k():
    """A hit at rank 5 does not count when k is 2."""
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0


def test_recall_with_no_relevant_chunks_is_zero():
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_precision_counts_relevant_over_returned():
    assert precision_at_k(["a", "b", "c", "d"], ["a", "b"], k=4) == 0.5


def test_reciprocal_rank_rewards_position():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_evaluate_splits_by_kind():
    queries = [
        LabelledQuery("q1", ["a"], kind="exact"),
        LabelledQuery("q2", ["z"], kind="conceptual"),
    ]
    summary = summarise(evaluate(FakeSearcher(["a", "b"]), queries, k=2))
    assert summary["exact"]["recall"] == 1.0
    assert summary["conceptual"]["recall"] == 0.0
    assert summary["all"]["n"] == 2