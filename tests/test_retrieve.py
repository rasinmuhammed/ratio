import numpy as np

from rag.retrieve import rank


def test_ranks_by_similarity():
    """Three unit vectors, hand-picked so the answer is obvious: v0 exactly
    matches the query, v2 is 0.6 similar (their dot product), v1 orthogonal."""
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], dtype=np.float32)
    query = np.array([1, 0, 0], dtype=np.float32)

    idx, scores = rank(query, vectors, k=2)

    assert list(idx) == [0, 2]
    assert scores[0] == 1.0
    assert abs(scores[1] - 0.6) < 1e-6


def test_k_zero_returns_nothing():
    vectors = np.array([[1, 0, 0]], dtype=np.float32)
    idx, scores = rank(np.array([1, 0, 0], dtype=np.float32), vectors, k=0)
    assert len(idx) == 0


def test_k_larger_than_corpus_returns_all():
    """k > N must not crash. argpartition itself would raise here, which is
    exactly why rank() has the `k = min(k, len(scores))` branch."""
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], dtype=np.float32)
    idx, scores = rank(np.array([1, 0, 0], dtype=np.float32), vectors, k=99)
    assert len(idx) == 3
    assert list(idx) == [0, 2, 1]  # still ordered best to worst