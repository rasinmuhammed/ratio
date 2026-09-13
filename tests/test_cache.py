import numpy as np
import pytest

from rag.cache import SemanticCache


class FakeEmbedder:
    """Maps a query to a hand-assigned unit vector rather than a real
    model, so similarity between two queries is exactly what the test
    says it is instead of whatever bge-small happens to produce."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def embed_query(self, query: str) -> np.ndarray:
        return self.vectors[query]


def test_conceptual_queries_hit_on_high_similarity():
    v = np.array([1.0, 0.0])
    embedder = FakeEmbedder({
        "grounds for a temporary injunction": v,
        "when can a court grant a temporary injunction": v,  # identical vector
    })
    cache = SemanticCache(embedder, similarity_threshold=0.9)
    cache.put("grounds for a temporary injunction", "cached answer")

    hit = cache.get("when can a court grant a temporary injunction")
    assert hit == "cached answer"
    assert cache.stats.hits == 1


def test_conceptual_queries_miss_below_threshold():
    embedder = FakeEmbedder({
        "grounds for a temporary injunction": np.array([1.0, 0.0]),
        "promissory estoppel against the state": np.array([0.0, 1.0]),  # orthogonal
    })
    cache = SemanticCache(embedder, similarity_threshold=0.9)
    cache.put("grounds for a temporary injunction", "cached answer")

    hit = cache.get("promissory estoppel against the state")
    assert hit is None
    assert cache.stats.misses == 1


def test_identifier_queries_never_match_by_embedding_similarity():
    """The exact failure this module exists to prevent: two different
    citations that embed almost identically (route.py measures this
    directly for 'AIR 1974' vs 'AIR 1006') must never share a cache entry,
    no matter how similar their embeddings are.
    """
    # Deliberately identical vectors: if this cache fell back to embedding
    # similarity for identifiers the same way it does for conceptual
    # queries, this test would see a false hit.
    shared_vector = np.array([1.0, 0.0])
    embedder = FakeEmbedder({
        "AIR 1974 Patna 164": shared_vector,
        "AIR 1006 Patna 164": shared_vector,
    })
    cache = SemanticCache(embedder, similarity_threshold=0.5)
    cache.put("AIR 1974 Patna 164", "answer about the 1974 case")

    hit = cache.get("AIR 1006 Patna 164")
    assert hit is None


def test_identifier_queries_match_on_exact_normalised_string():
    """Whitespace-only variation should still hit. Case is deliberately not
    folded further than route.classify() itself tolerates: 'AIR' is a
    citation, 'air' is not, per the same case-sensitive pattern route.py
    uses to route it, so lowercasing here would test a normalisation this
    cache does not actually claim to do."""
    embedder = FakeEmbedder({"AIR 1974 Patna 164": np.array([1.0, 0.0])})
    cache = SemanticCache(embedder, similarity_threshold=0.99)
    cache.put("AIR 1974 Patna 164", "answer")

    assert cache.get("  AIR 1974 Patna 164  ") == "answer"


def test_identifier_put_does_not_call_the_embedder():
    """Embedding is real compute; an identifier's exact-match path has no
    use for a vector, so put() should not spend it on one."""
    class ExplodingEmbedder:
        def embed_query(self, query: str) -> np.ndarray:
            raise AssertionError("embed_query should not be called for an identifier")

    cache = SemanticCache(ExplodingEmbedder())
    cache.put("AIR 1974 Patna 164", "answer")  # must not raise
    assert cache.get("AIR 1974 Patna 164") == "answer"


def test_max_entries_evicts_oldest_first():
    embedder = FakeEmbedder({
        "promissory estoppel against the state": np.array([1.0, 0.0, 0.0]),
        "doctrine of indoor management": np.array([0.0, 1.0, 0.0]),
        "test for vicarious liability": np.array([0.0, 0.0, 1.0]),
    })
    cache = SemanticCache(embedder, max_entries=2)
    cache.put("promissory estoppel against the state", "a1")
    cache.put("doctrine of indoor management", "a2")
    cache.put("test for vicarious liability", "a3")

    assert len(cache) == 2
    # The oldest entry (promissory estoppel) was evicted, so an exact
    # repeat of it can no longer hit even at a trivially low threshold.
    assert cache.get("promissory estoppel against the state") is None


def test_stats_track_hits_and_misses():
    embedder = FakeEmbedder({"q1": np.array([1.0, 0.0]), "q2": np.array([0.0, 1.0])})
    cache = SemanticCache(embedder, similarity_threshold=0.9)
    cache.put("q1", "a1")
    cache.get("q1")  # hit
    cache.get("q2")  # miss

    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == pytest.approx(0.5)
