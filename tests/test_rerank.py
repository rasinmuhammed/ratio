from rag.rerank import RerankedRetriever
from rag.retrieve import Result


class FakeSemantic:
    """Returns chunks in a fixed order, mirroring FakeSemantic in
    test_route.py: RerankedRetriever's logic should be testable without
    loading an embedding model, exactly like the router was."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, k=5):
        self.calls.append(k)
        return [
            Result(rank=i, score=0.5, chunk_id=c["id"], doc_id=c["id"],
                   text=c["text"], metadata={}, score_type="cosine")
            for i, c in enumerate(self.chunks[:k], start=1)
        ]


class FakeReranker:
    """Scores by lookup rather than by loading a model, so a test controls
    the outcome instead of hoping a real cross-encoder agrees with it."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, texts))
        return [self.scores.get(t, 0.0) for t in texts]


CHUNKS = [
    {"id": "c0", "text": "low relevance passage"},
    {"id": "c1", "text": "the passage that actually answers it"},
    {"id": "c2", "text": "medium relevance passage"},
]


def test_reranker_reorders_by_score_not_by_retrieval_rank():
    """The entire point. c1 arrives ranked last from the semantic retriever
    and must leave ranked first, because the cross-encoder score, not the
    bi-encoder rank, decides order."""
    semantic = FakeSemantic(CHUNKS)
    reranker = FakeReranker({
        "low relevance passage": 0.1,
        "the passage that actually answers it": 0.9,
        "medium relevance passage": 0.5,
    })
    results = RerankedRetriever(semantic, reranker).search("q", k=3)
    assert [r.chunk_id for r in results] == ["c1", "c2", "c0"]


def test_results_truncate_to_k_after_reranking():
    """Truncation happens after scoring, not before. Truncating first would
    let the semantic retriever's ranking decide who never gets a chance to
    be promoted, which defeats the purpose of reranking at all."""
    semantic = FakeSemantic(CHUNKS)
    reranker = FakeReranker({"low relevance passage": 0.9})
    results = RerankedRetriever(semantic, reranker, candidate_k=3).search("q", k=1)
    assert [r.chunk_id for r in results] == ["c0"]


def test_requests_at_least_candidate_k_from_the_semantic_retriever():
    """Reranking a shallow pool cannot promote anything the pool never had.
    RERANK_DEPTH exists specifically to go deeper than hybrid's own
    CANDIDATE_DEPTH, so the request has to actually reach that far."""
    semantic = FakeSemantic(CHUNKS)
    RerankedRetriever(semantic, FakeReranker({}), candidate_k=50).search("q", k=3)
    assert semantic.calls == [50]


def test_k_larger_than_candidate_k_is_not_silently_shrunk():
    semantic = FakeSemantic(CHUNKS)
    RerankedRetriever(semantic, FakeReranker({}), candidate_k=2).search("q", k=10)
    assert semantic.calls == [10]


def test_ranks_are_contiguous_from_one_after_reordering():
    """generate.build_prompt numbers sources from rank, so a gap here
    becomes a wrong citation downstream, the same invariant test_route.py
    pins for the router."""
    semantic = FakeSemantic(CHUNKS)
    reranker = FakeReranker({c["text"]: i for i, c in enumerate(CHUNKS)})
    results = RerankedRetriever(semantic, reranker).search("q", k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_empty_candidates_do_not_reach_the_reranker():
    """No text to score is not a zero-length batch to send to a model, it is
    nothing to do at all. Sending it anyway would be a wasted model call on
    every out-of-corpus query."""
    semantic = FakeSemantic([])
    reranker = FakeReranker({})
    assert RerankedRetriever(semantic, reranker).search("q", k=5) == []
    assert reranker.calls == []


def test_score_type_is_named_reranked_not_inherited_from_the_source():
    """A cross-encoder score is not in the same units as cosine or rrf.
    Inheriting score_type from the candidate would silently mislabel it,
    the mistake score_type exists to prevent."""
    semantic = FakeSemantic(CHUNKS[:1])
    results = RerankedRetriever(semantic, FakeReranker({})).search("q", k=1)
    assert results[0].score_type == "reranked"