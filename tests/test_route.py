
from rag.retrieve import Result
from rag.route import ExactIndex, RoutedRetriever, classify, extract, normalise

CHUNKS = [
    {"id": "c0", "doc_id": "d0", "metadata": {},
     "text": "The appeal cites AIR 1974 Patna 164 at some length."},
    {"id": "c1", "doc_id": "d0", "metadata": {},
     "text": "Reference is made to AIR 1974 and to Section 9 of the Act."},
    {"id": "c2", "doc_id": "d1", "metadata": {},
     "text": "Civil Appeal No.1091 of 2013 was heard on 24.06.2014."},
    {"id": "c3", "doc_id": "d2", "metadata": {},
     "text": "The principles governing an interim injunction are settled."},
    {"id": "c4", "doc_id": "d3", "metadata": {},
     "text": "Section 9 is discussed without any citation."},
]


class FakeSemantic:
    """Returns chunks in a fixed order, so backfill can be tested without
    loading a model."""

    def __init__(self, order):
        self.order = order
        self.calls = 0

    def search(self, query, k=5):
        self.calls += 1
        return [
            Result(rank=i, score=0.5, chunk_id=CHUNKS[j]["id"],
                   doc_id=CHUNKS[j]["doc_id"], text=CHUNKS[j]["text"],
                   metadata={}, score_type="cosine")
            for i, j in enumerate(self.order[:k], start=1)
        ]


class FakeHybridSemantic:
    """Like FakeSemantic, but accepts depth the way HybridRetriever does, so
    the depth-passthrough fix can be tested directly rather than only
    exercising the except-TypeError fallback FakeSemantic above forces."""

    def __init__(self, order):
        self.order = order
        self.last_depth = None

    def search(self, query, k=5, depth=20):
        self.last_depth = depth
        return [
            Result(rank=i, score=0.5, chunk_id=CHUNKS[j]["id"],
                   doc_id=CHUNKS[j]["doc_id"], text=CHUNKS[j]["text"],
                   metadata={}, score_type="cosine")
            for i, j in enumerate(self.order[:k], start=1)
        ]


def test_normalise_collapses_whitespace_inside_identifiers():
    """'No.1091 of 2013' and 'No. 1091  of 2013' are the same reference
    written by two different clerks."""
    assert normalise("No.  1091   of 2013") == "No. 1091 of 2013"


def test_extract_finds_every_pattern():
    found = extract("AIR 1974, Section 9, No.1091 of 2013, dated 24.06.2014")
    assert found == {"AIR 1974", "Section 9", "No.1091 of 2013", "24.06.2014"}


def test_classify_splits_identifier_from_conceptual():
    assert classify("AIR 1974") == "identifier"
    assert classify("grounds for granting an injunction") == "conceptual"


def test_classify_treats_a_question_containing_a_citation_as_identifier():
    """Mixed queries exist. Routing them to the exact path is correct because
    the identifier is a hard constraint; the backfill handles the rest."""
    assert classify("what did the court say in AIR 1974") == "identifier"


def test_exact_lookup_finds_every_chunk_containing_the_string():
    index = ExactIndex([c["text"] for c in CHUNKS])
    assert index.lookup("AIR 1974") == [0, 1]


def test_exact_lookup_intersects_rather_than_unions():
    """A query naming two references wants the chunk discussing both. Union
    would return more results and answer a different question."""
    index = ExactIndex([c["text"] for c in CHUNKS])
    assert index.lookup("AIR 1974 and Section 9") == [1]


def test_exact_lookup_is_empty_without_an_identifier():
    index = ExactIndex([c["text"] for c in CHUNKS])
    assert index.lookup("interim injunction") == []


def test_router_returns_exact_hits_first():
    semantic = FakeSemantic([3, 4, 2])
    router = RoutedRetriever(semantic, CHUNKS)
    results = router.search("AIR 1974", k=5)
    assert [r.chunk_id for r in results[:2]] == ["c0", "c1"]
    assert all(r.score_type == "exact" for r in results[:2])


def test_router_backfills_remaining_slots_semantically():
    semantic = FakeSemantic([3, 4, 2])
    router = RoutedRetriever(semantic, CHUNKS)
    results = router.search("AIR 1974", k=5)
    assert len(results) == 5
    assert [r.chunk_id for r in results[2:]] == ["c3", "c4", "c2"]
    assert all(r.score_type == "cosine" for r in results[2:])


def test_router_never_repeats_a_chunk_across_the_two_paths():
    """The semantic path does not know the exact path already ran, so it can
    and will return the same chunk. Two entries for one chunk would inflate
    precision and hand the model duplicate context."""
    semantic = FakeSemantic([0, 1, 3])
    router = RoutedRetriever(semantic, CHUNKS)
    results = router.search("AIR 1974", k=3)
    assert [r.chunk_id for r in results] == ["c0", "c1", "c3"]


def test_router_skips_exact_lookup_for_conceptual_queries():
    semantic = FakeSemantic([3, 2, 4])
    router = RoutedRetriever(semantic, CHUNKS)
    results = router.search("principles governing an injunction", k=2)
    assert [r.chunk_id for r in results] == ["c3", "c2"]
    assert all(r.score_type == "cosine" for r in results)


def test_router_ranks_are_contiguous_from_one():
    """evaluate.py reads order, not rank, but generate.answer numbers its
    sources from these, so a gap becomes a wrong citation."""
    semantic = FakeSemantic([3, 4, 2])
    router = RoutedRetriever(semantic, CHUNKS)
    results = router.search("AIR 1974", k=4)
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_router_honours_k_smaller_than_the_exact_hit_count():
    semantic = FakeSemantic([3])
    router = RoutedRetriever(semantic, CHUNKS)
    results = router.search("AIR 1974", k=1)
    assert [r.chunk_id for r in results] == ["c0"]
    assert semantic.calls == 0


def test_router_requests_depth_explicitly_when_the_backend_supports_it():
    """RerankedRetriever had this exact bug: passing k alone to a
    HybridRetriever-shaped backend leaves the pool capped by its own
    CANDIDATE_DEPTH default, silently shallower than what k asked for.
    Backfill must request depth=candidate_n explicitly, not rely on k."""
    semantic = FakeHybridSemantic([3, 4, 1, 2])
    router = RoutedRetriever(semantic, CHUNKS)
    router.search("principles governing an injunction", k=3)
    # No exact hits for this query, so candidate_n == k + 0 == 3.
    assert semantic.last_depth == 3


def test_router_falls_back_when_the_backend_does_not_accept_depth():
    """Plain Retriever and KeywordRetriever take no depth argument at all;
    the TypeError fallback must still return results, not raise."""
    semantic = FakeSemantic([3, 2])
    router = RoutedRetriever(semantic, CHUNKS)
    results = router.search("principles governing an injunction", k=2)
    assert [r.chunk_id for r in results] == ["c3", "c2"]
