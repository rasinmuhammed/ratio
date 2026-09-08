import pytest

from rag.expand import ParentExpandingRetriever
from rag.retrieve import Result


class FakeSemantic:
    """Returns a fixed Result list, mirroring FakeSemantic in
    test_rerank.py: expansion logic should be testable without a real
    index, only chunk_id shape (doc#index) and a by_id lookup matter."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, k=5, **kwargs):
        self.calls.append((query, k, kwargs))
        return self.results[:k]


def _result(chunk_id, doc_id, text, rank=1, score=0.9):
    return Result(rank=rank, score=score, chunk_id=chunk_id, doc_id=doc_id,
                  text=text, metadata={}, score_type="cosine")


def _by_id(*chunks):
    """chunks: (chunk_id, text) pairs, doc_id inferred as the part before '#'."""
    return {
        cid: {"id": cid, "doc_id": cid.rpartition("#")[0], "text": text}
        for cid, text in chunks
    }


def test_expands_to_neighbours_in_document_order():
    by_id = _by_id(
        ("d1#0", "first paragraph"),
        ("d1#1", "middle paragraph, the matched one"),
        ("d1#2", "third paragraph"),
    )
    inner = FakeSemantic([_result("d1#1", "d1", "middle paragraph, the matched one")])
    retriever = ParentExpandingRetriever(inner, by_id, window=1)

    [result] = retriever.search("query")

    assert result.text == "first paragraph\n\nmiddle paragraph, the matched one\n\nthird paragraph"


def test_stops_cleanly_at_document_boundary():
    """The matched chunk is the first in its document: there is no index -1
    to pull in, expansion should use only the neighbours that exist."""
    by_id = _by_id(
        ("d1#0", "first paragraph, the matched one"),
        ("d1#1", "second paragraph"),
    )
    inner = FakeSemantic([_result("d1#0", "d1", "first paragraph, the matched one")])
    retriever = ParentExpandingRetriever(inner, by_id, window=1)

    [result] = retriever.search("query")

    assert result.text == "first paragraph, the matched one\n\nsecond paragraph"


def test_never_pulls_a_sibling_from_a_different_document():
    """d1#1 and d2#1 share an index but not a document. Confirms the lookup
    key is doc-scoped (f"{doc_id}#{i}"), not index alone."""
    by_id = _by_id(
        ("d1#1", "the matched chunk"),
        ("d2#0", "unrelated document, same index shape"),
        ("d2#2", "also unrelated"),
    )
    inner = FakeSemantic([_result("d1#1", "d1", "the matched chunk")])
    retriever = ParentExpandingRetriever(inner, by_id, window=1)

    [result] = retriever.search("query")

    assert result.text == "the matched chunk"  # no siblings exist for d1 at 0 or 2


def test_window_zero_is_a_no_op():
    by_id = _by_id(("d1#0", "only chunk"), ("d1#1", "next chunk"))
    inner = FakeSemantic([_result("d1#0", "d1", "only chunk")])
    retriever = ParentExpandingRetriever(inner, by_id, window=0)

    [result] = retriever.search("query")

    assert result.text == "only chunk"


def test_malformed_chunk_id_passes_through_unchanged():
    """A chunk_id with no '#' cannot be split into (doc_id, index). Rather
    than raise, the hit returns exactly as the inner searcher produced it."""
    by_id: dict = {}
    inner = FakeSemantic([_result("not-a-chunk-id", "d1", "text")])
    retriever = ParentExpandingRetriever(inner, by_id, window=1)

    [result] = retriever.search("query")

    assert result.text == "text"


def test_rank_and_score_are_untouched_by_expansion():
    """The whole premise: matching happens on the precise chunk, so its
    rank and score describe that match. Expansion only changes what text
    a generator reads, never how the hit was scored."""
    by_id = _by_id(("d1#0", "matched"), ("d1#1", "neighbour"))
    inner = FakeSemantic([_result("d1#0", "d1", "matched", rank=3, score=0.72)])
    retriever = ParentExpandingRetriever(inner, by_id, window=1)

    [result] = retriever.search("query")

    assert result.rank == 3
    assert result.score == 0.72


def test_negative_window_rejected():
    with pytest.raises(ValueError):
        ParentExpandingRetriever(FakeSemantic([]), {}, window=-1)


def test_k_and_kwargs_pass_through_to_inner_searcher():
    inner = FakeSemantic([])
    retriever = ParentExpandingRetriever(inner, {}, window=1)

    retriever.search("query", k=8, depth=40)

    assert inner.calls == [("query", 8, {"depth": 40})]
