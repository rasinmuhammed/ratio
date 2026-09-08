import numpy as np
import pytest

from rag.hyde import HydeRetriever


class FakeLLM:
    """Records what it was asked and returns a fixed hypothetical passage,
    the same shape as FakeReranker in test_rerank.py: a test controls the
    outcome instead of hoping a real model cooperates."""

    def __init__(self, passage: str):
        self.passage = passage
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.passage


class FakeModel:
    """Records exactly what text it was asked to embed, and returns a
    preset vector for that text, or a zero vector for anything else so an
    accidental extra call is obviously wrong rather than silently scored."""

    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self.vectors_by_text = vectors_by_text
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, normalize_embeddings: bool = True):
        self.calls.append((text, normalize_embeddings))
        return np.array(self.vectors_by_text.get(text, [0.0, 0.0]), dtype=np.float32)


class FakeDense:
    def __init__(self, model, vectors, payloads, normalized=True):
        self.model = model
        self.vectors = vectors
        self.payloads = payloads
        self.meta = {"normalized": normalized}


PAYLOADS = [
    {"id": "d1#0", "doc_id": "d1", "text": "about bail conditions",
     "metadata": {}},
    {"id": "d2#0", "doc_id": "d2", "text": "about limitation periods",
     "metadata": {}},
]
VECTORS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_embeds_the_hypothetical_passage_not_the_raw_query():
    """The whole mechanism: the raw query string should never reach the
    embedding model. Only the LLM-generated passage should."""
    model = FakeModel({"a fabricated ruling on bail": [1.0, 0.0]})
    dense = FakeDense(model, VECTORS, PAYLOADS)
    llm = FakeLLM("a fabricated ruling on bail")
    retriever = HydeRetriever(dense, llm)

    retriever.search("grounds for anticipatory bail")

    assert model.calls == [("a fabricated ruling on bail", True)]


def test_does_not_use_the_query_prefix():
    """Retriever.embed_query prepends QUERY_PREFIX for real queries. A
    hypothetical passage is playing the role of a document, embedding it
    with the query-side instruction would reintroduce the register
    mismatch HyDE exists to remove, so it must be encoded bare."""
    model = FakeModel({"plain passage": [1.0, 0.0]})
    dense = FakeDense(model, VECTORS, PAYLOADS)
    llm = FakeLLM("plain passage")
    retriever = HydeRetriever(dense, llm)

    retriever.search("some query")

    (encoded_text, _), = model.calls
    assert encoded_text == "plain passage"
    assert "Represent this sentence" not in encoded_text


def test_ranks_by_similarity_to_the_hypothetical_not_the_query():
    """The hypothetical's vector is [0, 1], matching d2 exactly and d1 not
    at all, even though the raw query text is about bail (d1's topic).
    If search ranked by the query's own meaning this would fail."""
    model = FakeModel({"a passage about limitation": [0.0, 1.0]})
    dense = FakeDense(model, VECTORS, PAYLOADS)
    llm = FakeLLM("a passage about limitation")
    retriever = HydeRetriever(dense, llm)

    results = retriever.search("grounds for anticipatory bail", k=2)

    assert results[0].chunk_id == "d2#0"
    assert results[0].rank == 1


def test_result_shape_matches_payload_fields():
    model = FakeModel({"passage": [1.0, 0.0]})
    dense = FakeDense(model, VECTORS, PAYLOADS)
    llm = FakeLLM("passage")
    retriever = HydeRetriever(dense, llm)

    [top, *_] = retriever.search("query", k=1)

    assert top.chunk_id == "d1#0"
    assert top.doc_id == "d1"
    assert top.text == "about bail conditions"
    assert top.score_type == "cosine"


def test_empty_query_raises():
    dense = FakeDense(FakeModel({}), VECTORS, PAYLOADS)
    retriever = HydeRetriever(dense, FakeLLM("x"))

    with pytest.raises(ValueError):
        retriever.search("   ")


def test_hypothetical_prompt_carries_the_real_query():
    """The LLM has to see the actual query to write a relevant fabrication,
    only the embedding step skips the raw query, not generation."""
    model = FakeModel({"passage": [1.0, 0.0]})
    dense = FakeDense(model, VECTORS, PAYLOADS)
    llm = FakeLLM("passage")
    retriever = HydeRetriever(dense, llm)

    retriever.search("grounds for anticipatory bail")

    [(system, user)] = llm.calls
    assert "grounds for anticipatory bail" in user
    assert "judgment" in system.lower()
