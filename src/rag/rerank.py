"""Cross-encoder reranking: fix what bi-encoders structurally cannot see.

Every retriever elsewhere in this repo is a bi-encoder: query and passage are
each turned into a fixed representation *before* they meet, then compared
with a dot product or a token overlap. Measured directly, that architecture
cannot tell "when adverse possession is established" from "when adverse
possession cannot be established" apart: dense and BM25 both return roughly
the same top-10 for a query and its negation, because the negation lives in a
function word neither representation preserves.

A cross-encoder feeds [query, passage] through one model together, so every
query token can attend to every passage token in the same forward pass before
a score exists. It can see "cannot" sitting next to "established" in the
passage while it is still reading the query. That is structural, not a tuning
improvement, which is why it is the standard fix for this failure mode.

The cost is why this is a rerank step and not a first-pass retriever: scoring
is quadratic in combined length and nothing about a passage can be
precomputed, so running it over 414,122 chunks per query is not viable.
Running it over the ~50 candidates hybrid already found is.
"""

from __future__ import annotations

from typing import Protocol

from rag.retrieve import DEFAULT_K, Result

# bge-reranker-base: a cross-encoder fine-tuned for passage relevance, small
# enough to run on CPU. Not a legal-domain model, this is a general reranker,
# so it is not expected to fix everything the negation probe found. It is
# expected to fix the part that is architectural: recognising when the query
# and the passage disagree on polarity of the same proposition.
DEFAULT_RERANKER = "BAAI/bge-reranker-base"

# Candidates pulled from the underlying retriever before reranking. Deeper
# than CANDIDATE_DEPTH in hybrid.py (20), because the whole point is to give
# the reranker a chance to promote something hybrid buried. The k=20 rerun in
# retrieval_notes.md found 19% of relevant chunks sitting at rank 6-20 and
# never surfaced; 50 gives real room above that.
RERANK_DEPTH = 50

# Combined query+passage token budget. A chunk runs to roughly 450 tokens on
# its own, so this determines how much of a long passage the cross-encoder
# actually reads. The model's own tokenizer truncates to this, silently, so
# it is stated here rather than left implicit.
MAX_LENGTH = 512


class Reranker(Protocol):
    """Anything that scores (query, passage) pairs. Protocol so
    RerankedRetriever never depends on a specific model or provider,
    the same reason Searcher and LLM are Protocols elsewhere in this repo."""

    def score(self, query: str, texts: list[str]) -> list[float]: ...

class CrossEncoderReranker:
    """Wraps sentence_transformers.CrossEncoder behind the Reranker protocol.

    The model load is the expensive part (a few hundred MB), so it happens
    once in __init__ and the instance is meant to be shared across queries,
    the same pattern Retriever uses for the embedding model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        max_length: int = MAX_LENGTH,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]

class RerankedRetriever:
    """Retrieve deep with an existing Searcher, then reorder by cross-encoder
    score and truncate to k.

    Satisfies the same Searcher interface as Retriever, HybridRetriever and
    RoutedRetriever, so it drops in anywhere those do: as the semantic branch
    inside RoutedRetriever, or standalone in the evaluation sweep.
    """

    def __init__(
        self,
        semantic,
        reranker: Reranker,
        candidate_k: int = RERANK_DEPTH,
    ) -> None:
        self.semantic = semantic
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query: str, k: int = DEFAULT_K) -> list[Result]:
        # Pull at least candidate_k, but never fewer than k: a caller asking
        # for more results than the configured depth should not silently get
        # truncated to the depth instead.
        candidate_n = max(self.candidate_k, k)

        try:
            # HybridRetriever's fusion pool is capped by its own `depth`
            # parameter (default 20), separate from `k`. Passing only k=50
            # asks hybrid for 50 results from a pool that depth still limits
            # to ~40 fused candidates, so the reranker was silently reordering
            # a shallower pool than RERANK_DEPTH implied. depth has to be
            # requested explicitly to actually get a deep pool.
            candidates = self.semantic.search(query, k=candidate_n, depth=candidate_n)
        except TypeError:
            # Plain Retriever and KeywordRetriever take no depth argument.
            candidates = self.semantic.search(query, k=candidate_n)

        if not candidates:
            return []

        scores = self.reranker.score(query, [c.text for c in candidates])

        order = sorted(range(len(candidates)), key=lambda i: -scores[i])[:k]

        return [
            Result(
                rank=position,
                score=scores[i],
                chunk_id=candidates[i].chunk_id,
                doc_id=candidates[i].doc_id,
                text=candidates[i].text,
                metadata=candidates[i].metadata,
                # Named distinctly from "cosine" or "rrf": a cross-encoder
                # score is not a similarity in the same units as either, and
                # collapsing the name would hide that, the same mistake
                # score_type exists to prevent elsewhere.
                score_type="reranked",
            )
            for position, i in enumerate(order, start=1)
        ]
