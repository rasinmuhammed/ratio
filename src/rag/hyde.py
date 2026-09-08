"""HyDE: Hypothetical Document Embeddings.

A question and its answer are often written in different registers: a query
asks, a judgment rules. "Grounds for anticipatory bail" and "having regard
to the nature and gravity of the accusation... this Court is inclined to
grant anticipatory bail" are about the same thing, but an embedding model
is sensitive to that surface mismatch, not just the underlying meaning, so
a question-shaped query can sit further from an answer-shaped passage than
the actual overlap in meaning would suggest.

HyDE closes that gap by never embedding the raw query at all. It first asks
an LLM to write a short, fabricated passage, in the corpus's own register,
that would answer the query, then embeds *that* passage and searches with
it. The fabrication does not need to be factually correct, and it is never
shown to anyone, only its embedding is used. What does the work is the
register match: a hypothetical ruling, however wrong its invented details,
embeds close to real rulings, which is exactly the gap a bare question does
not close on its own.

This is a query-time technique. Nothing here touches the index, the stored
vectors, or the chunks, unlike contextual retrieval, which changes what was
embedded. HyDE only changes what gets embedded *for one query, on the fly*,
which is what makes it cheap enough to try without committing to anything.
"""

from __future__ import annotations

from typing import Protocol

from rag.retrieve import DEFAULT_K, Result, rank


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class DenseIndex(Protocol):
    """What HydeRetriever actually needs from a dense Retriever: the model
    to embed with, the vectors to rank against, the payloads to build
    Results from, and whether those vectors were normalized. Structural,
    not rag.retrieve.Retriever specifically, so a test double needs none of
    Retriever's disk/model-loading machinery to stand in for one."""

    model: object
    vectors: object
    payloads: list[dict]
    meta: dict


SYSTEM_PROMPT = (
    "You write a short excerpt from an Indian court judgment that would "
    "answer the question below, in the style an actual judgment is "
    "written: formal, stating legal reasoning, stating a holding. "
    "Invented case names, section numbers or facts are fine and expected, "
    "this text is never shown to anyone or treated as true, it exists "
    "only to search for the real passage that says this. Write 3-5 "
    "sentences, no preamble, no disclaimer, no meta-commentary about what "
    "you are doing."
)


class HydeRetriever:
    """Wraps a dense index. Generates a hypothetical passage for the query,
    embeds it the way a *document* is embedded at index time, plain
    model.encode with no QUERY_PREFIX, since a hypothetical passage plays
    the role of a passage, not a query, and running it through the
    query-side instruction would reintroduce the exact register mismatch
    this technique exists to remove, then ranks against the same vectors
    the dense retriever already holds.
    """

    def __init__(self, dense: DenseIndex, llm: LLM) -> None:
        self.dense = dense
        self.llm = llm

    def hypothetical(self, query: str) -> str:
        return self.llm.complete(SYSTEM_PROMPT, query)

    def search(self, query: str, k: int = DEFAULT_K) -> list[Result]:
        if not query or not query.strip():
            raise ValueError("empty query")

        passage = self.hypothetical(query)
        vector = self.dense.model.encode(
            passage, normalize_embeddings=self.dense.meta.get("normalized", True),
        )
        idx, scores = rank(vector, self.dense.vectors, k)

        results = []
        for position, (i, score) in enumerate(zip(idx, scores), start=1):
            payload = self.dense.payloads[int(i)]
            results.append(Result(
                rank=position,
                score=float(score),
                chunk_id=payload["id"],
                doc_id=payload["doc_id"],
                text=payload["text"],
                metadata=payload["metadata"],
                score_type="cosine",
            ))
        return results
