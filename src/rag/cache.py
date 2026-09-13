"""Semantic cache for repeat and near-duplicate queries.

The obvious version of this embeds every incoming query and returns the
highest-scoring cached answer above a similarity threshold, the same way
dense retrieval matches a query to a chunk. That is the wrong default here
specifically because it is exactly the failure route.py already measured:
"AIR 1974" and "AIR 1006" embed to nearly the same vector despite being
completely different citations, since to an embedding model they both just
mean "a citation to the All India Reporter". A semantic cache built on
embedding similarity alone would serve a query about AIR 1974's cached
answer to a query about AIR 1006, confidently and wrongly, which is worse
than a cache miss: a cache miss costs latency, a wrong cache hit costs
correctness with no visible sign anything went wrong.

So this cache routes the same way retrieval does. Identifier queries match
only on exact normalised string equality, the same guarantee ExactIndex
gives retrieval. Conceptual queries match on cosine similarity against a
high threshold, where two phrasings of the same legal question really do
embed close together and a wrong match is a lower-stakes failure than
misidentifying a citation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from rag.route import classify

# 0.97 is deliberately conservative. Lower thresholds were not swept against
# a labelled set the way chunk size and RRF depth were flagged as unswept in
# retrieval_notes.md; this is a starting point chosen to bias toward missing
# a cacheable duplicate over serving a wrong answer for a different question,
# not a value picked because it produced a good-looking hit rate.
SIMILARITY_THRESHOLD = 0.97

# Unbounded growth was the obvious failure mode to design out up front: a
# long-running server answering thousands of queries should not keep every
# embedding and answer in memory forever. Oldest-first eviction, not an LRU,
# because recency of use is not evidence a cached legal answer is still the
# one worth serving; insertion order is at least legible.
MAX_ENTRIES = 200


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> np.ndarray: ...


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass
class CacheEntry:
    query: str
    kind: str
    payload: Any
    embedding: np.ndarray | None = None
    created: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class SemanticCache:
    """Caches whatever payload a caller wants keyed on a query, routed the
    same way retrieval is: exact match for identifiers, embedding
    similarity for everything else.

    Not a Searcher and not part of the retrieval Protocol stack on purpose.
    generate.answer and the API layer decide whether and when to consult
    it; the cache itself has no opinion on what a "payload" is, so it works
    identically whether the caller is caching an Answer, a dict, or raw
    text.
    """

    def __init__(
        self,
        embedder: QueryEmbedder,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_entries: int = MAX_ENTRIES,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._entries: list[CacheEntry] = []
        self.stats = CacheStats()

    def get(self, query: str) -> Any | None:
        kind = classify(query)
        hit = self._get_identifier(query) if kind == "identifier" else self._get_conceptual(query)
        if hit is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return hit

    def _get_identifier(self, query: str) -> Any | None:
        target = _normalise(query)
        for entry in self._entries:
            if entry.kind == "identifier" and _normalise(entry.query) == target:
                return entry.payload
        return None

    def _get_conceptual(self, query: str) -> Any | None:
        candidates = [e for e in self._entries if e.kind == "conceptual"]
        if not candidates:
            return None
        query_vec = self._embedder.embed_query(query)
        best_score = -1.0
        best_entry: CacheEntry | None = None
        for entry in candidates:
            # Vectors are normalised at embed time (Retriever.embed_query
            # honours the index's own "normalized" flag), so the dot
            # product is the cosine similarity directly, the same
            # shortcut retrieve.py's own rank() takes.
            score = float(np.dot(query_vec, entry.embedding))
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry is not None and best_score >= self._threshold:
            return best_entry.payload
        return None

    def put(self, query: str, payload: Any) -> None:
        kind = classify(query)
        embedding = None if kind == "identifier" else self._embedder.embed_query(query)
        self._entries.append(
            CacheEntry(query=query, kind=kind, payload=payload, embedding=embedding)
        )
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def __len__(self) -> int:
        return len(self._entries)
