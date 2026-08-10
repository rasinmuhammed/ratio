"""Keyword Search: BM25 over chunked text."""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter, defaultdict
from typing import NamedTuple

# k1 controls how fast term frequency saturates: with k1=1.5, one occurrence
# gives 0.4 of the maximum, two gives 0.57, eight gives 0.84.
K1 = 1.5

# b controls length normalisation. 0 ignores length entirely, 1 fully divides
# by relative length. 0.75 is the standard compromise.
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")


class Match(NamedTuple):
    """A scored hit. Named rather than a bare tuple so callers write
    match.index and match.score instead of match[0] and match[1]."""

    index: int
    score: float

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics.

    Digits are kept as tokens deliberately: '1974', '1091', '138' are exactly
    the identifiers dense retrieval cannot handle, and they are the reason
    this module exists.
    """
    return _TOKEN.findall(text.lower())

class BM25Index:
    """An inverted index: term -> the chunks containing it, with counts.

    The obvious layout keeps a Counter per chunk *and* a postings list per
    term. That stores the same fact twice, and at corpus scale it decides
    whether the index fits in memory: measured on 200 documents it projects to
    3.7 GB across the full 10,588, before the payloads and the embedding model
    are accounted for.

    So term frequency lives in the postings, beside the chunk index it belongs
    to, in two parallel array.array buffers. A list of ints costs a pointer
    plus a boxed integer per element; an array stores the machine integer and
    nothing else.
    """

    def __init__(self, texts: list[str], ids: list[str]) -> None:
        if len(texts) != len(ids):
            raise ValueError(f"texts/ids mismatch: {len(texts)} vs {len(ids)}")

        self.ids = ids
        self.n = len(texts)
        self.lengths: list[int] = []

        # term -> (chunk indices as 'i', term frequencies as 'H'). 'H' caps a
        # count at 65,535; the chunker caps a chunk at 450 tokens, so a single
        # term cannot come near it.
        self.postings: dict[str, tuple[array, array]] = {}

        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self.lengths.append(len(tokens))

            # The Counter is transient, built and dropped once per chunk, so
            # the peak is one Counter rather than one per chunk. That is where
            # the saving comes from.
            for term, count in Counter(tokens).items():
                entry = self.postings.get(term)
                if entry is None:
                    entry = (array("i"), array("H"))
                    self.postings[term] = entry
                entry[0].append(i)
                entry[1].append(count)

        self.avg_length = sum(self.lengths) / self.n if self.n else 0.0

    def idf(self, term: str) -> float:
        """Rarer terms carry more signal. The +0.5 smoothing is the standard
        BM25 variant and keeps idf positive for very common terms.

        Document frequency is not stored separately: it is the length of the
        postings list, which is the same number by construction.
        """
        entry = self.postings.get(term)
        df = len(entry[0]) if entry is not None else 0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[Match]:
        """Return the k best matches, best first."""
        terms = tokenize(query)
        if not terms:
            return []

        scores: defaultdict[int, float] = defaultdict(float)
        for term in terms:
            entry = self.postings.get(term)
            if entry is None:
                continue
            indices, freqs = entry
            idf = self.idf(term)
            for i, tf in zip(indices, freqs):
                norm = 1 - B + B * (self.lengths[i] / self.avg_length)
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [Match(index=i, score=s) for i, s in ranked]

    def __len__(self) -> int:
        return self.n

class KeywordRetriever:
    """BM25 behind the same interface as Retriever and HybridRetriever.
    """
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.index = BM25Index([p["text"] for p in payloads], [p["id"] for p in payloads])

    def search(self, query: str, k: int = 5):
        from rag.retrieve import Result

        results = []
        for rank, match in enumerate(self.index.search(query, k=k), start=1):
            p = self.payloads[match.index]
            results.append(Result(
                rank=rank,
                score=float(match.score),
                chunk_id=p["id"],
                doc_id=p["doc_id"],
                text=p["text"],
                metadata=p["metadata"],
                score_type="bm25",
            ))

        return results