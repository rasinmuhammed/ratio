"""Keyword Search: BM25 over chunked text."""

from __future__ import annotations

import math
import re
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
    def __init__(self, texts: list[str], ids: list[str]) -> None:
        if len(texts) != len(ids):
            raise ValueError(f"texts/ids mismatch: {len(texts)} vs {len(ids)}")

        self.ids = ids
        self.n = len(texts)
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        self.lengths: list[int] = []

        self.postings: defaultdict[str, list[int]] = defaultdict(list)

        for i, text in enumerate(texts):
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.lengths.append(len(tokens))
            for term in tf:
                self.doc_freq[term] += 1
                self.postings[term].append(i)

        self.avg_length = sum(self.lengths) / self.n if self.n else 0.0

    def idf(self, term: str) -> float:
        """Rarer terms carry more signal. The +0.5 smoothing is the standard
        BM25 variant and keeps idf positive for very common terms."""
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[Match]:
        """Return the k best matches, best first."""
        terms = tokenize(query)
        if not terms:
            return []

        scores: defaultdict[int, float] = defaultdict(float)
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for i in postings:
                tf = self.term_freqs[i][term]
                norm = 1 - B + B * (self.lengths[i] / self.avg_length)
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [Match(index=i, score=s) for i, s in ranked]

    def __len__(self) -> int:
        return self.n
