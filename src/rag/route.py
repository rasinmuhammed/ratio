"""Route a query to the retriever that can actually answer it.

Measured on the full index, the query `AIR 1006` breaks down like this:

    air                 41,382 chunks   9.99%
    1006                   151 chunks    0.04%
    the phrase "AIR 1006"    2 chunks

Neither token identifies anything. The information is in their adjacency, and
adjacency is exactly what BM25 throws away when it tokenises. Dense retrieval
never had it either: to an embedding model `AIR 1006` and `AIR 1974` mean the
same thing, which is "a citation to the All India Reporter".

So this is not a ranking problem that needs a better ranker. It is a lookup
wearing a ranking problem's clothes, and the fix is to stop sending it to the
rankers. Reciprocal rank fusion cannot help here for a structural reason: it
averages two systems that are both wrong, which is why every hybrid weighting
tested landed between dense and BM25 and never above BM25.

Identifiers go to an exact index. Everything else goes to the semantic path.
Where a query carries both, the exact hits are placed first and the semantic
results fill what is left, because an identifier is a hard constraint and
topical similarity is a soft one.
"""

from __future__ import annotations

import math
import re
from array import array
from typing import Any

from rag.retrieve import DEFAULT_K, Result

COURT_WEIGHT = {
    "supreme_court": 1.00,
    "high_court_db": 0.85,
    "high_court_sb": 0.75,
    "tribunal":      0.60,
}

def authority_boost(result: Result) -> float:
    """Calculates a score multiplier based on court hierarchy and citation count."""
    court_type = result.metadata.get("court", "").lower()
    # Handle missing cited_by gracefully (default 1)
    cited_by = result.metadata.get("cited_by", 1)
    if not isinstance(cited_by, (int, float)):
        try:
            cited_by = int(cited_by)
        except (ValueError, TypeError):
            cited_by = 1
            
    # Default to 0.70 for unknown courts
    base = 0.70
    for key, weight in COURT_WEIGHT.items():
        if key in court_type.replace(" ", "_"):
            base = weight
            break
            
    cite_boost = min(math.log1p(max(0, cited_by)) / math.log1p(1000), 0.3)
    return base + cite_boost

# The same patterns that build_labels.py greps for. That is deliberate: the
# label set defines a relevant chunk as one containing the literal string, so
# anything these patterns match is answerable exactly and should never have
# been handed to a statistical ranker.
PATTERNS: dict[str, str] = {
    "citation": r"\bAIR \d{4}\b",
    "case_number": r"\bNo\.\s?\d+ of \d{4}\b",
    "section": r"\bSection \d+[A-Z]?\b",
    "date": r"\b\d{2}[./-]\d{2}[./-]\d{4}\b",
}

_COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}

# Whitespace inside an identifier varies between documents: "No.1091 of 2013"
# and "No. 1091 of 2013" are the same reference. Collapsing runs of space is
# enough to make them agree, and it is the only normalisation applied, because
# anything more aggressive starts merging references that are genuinely
# different.
_SPACE = re.compile(r"\s+")


def normalise(identifier: str) -> str:
    return _SPACE.sub(" ", identifier).strip()


def extract(text: str) -> set[str]:
    """Every identifier in a piece of text, normalised."""
    found = set()
    for pattern in _COMPILED.values():
        for match in pattern.finditer(text):
            found.add(normalise(match.group(0)))
    return found


def classify(query: str) -> str:
    """`identifier` if the query contains something exactly matchable.

    Note that this is not exclusive. "what does Section 9 mean" is classified
    as an identifier query and is also a real question, which is why the
    router backfills rather than routing to one path and stopping.

    >>> classify("AIR 1974")
    'identifier'
    >>> classify("grounds for granting an injunction")
    'conceptual'
    """
    return "identifier" if extract(query) else "conceptual"


class ExactIndex:
    """Identifier to the chunks containing it.

    Built by extracting identifiers at index time rather than scanning the
    corpus per query. A linear substring scan over 414,122 chunks costs about
    a second each time and does the same work repeatedly; the patterns are
    known ahead of time, so the work belongs at build time.

    Storage matches BM25Index: postings in array.array rather than lists of
    boxed integers, since the corpus carries roughly half a million
    identifier occurrences.
    """

    def __init__(self, texts: list[str]) -> None:
        self.postings: dict[str, array] = {}
        for i, text in enumerate(texts):
            for identifier in extract(text):
                entry = self.postings.get(identifier)
                if entry is None:
                    entry = array("i")
                    self.postings[identifier] = entry
                entry.append(i)

    def lookup(self, query: str) -> list[int]:
        """Chunk indices containing every identifier in the query.

        Intersection, not union. A query naming two references wants the chunk
        discussing both, and returning either one separately would be a
        strictly worse answer that also looks confident.
        """
        identifiers = extract(query)
        if not identifiers:
            return []

        hits: set[int] | None = None
        for identifier in identifiers:
            entry = self.postings.get(identifier)
            found = set(entry) if entry is not None else set()
            hits = found if hits is None else (hits & found)
            if not hits:
                return []
        return sorted(hits or ())

    def __len__(self) -> int:
        return len(self.postings)


class RoutedRetriever:
    """Exact lookup first, semantic retrieval for the rest.

    Satisfies the same Searcher interface as Retriever and HybridRetriever, so
    it drops into the evaluation sweep and into generate.answer unchanged.
    """

    def __init__(self, semantic: Any, payloads: list[dict],
                 exact: ExactIndex | None = None) -> None:
        self.semantic = semantic
        self.payloads = payloads
        self.exact = exact if exact is not None else ExactIndex(
            [p["text"] for p in payloads]
        )

    def search(self, query: str, k: int = DEFAULT_K) -> list[Result]:
        exact_hits = self.exact.lookup(query)[:k]

        results = [
            self._result(index, rank, score=1.0)
            for rank, index in enumerate(exact_hits, start=1)
        ]
        if len(results) >= k:
            return results

        # Backfill. An identifier query that matches two chunks still leaves
        # three slots, and topically similar text is a better use of them than
        # nothing. Semantic results that duplicate an exact hit are dropped,
        # otherwise the same chunk appears twice with two different scores.
        candidate_n = k + len(results)
        try:
            # Same bug RerankedRetriever had: HybridRetriever's own fusion
            # pool is capped by its own CANDIDATE_DEPTH (default 20)
            # regardless of what k asks for. Passing k alone silently caps
            # the pool below what was actually requested; depth has to be
            # asked for explicitly. Latent here until conceptual queries
            # started running through this path in volume, same as the
            # RerankedRetriever version of this bug was latent until it was
            # measured directly.
            candidates = self.semantic.search(query, k=candidate_n, depth=candidate_n)
        except TypeError:
            # Plain Retriever and KeywordRetriever take no depth argument.
            candidates = self.semantic.search(query, k=candidate_n)

        # Apply authority-weighted ranking
        boosted_candidates = []
        for c in candidates:
            boost = authority_boost(c)
            boosted = Result(
                rank=0, # Recalculated below
                score=c.score * boost,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                metadata=c.metadata,
                score_type=c.score_type,
            )
            boosted_candidates.append(boosted)
            
        boosted_candidates.sort(key=lambda x: x.score, reverse=True)

        seen = {r.chunk_id for r in results}
        for candidate in boosted_candidates:
            if candidate.chunk_id in seen:
                continue
            results.append(Result(
                rank=len(results) + 1,
                score=candidate.score,
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                text=candidate.text,
                metadata=candidate.metadata,
                score_type=candidate.score_type,
            ))
            if len(results) >= k:
                break
        return results

    def _result(self, index: int, rank: int, score: float) -> Result:
        payload = self.payloads[index]
        return Result(
            rank=rank,
            score=score,
            chunk_id=payload["id"],
            doc_id=payload["doc_id"],
            text=payload["text"],
            metadata=payload["metadata"],
            # Named rather than reusing "cosine" or "bm25". An exact hit is
            # not a similarity at all, it is a boolean, and 1.0 would otherwise
            # read as an extremely confident cosine.
            score_type="exact",
        )
