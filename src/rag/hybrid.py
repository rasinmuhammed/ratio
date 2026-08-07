"""Hybrid retrieval: fuse dense and keyword rankings."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rag.embed import INDEX_DIR
from rag.keyword import BM25Index
from rag.retrieve import DEFAULT_K, Result, Retriever

# Damping constant. Larger k flattens the difference between rank 1 and 2,
# so no single system can dominate the fusion. 60 is the value from the
# original RRF paper and is the usual default.
RRF_K = 60

# How deep to go in each ranking before fusing. Too shallow and a document
# ranked 15th by one system never gets a chance to be promoted.
CANDIDATE_DEPTH = 20


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Combine rankings by position, deliberately discarding the scores.

    Dense scores sit in a narrow 0.5-0.8 band. BM25 scores are unbounded and
    depend on term rarity. Averaging them is meaningless, and normalising
    requires knowing each distribution per query. Rank is always comparable.

    `weights` exists because plain RRF assumes both systems are equally
    trustworthy. Measured on this corpus that is false for exact-match
    queries, where dense retrieval is not merely weaker but actively wrong,
    and its confident-but-irrelevant results dilute BM25's correct ones.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights/rankings mismatch: {len(weights)} vs {len(rankings)}")

    fused: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] += weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.dense = Retriever(index_dir)
        payloads = self.dense.payloads
        self.by_id = {p["id"]: p for p in payloads}
        self.keyword = BM25Index(
            [p["text"] for p in payloads],
            [p["id"] for p in payloads],
        )

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        depth: int = CANDIDATE_DEPTH,
        dense_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> list[Result]:
        dense_ids = [r.chunk_id for r in self.dense.search(query, k=depth)]
        kw_ids = [
            self.keyword.ids[m.index] for m in self.keyword.search(query, k=depth)
        ]

        fused = reciprocal_rank_fusion(
            [dense_ids, kw_ids], weights=[dense_weight, keyword_weight]
        )[:k]

        results = []
        for position, (chunk_id, score) in enumerate(fused, start=1):
            p = self.by_id.get(chunk_id)
            if p is None:
                # Fusion can only emit ids that came from one of the input
                # rankings, so this is a broken invariant rather than bad input.
                raise KeyError(f"fused id {chunk_id!r} is not in the payloads")
            results.append(Result(
                rank=position,
                score=float(score),
                chunk_id=p["id"],
                doc_id=p["doc_id"],
                text=p["text"],
                metadata=p["metadata"],
                score_type="rrf",
            ))
        return results  