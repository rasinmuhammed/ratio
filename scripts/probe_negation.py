"""
Can the retriever tell a query from its negation apart? Same probe as before,
now with reranking as a fourth column, since the reranker's entire
justification is that it fixes exactly this.

    uv run python scripts/probe_negation.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from rag.hybrid import HybridRetriever
from rag.keyword import KeywordRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever

INDEX_DIR = Path("data/index-full")
K = 10

PAIRS = [
    ("when will a court grant a temporary injunction",
     "when will a court refuse a temporary injunction"),
    ("when adverse possession is established",
     "when adverse possession cannot be established"),
    ("when bail should be granted",
     "when bail should be denied"),
    ("evidence that is admissible in a criminal trial",
     "evidence that is inadmissible in a criminal trial"),
]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    hybrid = HybridRetriever(INDEX_DIR)
    dense = hybrid.dense
    kw = KeywordRetriever(dense.payloads, hybrid.keyword)
    reranked = RerankedRetriever(hybrid, CrossEncoderReranker())

    configs = {"dense": dense, "bm25": kw, "hybrid": hybrid, "reranked": reranked}

    print(f"{'dense':>7}{'bm25':>7}{'hybrid':>8}{'reranked':>10}   pair")
    print("-" * 100)
    for a, b in PAIRS:
        row = []
        for name, s in configs.items():
            A = {r.chunk_id for r in s.search(a, k=K)}
            B = {r.chunk_id for r in s.search(b, k=K)}
            row.append(len(A & B))
        print(f"{row[0]:>7}{row[1]:>7}{row[2]:>8}{row[3]:>10}   {a[:40]}")

    print(f"\nOverlap out of {K}. Lower is better: it means the retriever can")
    print("tell the query and its negation apart.")


if __name__ == "__main__":
    main()