"""Sweep every retrieval configuration over the labelled query set.

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --index data/index-full \
        --labels data/labels-full.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag.evaluate import LABELS_PATH, evaluate, load_labels, summarise
from rag.hybrid import HybridRetriever
from rag.keyword import KeywordRetriever
from rag.retrieve import Retriever
from rag.route import RoutedRetriever

INDEX_DIR = Path("data/index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="evenly spaced subsample of the label set")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    queries = load_labels(args.labels)
    if args.limit:
        # Deterministic subsample. The full sweep over 7,323 queries takes
        # hours, and a fixed sample answers the same question in minutes.
        queries = queries[::max(1, len(queries) // args.limit)][:args.limit]

    # Everything is derived from one HybridRetriever, so there is exactly one
    # copy of the vectors, the payloads and the BM25 index in memory. Building
    # each config independently costs three of each, which at 414,122 chunks
    # is more memory than this machine has.
    hybrid = HybridRetriever(args.index)
    dense = hybrid.dense

    configs = {
        "dense": dense,
        "bm25": KeywordRetriever(dense.payloads, hybrid.keyword),
        "hybrid 1:1": hybrid,
        "hybrid 1:3": _Weighted(hybrid, keyword_weight=3.0),
        # Hybrid is the semantic branch for now. Which retriever belongs there
        # is a question about conceptual queries, and there are no conceptual
        # labels yet to answer it with.
        "routed": RoutedRetriever(hybrid, dense.payloads),
    }

    print(f"{len(queries)} queries, k={args.k}\n")
    header = f"{'config':<14}{'recall':>9}{'precision':>11}{'MRR':>8}"
    print(header)
    print("-" * len(header))

    for name, searcher in configs.items():
        summary = summarise(evaluate(searcher, queries, k=args.k))["all"]
        print(f"{name:<14}{summary['recall']:>9.3f}"
              f"{summary['precision']:>11.3f}{summary['mrr']:>8.3f}")


class _Weighted:
    """Pins fusion weights so a weighted hybrid satisfies Searcher."""

    def __init__(self, hybrid: HybridRetriever, keyword_weight: float) -> None:
        self.hybrid = hybrid
        self.keyword_weight = keyword_weight

    def search(self, query: str, k: int = 5):
        return self.hybrid.search(query, k=k, keyword_weight=self.keyword_weight)


if __name__ == "__main__":
    main()