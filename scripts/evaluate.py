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
from rag.generate import get_llm
from rag.hybrid import HybridRetriever
from rag.hyde import HydeRetriever
from rag.keyword import KeywordRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever
from rag.route import RoutedRetriever

INDEX_DIR = Path("data/index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="evenly spaced subsample of the label set")
    parser.add_argument("--hyde", action="store_true",
                        help="add the HyDE config to the sweep. Off by "
                             "default: unlike every other config here, this "
                             "one costs an LLM call per query, real spend "
                             "on a query set this size, not something to "
                             "pay for on every routine run.")
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
    reranker = CrossEncoderReranker()

    dense = hybrid.dense

    configs = {
        "dense": dense,
        "bm25": KeywordRetriever(dense.payloads, hybrid.keyword),
        "hybrid 1:1": hybrid,
        "hybrid 1:3": _Weighted(hybrid, keyword_weight=3.0),
        # Hybrid is the semantic branch for now. Which retriever belongs
        # there is a question about conceptual queries specifically, now
        # answerable: see the "conceptual" breakdown below, and --hyde.
        "reranked": RerankedRetriever(hybrid, reranker),
        "routed": RoutedRetriever(hybrid, dense.payloads),
    }
    if args.hyde:
        # Compared against "dense" specifically, not hybrid or reranked:
        # HyDE only changes the query embedding fed to the dense branch, so
        # the honest comparison is dense-vs-dense, everything else held
        # fixed, not a confound with fusion or reranking on top of it.
        configs["hyde"] = HydeRetriever(dense, get_llm())

    present_kinds = sorted({q.kind for q in queries})
    kinds = present_kinds + (["all"] if len(present_kinds) > 1 else [])
    n_by_kind = {k: sum(1 for q in queries if q.kind == k) for k in present_kinds}

    print(f"{len(queries)} queries ({n_by_kind}), k={args.k}\n")

    # evaluate() once per searcher, not once per (searcher, kind): HyDE costs
    # a real LLM call per query, and summarise() already splits one run's
    # results by kind, so re-running the search to get a different slice of
    # the same numbers would silently multiply HyDE's cost by len(kinds).
    summaries = {
        name: summarise(evaluate(searcher, queries, k=args.k))
        for name, searcher in configs.items()
    }

    for kind in kinds:
        header = f"{'config':<14}{'recall':>9}{'precision':>11}{'MRR':>8}"
        print(f"-- {kind} --")
        print(header)
        print("-" * len(header))
        for name in configs:
            summary = summaries[name][kind]
            print(f"{name:<14}{summary['recall']:>9.3f}"
                  f"{summary['precision']:>11.3f}{summary['mrr']:>8.3f}")
        print()


class _Weighted:
    """Pins fusion weights so a weighted hybrid satisfies Searcher."""

    def __init__(self, hybrid: HybridRetriever, keyword_weight: float) -> None:
        self.hybrid = hybrid
        self.keyword_weight = keyword_weight

    def search(self, query: str, k: int = 5):
        return self.hybrid.search(query, k=k, keyword_weight=self.keyword_weight)


if __name__ == "__main__":
    main()