"""Pool retrieval candidates for hand annotation.

Judging 414,122 chunks against every query is not possible, so the pool is the
union of what several retrievers return. This is standard TREC pooling, and it
carries a bias worth stating rather than discovering later: a chunk that no
retriever surfaced can never be labelled relevant, so recall is measured
against the pool and not against the corpus. Add a retriever later and its
genuinely novel finds are scored as mistakes until the pool is rebuilt.

    uv run python scripts/pool_candidates.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from rag.hybrid import HybridRetriever
from rag.keyword import KeywordRetriever

INDEX_DIR = Path("data/index-full")
QUERIES = Path("data/conceptual_queries.txt")
OUT = Path("data/pool.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--depth", type=int, default=15,
                        help="per retriever, before deduplication")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    queries = [
        line.strip() for line in args.queries.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    # One HybridRetriever, and the others borrow its components. Constructing
    # them separately means three copies of a 414,122 chunk index.
    hybrid = HybridRetriever(args.index)
    dense = hybrid.dense
    keyword = KeywordRetriever(dense.payloads, hybrid.keyword)

    # The router is deliberately absent. On a conceptual query it finds no
    # identifier and falls straight through to the semantic path, so it would
    # contribute nothing to the pool except duplicates.
    searchers = (dense, keyword, hybrid)

    pooled = []
    for i, query in enumerate(queries, start=1):
        candidates: dict[str, dict] = {}
        overlap = []
        for searcher in searchers:
            found = searcher.search(query, k=args.depth)
            overlap.append(len(found))
            for r in found:
                # Which retriever found it is deliberately not recorded. An
                # annotator who can see that a chunk came from the system
                # under test is no longer judging blind.
                candidates.setdefault(r.chunk_id, {
                    "id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "text": r.text,
                })

        # Sorted for determinism, then shuffled on a seed derived from the
        # query, so the order is reproducible and carries no rank information.
        items = [candidates[key] for key in sorted(candidates)]
        random.Random(query).shuffle(items)

        pooled.append({"query": query, "candidates": items})
        print(f"[{i:>2}/{len(queries)}] {sum(overlap):>3} returned -> "
              f"{len(items):>3} unique  {query}")

    args.out.write_text(json.dumps(pooled, indent=2))
    total = sum(len(p["candidates"]) for p in pooled)
    returned = len(searchers) * args.depth * len(pooled)
    print(f"\n{len(pooled)} queries, {total} judgments to make -> {args.out}")
    print(f"overlap: {returned} results collapsed to {total} unique "
          f"({100 * (1 - total / returned):.0f}% agreement between systems)")


if __name__ == "__main__":
    main()
