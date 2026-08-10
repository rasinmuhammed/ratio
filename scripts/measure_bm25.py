"""Measure what BM25Index costs in memory and project it to the full corpus.

Vectors are written to disk in shards and memory-mapped back, so they scale
without much thought. The BM25 index does not: it is built entirely in RAM and
stays there, which makes its footprint the thing that decides whether a
full-corpus build is possible at all on this machine.

    uv run python scripts/measure_bm25.py
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

from rag.keyword import BM25Index

PAYLOADS = Path("data/index/payloads.jsonl")

# The current index was built with --limit 200. ingest.py deduplicates the
# multi-label flattening down to 10,588 distinct judgments.
INDEXED_DOCS = 200
CORPUS_DOCS = 10_588


def main() -> None:
    payloads = [json.loads(line) for line in PAYLOADS.open()]
    texts = [p["text"] for p in payloads]
    ids = [p["id"] for p in payloads]

    # texts and ids are allocated before tracing starts, so what gets measured
    # is the marginal cost of the index structures rather than the corpus
    # itself. That is the number that matters: the corpus streams from disk,
    # the index cannot.
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    index = BM25Index(texts, ids)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cost = current - baseline
    scale = CORPUS_DOCS / INDEXED_DOCS

    mb = 1024 ** 2
    gb = 1024 ** 3
    print(f"chunks indexed      {len(texts):>12,}")
    print(f"distinct terms      {len(index.postings):>12,}")
    print(f"index cost          {cost / mb:>12,.1f} MB")
    print(f"peak during build   {peak / mb:>12,.1f} MB")
    print()
    print(f"scale factor        {scale:>12.1f}x")
    print(f"projected chunks    {len(texts) * scale:>12,.0f}")
    print(f"projected cost      {cost * scale / gb:>12,.2f} GB")


if __name__ == "__main__":
    main()
