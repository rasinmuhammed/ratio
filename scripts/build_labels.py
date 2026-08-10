"""Build exact-match labels from the corpus itself.

If six chunks contain the literal string "AIR 1974", then a query for
"AIR 1974" has exactly six correct answers and we know them without asking
anyone. That gives a real labelled set for free.

It only tests one kind of retrieval, but it tests it perfectly, and it gets
the harness working end to end before investing hours in hand annotation.

The chunk-count window below is a property of the index it was run against,
not of the corpus, so labels have to be rebuilt whenever the index changes
size rather than carried over.

    uv run python scripts/build_labels.py
    uv run python scripts/build_labels.py --index data/index-full \
        --out data/labels-full.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from rag.evaluate import LABELS_PATH, LabelledQuery, save_labels

INDEX_DIR = Path("data/index")

PATTERNS = {
    "citation": r"\bAIR \d{4}\b",
    "case_number": r"\bNo\.\s?\d+ of \d{4}\b",
    "section": r"\bSection \d+[A-Z]?\b",
}

MIN_CHUNKS, MAX_CHUNKS = 2, 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--out", type=Path, default=LABELS_PATH)
    args = parser.parse_args()

    payloads = [json.loads(line) for line in (args.index / "payloads.jsonl").open()]

    occurrences: dict[str, set[str]] = defaultdict(set)
    for p in payloads:
        for pattern in PATTERNS.values():
            for match in re.finditer(pattern, p["text"]):
                occurrences[match.group(0)].add(p["id"])

    queries = [
        LabelledQuery(
            query=term,
            relevant_chunk_ids=sorted(ids),
            kind="exact",
            note="every chunk containing this literal string",
        )
        for term, ids in sorted(occurrences.items())
        if MIN_CHUNKS <= len(ids) <= MAX_CHUNKS
    ]

    save_labels(queries, args.out)
    print(f"{len(queries)} exact-match queries from {len(payloads)} chunks "
          f"-> {args.out}")
    for q in queries[:5]:
        print(f"  {q.query!r}: {len(q.relevant_chunk_ids)} relevant chunks")


if __name__ == "__main__":
    main()