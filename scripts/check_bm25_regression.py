"""Freeze BM25 output, then verify a refactor did not change it.

Restructuring how postings are stored changes the memory layout, not what BM25
computes, so every score has to come back bit-identical. The unit tests will
not catch drift here: they assert on specific pairs, not on the shape of the
score distribution, so a refactor could reorder every result and still pass.

    uv run python scripts/check_bm25_regression.py --save data/bm25_before.json
    uv run python scripts/check_bm25_regression.py --check data/bm25_before.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.keyword import BM25Index

PAYLOADS = Path("data/index/payloads.jsonl")
DEPTH = 20

QUERIES = [
    "AIR 1974",                  # the exact-match case dense retrieval loses
    "No.1091 of 2013",           # case number, multi-token
    "Section 9",                 # very common, exercises idf and saturation
    "24.06.2014",                # date, tokenises into three numbers
    "adverse possession",        # the crowded query
    "Maheshwar Mandal",          # party name carried by the context prefix
    "kumar kumar",               # repeated term, must count twice
    "zzzqxv nonexistent",        # no postings at all, must not raise
    "",                          # empty query, must return []
]


def snapshot(index: BM25Index) -> dict[str, list]:
    """repr() of the score, not round(). Rounding would hide a change in the
    last bit of a float, and a change there means the arithmetic moved."""
    return {
        q: [[m.index, repr(m.score)] for m in index.search(q, k=DEPTH)]
        for q in QUERIES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()

    payloads = [json.loads(line) for line in PAYLOADS.open()]
    index = BM25Index([p["text"] for p in payloads], [p["id"] for p in payloads])
    got = snapshot(index)

    if args.save:
        args.save.write_text(json.dumps(got, indent=2))
        hits = sum(len(v) for v in got.values())
        print(f"saved {len(QUERIES)} queries, {hits} results -> {args.save}")
        return

    want = json.loads(args.check.read_text())
    changed = [q for q in QUERIES if want.get(q) != got.get(q)]

    if changed:
        for q in changed:
            print(f"CHANGED {q!r}")
            print(f"  before {want.get(q)}")
            print(f"  after  {got.get(q)}")
        raise SystemExit(f"{len(changed)} of {len(QUERIES)} queries changed")

    print(f"identical across {len(QUERIES)} queries")


if __name__ == "__main__":
    main()
