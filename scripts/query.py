"""Search the index from the command line.

    uv run python scripts/query.py "adverse possession of land"
    uv run python scripts/query.py "AIR 1974" --mode keyword
    uv run python scripts/query.py "AIR 1974" --mode hybrid --depth 5 --kw-weight 2
"""

from __future__ import annotations

import argparse

from rag.hybrid import CANDIDATE_DEPTH, HybridRetriever
from rag.keyword import BM25Index
from rag.retrieve import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=("dense", "keyword", "hybrid"),
                        default="dense")
    parser.add_argument("--depth", type=int, default=CANDIDATE_DEPTH,
                        help="how deep in each ranking to fuse (hybrid only)")
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--kw-weight", type=float, default=1.0)
    args = parser.parse_args()

    def show(rank: int, score: float, unit: str, chunk_id: str, text: str) -> None:
        body = text.split("\n\n", 1)[1] if "\n\n" in text else text
        print(f"{rank}. {score:.4f} ({unit})  {chunk_id}")
        print(f"   {body[:200]}")
        print()

    if args.mode == "keyword":
        # No model needed, so build BM25 straight off the payloads.
        dense = Retriever()
        bm = BM25Index([p["text"] for p in dense.payloads],
                       [p["id"] for p in dense.payloads])
        for rank, m in enumerate(bm.search(args.query, k=args.k), start=1):
            p = dense.payloads[m.index]
            show(rank, m.score, "bm25", p["id"], p["text"])

    elif args.mode == "hybrid":
        for r in HybridRetriever().search(
            args.query, k=args.k, depth=args.depth,
            dense_weight=args.dense_weight, keyword_weight=args.kw_weight,
        ):
            show(r.rank, r.score, r.score_type, r.chunk_id, r.text)

    else:
        for r in Retriever().search(args.query, k=args.k):
            show(r.rank, r.score, r.score_type, r.chunk_id, r.text)


if __name__ == "__main__":
    main()
