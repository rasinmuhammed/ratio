"""Does voyage-law-2 rank correct chunks higher than bge-small, on an
identical candidate pool?

Comparing against the full-corpus bge-small numbers in retrieval_notes.md
would be invalid: this pool is thousands of chunks, the real index is
414,122, and a smaller pool makes any model look better regardless of
quality. So both models search the same pool here, built from the chunks
that back a sample of labelled queries plus shared distractors, and only a
recall difference on that identical pool is a fair comparison.

    export VOYAGE_API_KEY=...
    uv run python scripts/probe_voyage.py --queries 60 --distractors 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from rag.evaluate import LabelledQuery, evaluate, summarise
from rag.retrieve import DEFAULT_K, Result, load_model, rank
from rag.voyage_embed import VoyageEmbedder

INDEX_DIR = Path("data/index-full")
LABELS_PATH = Path("data/labels-full.json")


class PoolSearcher:
    """A Searcher over an in-memory pool rather than the real index.

    Satisfies the same protocol evaluate() already knows how to score, so
    nothing about the metrics code has to change to test a candidate model.
    """

    def __init__(self, embedder, payloads: list[dict], vectors) -> None:
        self.embedder = embedder
        self.payloads = payloads
        self.vectors = vectors

    def search(self, query: str, k: int = DEFAULT_K) -> list[Result]:
        query_vec = self.embedder.embed_query(query)
        idx, scores = rank(query_vec, self.vectors, k)
        return [
            Result(rank=i + 1, score=float(s), chunk_id=self.payloads[j]["id"],
                   doc_id=self.payloads[j]["doc_id"], text=self.payloads[j]["text"],
                   metadata=self.payloads[j]["metadata"], score_type="cosine")
            for i, (j, s) in enumerate(zip(idx, scores))
        ]


def build_pool(labels: list[LabelledQuery], all_payloads: dict[str, dict],
                n_distractors: int, seed: int) -> list[dict]:
    positives = {cid for q in labels for cid in q.relevant_chunk_ids}
    pool = {cid: all_payloads[cid] for cid in positives if cid in all_payloads}

    rng = random.Random(seed)
    remaining = [cid for cid in all_payloads if cid not in pool]
    for cid in rng.sample(remaining, min(n_distractors, len(remaining))):
        pool[cid] = all_payloads[cid]

    return list(pool.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--queries", type=int, default=60,
                        help="labelled queries to test, evenly sampled")
    parser.add_argument("--distractors", type=int, default=5000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    raw = json.loads(args.labels.read_text())
    all_labels = [LabelledQuery(**row) for row in raw]
    labels = all_labels[::max(1, len(all_labels) // args.queries)][:args.queries]

    print(f"loading payloads from {args.index}...")
    with (args.index / "payloads.jsonl").open() as f:
        all_payloads = {json.loads(line)["id"]: json.loads(line)
                        for line in f}

    pool = build_pool(labels, all_payloads, args.distractors, args.seed)
    print(f"pool: {len(pool)} chunks ({len(pool) - args.distractors} positive, "
          f"{args.distractors} distractor), {len(labels)} queries")

    texts = [p["text"] for p in pool]

    print("embedding pool with bge-small...")
    bge_model = load_model()
    bge_vecs = bge_model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False)

    class BgeAdapter:
        def embed_query(self_, q):
            from rag.retrieve import QUERY_PREFIX
            return bge_model.encode(QUERY_PREFIX + q, normalize_embeddings=True)

    print("embedding pool with voyage-law-2 (this spends API tokens)...")
    voyage = VoyageEmbedder()
    voyage_vecs = voyage.embed_documents(texts)

    configs = {
        "bge-small (this pool)": PoolSearcher(BgeAdapter(), pool, bge_vecs),
        "voyage-law-2":          PoolSearcher(voyage, pool, voyage_vecs),
    }

    print(f"\n{len(labels)} queries, k={args.k}, pool={len(pool)} chunks\n")
    header = f"{'config':<24}{'recall':>9}{'precision':>11}{'MRR':>8}"
    print(header)
    print("-" * len(header))
    for name, searcher in configs.items():
        summary = summarise(evaluate(searcher, labels, k=args.k))["all"]
        print(f"{name:<24}{summary['recall']:>9.3f}"
              f"{summary['precision']:>11.3f}{summary['mrr']:>8.3f}")


if __name__ == "__main__":
    main()