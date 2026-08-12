"""Is retrieval authority-blind?

Similarity says nothing about whether a judgment matters. A passage from a
judgment nobody has ever cited scores exactly like the same passage from a
leading case, because cosine distance has no opinion about precedent and BM25
has none either.

Whether that is actually happening needs no relevance labels to answer. Compare
the judgments a retriever returns against the corpus it drew them from: if
cited_by and court_type look like a random draw, the retriever is blind to
authority, and the corpus itself is the null hypothesis.

    uv run python scripts/profile_retrieval_authority.py
"""

from __future__ import annotations

import argparse
import logging
import statistics as st
from pathlib import Path

from rag.hybrid import HybridRetriever
from rag.keyword import KeywordRetriever

INDEX_DIR = Path("data/index-full")
QUERIES = Path("data/conceptual_queries.txt")


def profile(results) -> tuple[float, float, float]:
    """Median and mean cited_by, and the share from the Supreme Court.

    Median matters more than mean here. cited_by has a 5,977 maximum, so a
    single leading case in the results drags the mean somewhere the typical
    result never goes.
    """
    cited = [r.metadata.get("cited_by", 0) for r in results]
    supreme = sum(
        1 for r in results if r.metadata.get("court_type") == "Supreme_Court"
    )
    return st.median(cited), st.mean(cited), supreme / len(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    queries = [
        line.strip() for line in args.queries.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    hybrid = HybridRetriever(args.index)
    dense = hybrid.dense
    keyword = KeywordRetriever(dense.payloads, hybrid.keyword)

    # The corpus is the null hypothesis. A retriever sitting on these numbers
    # is choosing its results without reference to authority at all.
    corpus: dict[str, dict] = {}
    for p in dense.payloads:
        corpus.setdefault(p["doc_id"], p["metadata"])
    base_cited = [m.get("cited_by", 0) for m in corpus.values()]
    base_supreme = sum(
        1 for m in corpus.values() if m.get("court_type") == "Supreme_Court"
    ) / len(corpus)

    print(f"{len(queries)} conceptual queries, k={args.k}\n")
    header = f"{'config':<10}{'median cited_by':>17}{'mean':>10}{'supreme court':>16}"
    print(header)
    print("-" * len(header))
    print(f"{'corpus':<10}{st.median(base_cited):>17.0f}"
          f"{st.mean(base_cited):>10.0f}{base_supreme:>15.1%}")

    for name, searcher in (("dense", dense), ("bm25", keyword), ("hybrid", hybrid)):
        rows = [profile(searcher.search(q, k=args.k)) for q in queries]
        print(f"{name:<10}{st.mean(r[0] for r in rows):>17.0f}"
              f"{st.mean(r[1] for r in rows):>10.0f}"
              f"{st.mean(r[2] for r in rows):>15.1%}")


if __name__ == "__main__":
    main()
