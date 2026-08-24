"""Ask a question of the corpus and get a cited answer.

This is the whole system in one place: route the query, retrieve, assemble a
prompt whose sources are labelled with whose voice they are in, generate, and
show what the answer actually rests on.

    uv run python scripts/ask.py "when can a writ petition be dismissed for delay"
    uv run python scripts/ask.py "AIR 1974 Patna 164"
    uv run python scripts/ask.py                      # interactive

Startup loads the index, builds the BM25 postings and extracts identifiers
across 414,122 chunks, which takes a couple of minutes and is paid once. Run it
without a query to keep the process alive and ask repeatedly.

    --no-stance   drop the stance labels from the prompt, for comparison
    --sources     print the retrieved chunks in full
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rag.generate import REFUSAL, GroqLLM, answer
from rag.hybrid import HybridRetriever
from rag.route import RoutedRetriever, classify
from rag.rerank import CrossEncoderReranker, RerankedRetriever
from rag.stance import classify as stance_of

INDEX_DIR = Path("data/index-full")
RULE = "=" * 78


def show(result, args) -> None:
    if result.refused:
        print("\nNo answer. The retrieved passages do not cover this question.")
        print(f"(model returned {REFUSAL})")
    else:
        print(f"\n{result.text}\n")

    if not result.sources:
        return

    print(RULE)
    print(f"{'#':<3}{'stance':<10}{'cited':>7}  court and judgment")
    print("-" * len(RULE))
    for i, source in enumerate(result.sources, start=1):
        meta = source.metadata
        # A source the model never referred to did not support the answer, so
        # it is marked rather than hidden: a question answered from two of six
        # sources is a different thing from one answered from all six.
        used = "*" if i in result.cited else " "
        title = f"{meta.get('court', '?')} | {meta.get('title', '?')}"
        print(f"{used}{i:<2}{stance_of(source.text):<10}"
              f"{meta.get('cited_by', 0):>7}  {title[:52]}")
        print(f"   {source.chunk_id}")

    if args.sources:
        for i, source in enumerate(result.sources, start=1):
            print(f"\n--- [{i}] {stance_of(source.text)}")
            print(" ".join(source.text.split())[:1200])

    print(f"\n* cited by the model. {len(result.cited)} of "
          f"{len(result.sources)} sources used.")
    if result.invalid_citations:
        print(f"model cited sources that do not exist: {result.invalid_citations}")


def run(query: str, retriever, llm, args) -> None:
    route = classify(query)
    print(f"\n{RULE}\n{query}")
    print(f"routed as: {route}"
          + ("  (exact lookup first, semantic backfill)" if route == "identifier"
             else "  (semantic)"))

    started = time.time()
    result = answer(query, retriever, llm, k=args.k,
                    stance_notes=not args.no_stance)
    show(result, args)
    print(f"{time.time() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="omit for interactive mode")
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--no-stance", action="store_true",
                        help="omit stance labels from the prompt")
    parser.add_argument("--sources", action="store_true",
                        help="print retrieved passages in full")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print("loading index...", file=sys.stderr)
    started = time.time()
    hybrid = HybridRetriever(args.index)
    reranked = RerankedRetriever(hybrid, CrossEncoderReranker())
    retriever = RoutedRetriever(reranked, hybrid.dense.payloads)

    llm = GroqLLM()
    print(f"ready in {time.time() - started:.0f}s "
          f"({len(hybrid.dense.payloads):,} chunks, "
          f"{len(retriever.exact):,} identifiers)", file=sys.stderr)

    if args.query:
        run(args.query, retriever, llm, args)
        return

    while True:
        try:
            query = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query in {"quit", "exit"}:
            return
        try:
            run(query, retriever, llm, args)
        except Exception as exc:
            # A failed answer should not cost the two minute startup.
            print(f"failed: {exc}")


if __name__ == "__main__":
    main()
