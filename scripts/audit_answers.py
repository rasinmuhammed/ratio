"""Measure how much of an answer its own citations actually support.

Retrieval in this repo is measured hard. Generation is not measured at all,
which is the wrong way round: a retrieval failure produces a vague answer or a
refusal, and both are visible. An attribution failure produces a confident
sentence with a citation attached to a passage that does not say it, and a
reader who checks the number is reassured by it.

Unlike relevance, this needs no hand-labelled ground truth. "Does chunk 7
answer this query" requires legal judgment. "Does chunk 7 say what this
sentence claims it says" requires only the sentence and the chunk, both of
which the system already produced.

    export GROQ_API_KEY=...
    uv run python scripts/audit_answers.py --queries 8
    uv run python scripts/audit_answers.py --show          # print every verdict
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rag.attribute import (
    MISSING,
    SUPPORTED,
    UNCLEAR,
    check_claims,
    split_claims,
    summarise,
)
from rag.generate import GROQ_MODEL, GroqLLM, answer
from rag.hybrid import HybridRetriever
from rag.route import RoutedRetriever

INDEX_DIR = Path("data/index-full")
QUERIES = Path("data/conceptual_queries.txt")
RULE = "=" * 78


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--limit", type=int, default=8,
                        help="how many queries to audit")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--show", action="store_true",
                        help="print every claim and verdict")
    parser.add_argument("--judge-model", default="openai/gpt-oss-20b",
                        help="a different model from the one being audited")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    queries = [
        line.strip() for line in args.queries.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ][:args.limit]

    print("loading index...", file=sys.stderr)
    started = time.time()
    hybrid = HybridRetriever(args.index)
    retriever = RoutedRetriever(hybrid, hybrid.dense.payloads)
    llm = GroqLLM()

    # A separate, smaller model does the judging, for two reasons. Asking a
    # model to mark its own homework invites self-preference bias: it rates
    # its own phrasing as supported more readily than a stranger's. And the
    # free tier meters tokens per minute per model, so splitting the work
    # across two of them roughly doubles the throughput of a long run.
    judge = GroqLLM(model=args.judge_model)
    print(f"ready in {time.time() - started:.0f}s "
          f"(answers: {GROQ_MODEL}, judge: {args.judge_model})", file=sys.stderr)

    all_claims, all_checks = [], []
    truncated = 0
    uncited_answers = 0

    for i, query in enumerate(queries, start=1):
        result = answer(query, retriever, llm, k=args.k)
        if result.refused:
            print(f"[{i}/{len(queries)}] refused: {query}")
            continue
        if result.truncated:
            # Excluded, not counted. A cut-off answer is missing citations it
            # would have made, and scoring it would measure max_tokens rather
            # than the model.
            truncated += 1
            print(f"[{i}/{len(queries)}] TRUNCATED, excluded: {query}")
            continue

        claims = split_claims(result.text)
        if not result.cited:
            # Counted separately. An answer with no citations at all is a
            # different failure from an answer that cites badly, and folding
            # it into the uncited-claim rate hides which one is happening.
            uncited_answers += 1
        checks = check_claims(claims, result.sources, judge)
        all_claims.extend(claims)
        all_checks.extend(checks)

        bad = sum(1 for c in checks if c.verdict not in (SUPPORTED, UNCLEAR))
        print(f"[{i}/{len(queries)}] {len(claims):>2} claims, "
              f"{len(checks):>2} citations, {bad} not supported  {query}")

        if args.show:
            for claim in claims:
                verdicts = [c.verdict for c in checks if c.claim.start == claim.start]
                label = "/".join(verdicts) if verdicts else "uncited"
                print(f"     {label:<24} {claim.text[:60]}")

    print(f"\n{RULE}")
    print(f"  {'answers with no cites':<20} {uncited_answers:>8} of {len(queries)}")
    if truncated:
        print(f"  {'truncated (excluded)':<20} {truncated:>8}")
    for key, value in summarise(all_claims, all_checks).items():
        if isinstance(value, float):
            print(f"  {key:<20} {value:>8.3f}")
        else:
            print(f"  {key:<20} {value:>8}")

    print("\ncitation_precision: of the citations the model made, the share")
    print("pointing at a passage that says the thing.")
    print("claim_support: of the claims that cited anything, the share with at")
    print("least one source that holds up.")
    print(f"uncited claims are counted but never judged; {MISSING} means the")
    print("model cited a source it was never given.")


if __name__ == "__main__":
    main()
