"""Does retrieval return what counsel argued, or what the court held?

A judgment is not uniform text. It recites facts, records what each side
submitted, and then states what the court decided, and the first two routinely
contradict the third. "Learned counsel submitted that limitation does not
apply" is a proposition that may have been rejected two paragraphs later.

Nothing in cosine similarity or BM25 can see that difference. Both score a
passage on whether it discusses limitation, not on whether the court agreed.
A retrieved argument passage handed to a language model produces a confident
statement of the opposite of the law, correctly cited, which is the worst
failure mode available to a legal RAG system: wrong and credible.

This measures two things. How much of the corpus is argument rather than
holding, and whether retrieval prefers one over the other.

    uv run python scripts/profile_stance.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

INDEX_DIR = Path("data/index-full")
QUERIES = Path("data/conceptual_queries.txt")

# What a party said. These are reporting verbs: the judgment is describing a
# position, not adopting one.
ARGUMENT = re.compile(
    r"\b(?:"
    r"learned\s+(?:counsel|advocate|senior\s+counsel|standing\s+counsel)"
    r"|it\s+(?:was|is)\s+(?:submitted|contended|urged|argued)"
    r"|(?:submitted|contended|urged|argued)\s+(?:that|by)"
    r"|on\s+behalf\s+of\s+the\s+(?:petitioner|respondent|appellant|applicant)"
    r"|the\s+(?:petitioner|respondent|appellant|applicant)\s+"
    r"(?:submits|contends|urges|argues|submitted|contended)"
    r")\b",
    re.I,
)

# What the court decided. First person plural and decisional verbs, which a
# judgment reserves for its own voice.
HOLDING = re.compile(
    r"\b(?:"
    r"we\s+(?:are\s+of\s+the\s+(?:opinion|view)|hold|find|are\s+satisfied|conclude)"
    r"|in\s+our\s+(?:considered\s+)?(?:opinion|view|judgment)"
    r"|it\s+is\s+(?:held|well\s+settled)"
    r"|this\s+court\s+(?:holds|is\s+of\s+the\s+view)"
    r"|the\s+law\s+is\s+(?:well\s+)?settled"
    r"|we\s+are\s+unable\s+to\s+accept"
    r")\b",
    re.I,
)


def stance(text: str) -> str:
    """Coarse on purpose. A 450 token chunk can hold both voices, and how
    often it does is itself the question: a chunk carrying an argument and its
    rejection is safe, one carrying only the argument is not."""
    argument = bool(ARGUMENT.search(text))
    holding = bool(HOLDING.search(text))
    if argument and holding:
        return "both"
    if argument:
        return "argument"
    if holding:
        return "holding"
    return "neither"


def report(name: str, counts: dict[str, int]) -> None:
    total = sum(counts.values()) or 1
    row = "".join(
        f"{100 * counts.get(key, 0) / total:>11.1f}%"
        for key in ("argument", "holding", "both", "neither")
    )
    print(f"{name:<10}{row}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    corpus: dict[str, int] = {}
    with (args.index / "payloads.jsonl").open() as handle:
        for line in handle:
            label = stance(json.loads(line)["text"])
            corpus[label] = corpus.get(label, 0) + 1

    header = f"{'':<10}{'argument':>12}{'holding':>12}{'both':>12}{'neither':>12}"
    print(header)
    print("-" * len(header))
    report("corpus", corpus)

    # Imported here so the corpus pass above runs without loading a model,
    # which makes the expensive half opt-out rather than mandatory.
    from rag.hybrid import HybridRetriever
    from rag.keyword import KeywordRetriever

    queries = [
        line.strip() for line in args.queries.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    hybrid = HybridRetriever(args.index)
    dense = hybrid.dense
    keyword = KeywordRetriever(dense.payloads, hybrid.keyword)

    for name, searcher in (("dense", dense), ("bm25", keyword), ("hybrid", hybrid)):
        counts: dict[str, int] = {}
        for query in queries:
            for r in searcher.search(query, k=args.k):
                label = stance(r.text)
                counts[label] = counts.get(label, 0) + 1
        report(name, counts)

    print(f"\n{len(queries)} conceptual queries, k={args.k}")
    print("'argument' is a chunk recording what a party said with no sign of "
          "the court's own view.")


if __name__ == "__main__":
    main()
