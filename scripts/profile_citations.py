"""Measure whether this corpus can support citation-graph retrieval.

Similarity retrieval has no notion of authority. It cannot tell a leading case
from a passing mention, and it cannot tell a binding precedent from one that
was overruled. A citation graph can, but only if the corpus has the shape to
support one, and that is a question about the data rather than about the idea.

Four things decide it:

1. Is there an authority signal at all, or is every judgment cited equally?
2. Is the court hierarchy recorded, so binding can be distinguished from
   merely persuasive?
3. Do judgments cite in a format that can be extracted reliably?
4. Can a citation be resolved to a judgment *in this corpus*, or does it point
   outside it? This is the one that decides whether edges connect anything.

    uv run python scripts/profile_citations.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

INDEX_DIR = Path("data/index-full")

# Reported citations as they appear in the text. The bracketed form is the
# common convention and it is what makes party names recoverable: the citation
# sits immediately after the case name it belongs to.
CITATION = re.compile(
    r"\b(?:AIR\s+\d{4}\s+[A-Z][A-Za-z]*\s+\d+"          # AIR 1974 Patna 164
    r"|\(\d{4}\)\s*\d+\s*SCC\s*\d+"                      # (1998) 2 SCC 341
    r"|\d{4}\s*\(\d+\)\s*[A-Z]{2,}\s*\d+)"               # 1998 (2) BLJR 341
)

# "Nand Kumar Rai & Others Vs. State of Bihar & Others [AIR 1974 Patna 164]"
# The party names are the bridge from a citation to a judgment, since the
# citation string itself never appears in the cited judgment's own metadata.
CITED_CASE = re.compile(
    r"([A-Z][A-Za-z.&'\- ]{4,80}?\s+(?:Vs?\.?|versus)\s+[A-Z][A-Za-z.&'\- ]{4,80}?)"
    r"\s*[\[\(]\s*(" + CITATION.pattern + r")\s*[\]\)]"
)

STOP = {"the", "of", "and", "state", "others", "anr", "ors", "vs", "v",
        "union", "india", "in", "re", "smt", "shri", "&"}


def significant(name: str) -> set[str]:
    """Tokens distinctive enough to identify a party.

    'State of Bihar & Others' is most of the corpus, so the discriminating
    tokens are the ones that are not boilerplate.
    """
    return {t for t in re.findall(r"[a-z]+", name.lower())
            if t not in STOP and len(t) > 2}


def describe(name: str, values: list[int]) -> None:
    zeros = sum(1 for v in values if v == 0)
    order = sorted(values)
    print(f"  {name:<12} mean {st.mean(values):>7.1f}   median {st.median(values):>5.0f}   "
          f"p90 {order[int(0.9 * len(order))]:>5}   max {max(values):>6}   "
          f"zero {100 * zeros / len(values):>5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    args = parser.parse_args()

    docs: dict[str, dict] = {}
    text_by_doc: dict[str, list[str]] = {}

    with (args.index / "payloads.jsonl").open() as handle:
        for line in handle:
            p = json.loads(line)
            doc_id = p["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = p["metadata"]
                text_by_doc[doc_id] = []
            text_by_doc[doc_id].append(p["text"])

    print(f"{len(docs):,} distinct judgments\n")

    # 1. Authority signal.
    print("Authority, from dataset metadata")
    describe("cited_by", [m.get("cited_by", 0) for m in docs.values()])
    describe("cites", [m.get("cites", 0) for m in docs.values()])

    top = sorted(docs.items(), key=lambda kv: -kv[1].get("cited_by", 0))[:5]
    print("\n  most-cited judgments in the corpus:")
    for doc_id, meta in top:
        print(f"    {meta.get('cited_by', 0):>5}  {meta.get('court', '?')[:22]:<22} "
              f"{meta.get('title', '?')[:58]}")

    # 2. Hierarchy.
    print("\nCourt hierarchy")
    for court_type, n in Counter(
        m.get("court_type", "unknown") for m in docs.values()
    ).most_common():
        print(f"  {court_type:<16} {n:>7,}  ({100 * n / len(docs):>5.1f}%)")

    # 3. Extractable citations.
    citing_docs = 0
    all_citations: Counter[str] = Counter()
    pairs: list[tuple[str, str]] = []

    for doc_id, chunks in text_by_doc.items():
        body = "\n".join(chunks)
        found = CITATION.findall(body)
        if found:
            citing_docs += 1
            all_citations.update(re.sub(r"\s+", " ", c).strip() for c in found)
        for name, citation in CITED_CASE.findall(body):
            pairs.append((name.strip(), re.sub(r"\s+", " ", citation).strip()))

    print("\nCitations in the text")
    print(f"  judgments citing at least one   {citing_docs:>7,}  "
          f"({100 * citing_docs / len(docs):>5.1f}%)")
    print(f"  distinct citations              {len(all_citations):>7,}")
    print(f"  total occurrences               {sum(all_citations.values()):>7,}")
    print(f"  with a recoverable case name    {len(pairs):>7,}  "
          f"({100 * len(pairs) / max(1, sum(all_citations.values())):>5.1f}% of occurrences)")

    print("\n  most-cited authorities by in-text mentions:")
    for citation, n in all_citations.most_common(5):
        print(f"    {n:>5}  {citation}")

    # 4. Resolution. The question that decides whether edges connect anything.
    titles = [
        (doc_id, significant(meta.get("title", "")))
        for doc_id, meta in docs.items()
    ]
    resolved = 0
    seen: set[str] = set()
    for name, citation in pairs:
        if citation in seen:
            continue
        seen.add(citation)
        want = significant(name)
        if len(want) < 2:
            continue
        if any(want <= tokens for _, tokens in titles):
            resolved += 1

    print("\nResolution against this corpus")
    print(f"  distinct cited cases with a name  {len(seen):>7,}")
    print(f"  matched to a judgment here        {resolved:>7,}  "
          f"({100 * resolved / max(1, len(seen)):>5.1f}%)")


if __name__ == "__main__":
    main()
