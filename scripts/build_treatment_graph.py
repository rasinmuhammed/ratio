"""Build the citation-treatment graph: every AIR/SCC citation mention in the
corpus, with a shallow read of how the citing passage treated it.

Runs over full document text, not chunks, so a signal word split across a
chunk boundary by the 450-token chunker still lands in the same window as
the citation it belongs to. Pure regex, no model and no API call, so this is
safe to run alongside anything else that is CPU- or network-bound.

    PYTHONPATH=src python scripts/build_treatment_graph.py
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys

from rag.ingest import load_documents
from rag.treatment import classify_treatment, find_citations
from rag.treatment_db import DB_PATH, record_mentions, stats

BATCH_SIZE = 500
SNIPPET_CHARS = 160


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="number of documents to scan (default: whole corpus)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    docs_scanned = 0
    batch: list[tuple[str, str, str, str, str]] = []

    def flush() -> None:
        nonlocal batch
        if batch:
            record_mentions(batch)
            batch = []

    docs = load_documents()
    if args.limit:
        docs = itertools.islice(docs, args.limit)

    for doc in docs:
        docs_scanned += 1
        for citation in find_citations(doc.text):
            treatment = classify_treatment(doc.text, citation)
            snippet = doc.text[
                max(0, citation.start - SNIPPET_CHARS // 2):
                citation.end + SNIPPET_CHARS // 2
            ].strip()
            batch.append((citation.key, doc.id, treatment, citation.raw, snippet))

        if len(batch) >= BATCH_SIZE:
            flush()

        if docs_scanned % 1000 == 0:
            print(f"[{docs_scanned}] documents scanned", file=sys.stderr)

    flush()

    report = stats()
    print(f"\nScanned {docs_scanned} documents -> {DB_PATH}", file=sys.stderr)
    print(
        f"{report['total_mentions']} citation mentions, "
        f"{report['distinct_citations']} distinct citations",
        file=sys.stderr,
    )
    print(f"By treatment: {report['by_treatment']}", file=sys.stderr)


if __name__ == "__main__":
    main()
