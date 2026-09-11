"""Build the embedding index.

Chunk sizes are measured with the model's own tokenizer, not estimated from
characters, so nothing is silently truncated at the embedding step.

    uv run python scripts/build_index.py --limit 200
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

from rag.chunk import chunk_document
from rag.embed import build_index, load_model, token_length
from rag.ingest import load_documents


def configure_logging(verbose: bool) -> None:
    """Our own logs at INFO, third-party libraries at WARNING.

    Without this, basicConfig(INFO) also switches on httpx and transformers,
    which bury the two lines we actually care about under HTTP traffic.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not verbose:
        for noisy in ("httpx", "transformers", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="number of documents (default: whole corpus)")
    parser.add_argument("--out", type=Path, default=Path("data/index"))
    parser.add_argument("--verbose", action="store_true",
                        help="include third-party library logs")
    args = parser.parse_args()

    configure_logging(args.verbose)
    log = logging.getLogger("build_index")

    # Loaded once and reused: chunking needs the tokenizer for length, and
    # embedding needs the model itself. Two loads cost ~30s for nothing.
    model = load_model()
    length = token_length(model)

    import sqlite3
    
    docs = load_documents()
    if args.limit:
        docs = itertools.islice(docs, args.limit)

    # Optional: Load Summary-Augmented Chunking (SAC) summaries if they exist
    summaries_db = Path("data/summaries.db")
    doc_summaries = {}
    if summaries_db.exists():
        log.info("Loading SAC summaries from %s", summaries_db)
        try:
            conn = sqlite3.connect(summaries_db)
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id, text FROM summary")
            doc_summaries = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            log.info("Loaded %d SAC summaries", len(doc_summaries))
        except Exception as e:
            log.warning("Failed to load SAC summaries: %s", e)
            
    def attach_and_chunk(doc):
        if doc.id in doc_summaries:
            # We must not mutate the frozen dataclass directly, but metadata is a dict
            doc.metadata["doc_summary"] = doc_summaries[doc.id]
        return chunk_document(doc, length=length)

    chunks = (c for d in docs for c in attach_and_chunk(d))

    log.info("building index -> %s", args.out)
    started = time.time()
    meta = build_index(chunks, out_dir=args.out, model=model)
    elapsed = time.time() - started

    log.info("done in %.1fs | %d chunks | %d truncated",
             elapsed, meta["count"], meta["truncated"])
    print(meta)


if __name__ == "__main__":
    main()
