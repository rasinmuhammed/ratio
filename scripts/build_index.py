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

    docs = load_documents()
    if args.limit:
        docs = itertools.islice(docs, args.limit)

    chunks = (c for d in docs for c in chunk_document(d, length=length))

    log.info("building index -> %s", args.out)
    started = time.time()
    meta = build_index(chunks, out_dir=args.out, model=model)
    elapsed = time.time() - started

    log.info("done in %.1fs | %d chunks | %d truncated",
             elapsed, meta["count"], meta["truncated"])
    print(meta)


if __name__ == "__main__":
    main()
