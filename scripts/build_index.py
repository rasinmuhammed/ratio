"""Build the embedding index.

Chunk sizes are measured with the model's own tokenizer, not estimated from
characters, so nothing is silently truncated at the embedding step.

    uv run python scripts/build_index.py --limit 200
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

from rag.chunk import chunk_document
from rag.embed import build_index, load_model, token_length
from rag.ingest import load_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="number of documents (default: whole corpus)")
    parser.add_argument("--out", type=Path, default=Path("data/index"))
    args = parser.parse_args()

    model = load_model()
    length = token_length(model)

    docs = load_documents()
    if args.limit:
        docs = itertools.islice(docs, args.limit)

    chunks = (c for d in docs for c in chunk_document(d, length=length))
    meta = build_index(chunks, out_dir=args.out)

    print(meta)


if __name__ == "__main__":
    main()
