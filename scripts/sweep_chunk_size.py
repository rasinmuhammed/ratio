"""Sweep over chunk sizes to find the optimal configuration.

This builds multiple small indexes (e.g. 500 documents) with different chunk sizes
and runs evaluate.py against them to measure exact-match recall.
"""

import argparse
import itertools
import logging
import subprocess
import time
from pathlib import Path

from rag.chunk import chunk_document
from rag.embed import build_index, load_model, token_length
from rag.ingest import load_documents
import sqlite3

CONFIGS = [
    {"size": 450, "overlap": 100},  # current default but with SAC overhead
    {"size": 700, "overlap": 200},
    {"size": 1000, "overlap": 300},
]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500,
                        help="number of documents to use for the sweep")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("sweep")

    model = load_model()
    length = token_length(model)

    # Pre-load documents into memory so we don't re-parse JSONL for every config
    log.info(f"Loading first {args.limit} documents...")
    all_docs = list(itertools.islice(load_documents(), args.limit))
    
    # Load SAC summaries
    summaries_db = Path("data/summaries.db")
    doc_summaries = {}
    if summaries_db.exists():
        conn = sqlite3.connect(summaries_db)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, text FROM summary")
        doc_summaries = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        log.info(f"Loaded {len(doc_summaries)} SAC summaries")

    for doc in all_docs:
        if doc.id in doc_summaries:
            summary = doc_summaries[doc.id]
            if len(summary) < 2000:
                doc.metadata["doc_summary"] = summary
            else:
                log.warning(f"Ignoring abnormally long summary for {doc.id}")

    results = []

    for config in CONFIGS:
        size = config["size"]
        overlap = config["overlap"]
        out_dir = Path(f"data/index-sweep-{size}")
        
        log.info(f"\n--- Building index for size={size}, overlap={overlap} ---")
        
        def attach_and_chunk(doc):
            return chunk_document(doc, size=size, overlap_chars=overlap, length=length)
            
        chunks = (c for d in all_docs for c in attach_and_chunk(d))
        
        started = time.time()
        meta = build_index(chunks, out_dir=out_dir, model=model)
        elapsed = time.time() - started
        
        log.info(f"Built {meta['count']} chunks in {elapsed:.1f}s")
        
        log.info(f"Evaluating index size={size}...")
        # Run evaluate.py on this index
        cmd = ["uv", "run", "python", "scripts/evaluate.py", "--index", str(out_dir), "--limit", str(args.limit)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # The last line of evaluate.py output is usually the metrics dictionary
            output_lines = [line for line in res.stdout.splitlines() if line.strip()]
            metrics = output_lines[-1] if output_lines else "No metrics found"
            log.info(f"Result for size={size}: {metrics}")
            results.append((size, metrics))
        except subprocess.CalledProcessError as e:
            log.error(f"Evaluation failed for size={size}: {e.stderr}")
            results.append((size, "FAILED"))

    log.info("\n--- SWEEP RESULTS ---")
    for size, res in results:
        log.info(f"Size {size:3d}: {res}")

if __name__ == "__main__":
    main()
