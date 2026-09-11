"""Generate Summary-Augmented Chunking (SAC) summaries for documents.

This script uses the LLM (K2-Horizon by default) to generate a 2-3 sentence
global summary of each judgment. These summaries are saved to a SQLite database
and later prepended to every chunk to solve Document-Level Retrieval Mismatch.

It is resumable and supports batching (e.g. 500 or 2000 documents at a time).
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

from rag.generate import get_llm, LLM
from rag.ingest import load_documents, Document

DB_PATH = Path("data/summaries.db")

import re

# A specialized system prompt asking for exactly the components needed for SAC
SAC_SYSTEM = (
    "You are an expert Indian legal assistant. "
    "Summarise this Indian court judgment in exactly 2-3 sentences. "
    "State: (1) the core legal question or issue, (2) the court's holding/outcome, "
    "and (3) the court and bench level. "
    "Keep it concise and plain text only. "
    "IMPORTANT: You must wrap your final 2-3 sentence summary inside <summary> tags. "
    "Do not include any reasoning inside the <summary> tags."
)

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)

def setup_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS summary "
        "(doc_id TEXT PRIMARY KEY, text TEXT)"
    )
    conn.commit()
    return conn

def generate_summary(doc: Document, llm: LLM) -> str | None:
    # We pass the first 12,000 chars (approx 3000 tokens) to get the core facts.
    # The start of a judgment almost always contains the facts, issue, and court.
    prompt = f"Judgment Text:\n{doc.text[:12000]}"
    reply = llm.complete(SAC_SYSTEM, prompt)
    match = _SUMMARY_RE.search(reply)
    if match:
        return match.group(1).strip()
    # Fallback if the model forgot the tags or got truncated
    logging.warning("No <summary> tags found for %s (length %d)", doc.id, len(reply))
    return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of documents to process in this run")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    conn = setup_db(args.db)
    cursor = conn.cursor()

    cursor.execute("SELECT doc_id FROM summary")
    done = {row[0] for row in cursor.fetchall()}

    print(f"Loaded {len(done)} existing summaries from {args.db}", file=sys.stderr)

    llm = get_llm()
    print(f"Using LLM: {getattr(llm, 'model', type(llm).__name__)}", file=sys.stderr)

    processed = 0
    docs = load_documents()
    
    for doc in docs:
        if processed >= args.limit:
            break
        
        if doc.id in done:
            continue
            
        summary = generate_summary(doc, llm)
        if summary:
            cursor.execute(
                "INSERT INTO summary (doc_id, text) VALUES (?, ?)",
                (doc.id, summary)
            )
            conn.commit()
            processed += 1
            print(f"[{processed}/{args.limit}] Summarized {doc.id}")

    print(f"\nFinished. Processed {processed} new documents.", file=sys.stderr)

if __name__ == "__main__":
    main()
