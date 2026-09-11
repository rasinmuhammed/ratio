"""Pool top chunks from multiple retrievers for conceptual queries.

This fetches the top-K chunks from Dense, BM25, Hybrid, and Reranked
paths to create a comprehensive pool of candidates for the AI judge to annotate.
"""

import argparse
import json
import logging
from pathlib import Path

from rag.evaluate import load_labels
from rag.hybrid import HybridRetriever
from rag.keyword import KeywordRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/index-sweep-450"),
                        help="Index to search against")
    parser.add_argument("--labels", type=Path, default=Path("data/labels.json"),
                        help="Path to conceptual labels (queries)")
    parser.add_argument("--out", type=Path, default=Path("data/pool.json"),
                        help="Output path for the pooled chunks")
    parser.add_argument("--k", type=int, default=20,
                        help="Top K to retrieve per method")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("pool")

    # Load conceptual queries
    queries = [q for q in load_labels(args.labels) if q.kind == "conceptual"]
    if not queries:
        log.warning("No conceptual queries found!")
        return
        
    log.info(f"Loaded {len(queries)} conceptual queries")

    # Initialize retrievers
    hybrid = HybridRetriever(args.index)
    dense = hybrid.dense
    keyword = KeywordRetriever(dense.payloads, hybrid.keyword)
    reranker = CrossEncoderReranker()
    reranked = RerankedRetriever(hybrid, reranker)
    
    retrievers = {
        "dense": dense,
        "bm25": keyword,
        "hybrid": hybrid,
        "reranked": reranked
    }

    pool_data = []

    for query in queries:
        query_text = query.query
        log.info(f"Pooling for query: {query_text}")
        
        unique_chunks = {}
        
        for name, searcher in retrievers.items():
            try:
                results = searcher.search(query_text, k=args.k)
                for res in results:
                    unique_chunks[res.chunk_id] = res.text
            except Exception as e:
                log.error(f"  {name} search failed: {e}")
                
        log.info(f"  Total unique chunks: {len(unique_chunks)}")
        
        chunks_list = [{"id": cid, "text": text} for cid, text in unique_chunks.items()]
        
        pool_data.append({
            "query": query_text,
            "chunks": chunks_list
        })

    with open(args.out, "w") as f:
        json.dump(pool_data, f, indent=2)
    
    log.info(f"\nSaved {len(pool_data)} pooled queries to {args.out}")

if __name__ == "__main__":
    main()
