#!/usr/bin/env python3
"""Build a K-NN Semantic Graph (GraphRAG) from the existing FAISS vectors.

Since the HuggingFace dataset provides citation counts but no explicit edge 
lists, this script constructs a semantic citation network by linking chunks 
that are highly semantically related (high cosine similarity). 
"""

import json
import logging
import sqlite3
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/index-sweep-450")
K_NEIGHBORS = 5

def build_graph():
    if not INDEX_DIR.exists():
        logger.error(f"Index directory {INDEX_DIR} does not exist.")
        return

    vectors_path = INDEX_DIR / "vectors-0000.npy"
    ids_path = INDEX_DIR / "ids.json"
    payloads_path = INDEX_DIR / "payloads.jsonl"
    db_path = INDEX_DIR / "graph.db"

    logger.info("Loading vectors and IDs...")
    vectors = np.load(vectors_path)
    with open(ids_path) as f:
        chunk_ids = json.load(f)

    # Normalize vectors just in case to ensure dot product == cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-9)

    logger.info(f"Computing pairwise similarity for {len(chunk_ids)} chunks...")
    # Matrix multiply for cosine similarity
    sim_matrix = vectors @ vectors.T

    # Set diagonal to -1 so a chunk doesn't link to itself
    np.fill_diagonal(sim_matrix, -1.0)

    # Get top K indices for each row
    logger.info(f"Extracting top {K_NEIGHBORS} neighbors...")
    # argpartition is faster than argsort for top-k
    k_indices = np.argpartition(-sim_matrix, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
    
    # Sort the top K for each row
    row_indices = np.arange(len(chunk_ids))[:, None]
    top_k_sims = sim_matrix[row_indices, k_indices]
    sorted_top_k_pos = np.argsort(-top_k_sims, axis=1)
    sorted_k_indices = k_indices[row_indices, sorted_top_k_pos]

    logger.info(f"Writing graph to {db_path}...")
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE nodes (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            title TEXT,
            court TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE edges (
            source_id TEXT,
            target_id TEXT,
            score REAL,
            PRIMARY KEY (source_id, target_id)
        )
    """)
    
    # Create indexes for fast traversal
    cur.execute("CREATE INDEX idx_edges_source ON edges(source_id)")
    cur.execute("CREATE INDEX idx_edges_target ON edges(target_id)")

    # Load payloads for node metadata
    logger.info("Populating nodes...")
    nodes_data = []
    with open(payloads_path) as f:
        for line in f:
            p = json.loads(line)
            chunk_id = p["id"]
            doc_id = p["doc_id"]
            title = p.get("metadata", {}).get("title", "")
            court = p.get("metadata", {}).get("court", "")
            nodes_data.append((chunk_id, doc_id, title, court))
            
    cur.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?)", nodes_data)

    logger.info("Populating edges...")
    edges_data = []
    for i in range(len(chunk_ids)):
        source_id = chunk_ids[i]
        for j in range(K_NEIGHBORS):
            target_idx = sorted_k_indices[i, j]
            target_id = chunk_ids[target_idx]
            score = float(sim_matrix[i, target_idx])
            edges_data.append((source_id, target_id, score))

    cur.executemany("INSERT INTO edges VALUES (?, ?, ?)", edges_data)

    conn.commit()
    conn.close()
    
    logger.info(f"Successfully built K-NN Semantic Graph with {len(nodes_data)} nodes and {len(edges_data)} edges.")

if __name__ == "__main__":
    build_graph()
