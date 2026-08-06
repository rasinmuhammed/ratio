"""Embedding: Chunk Stream -> normalized vectors on disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunk import Chunk

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64
SHARD_SIZE = 50_000
INDEX_DIR = Path("data/index")

def load_model(name: str = MODEL_NAME, device: str | None = None) -> SentenceTransformer:
    model = SentenceTransformer(name, device = device)
    logger.info("loaded %s | dim %d | max_seq=%d", name,
                model.get_embedding_dimension(), model.max_seq_length)
    return model

def token_length(model: SentenceTransformer):
    """Length function for the chunker, for chunk size to be measured
    in the model's own tokens and not from characters."""
    tokenizer = model.tokenizer
    return lambda text: len(tokenizer.encode(text, add_special_tokens=True))

def build_index(
        chunks: Iterable[Chunk],
        out_dir: Path = INDEX_DIR,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        shard_size: int = SHARD_SIZE,
        device: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_name, device)
    limit = model.max_seq_length
    tok_len = token_length(model)

    ids: list[str] = []
    shard: list[str] = []
    shard_no = 0
    truncated = 0
    total = 0

    def flush() -> None:
        nonlocal shard, shard_no
        if not shard:
            return
        vecs = model.encode(
            shard, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        np.save(out_dir / f"vectors-{shard_no:04d}.npy", vecs)
        logger.info("wrote shard %d (%d vectors)", shard_no, len(vecs))
        shard = []
        shard_no += 1

    for chunk in chunks:
        if tok_len(chunk.text) > limit:
            truncated += 1
        ids.append(chunk.id)
        shard.append(chunk.text)
        total += 1
        if len(shard) >= shard_size:
            flush()

    flush()

    (out_dir / "ids.json").write_text(json.dumps(ids))
    meta = {
        "model": model_name,
        "dim": model.get_embedding_dimension(),
        "max_seq_length": limit,
        "count": total,
        "truncated": truncated,
        "shards": shard_no,
        "normalized": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if truncated:
        logger.warning("%d/%d chunks exceeded %d tokens and were truncated",
                       truncated, total, limit)
    return meta

def load_index(out_dir: Path = INDEX_DIR) -> tuple[np.ndarray, list[str], dict]:
    meta = json.loads((out_dir / "meta.json").read_text())
    ids = json.loads((out_dir / "ids.json").read_text())
    shards = [np.load(p) for p in sorted(out_dir.glob("vectors-*.npy"))]
    vectors = np.vstack(shards)
    if len(ids) != len(vectors):
        raise ValueError(f"id/vector mismatch: {len(ids)} vs {len(vectors)}")
    return vectors, ids, meta