"""Embedding: Chunk Stream -> normalized vectors on disk."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

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

CHECKPOINT_NAME = ".checkpoint.json"


def _read_checkpoint(out_dir: Path) -> dict | None:
    path = out_dir / CHECKPOINT_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resume_state(out_dir: Path, checkpoint: dict) -> tuple[list[str], int]:
    """Reconcile payloads.jsonl with the last confirmed checkpoint.

    A crash can land after more lines were written to payloads.jsonl than
    the last flush() actually embedded to disk (payloads are written
    per-chunk, vectors per-shard), so payloads.jsonl can hold a tail of
    chunks with no corresponding vectors. That tail is truncated away here
    rather than trusted, so the file on resume describes exactly the chunks
    the existing vectors-*.npy shards actually cover. ids.json is not kept
    as a running file (only written once, at the very end, by design, so a
    crash never leaves a stale one to disagree with reality), so the id list
    for everything already done is rebuilt from payloads.jsonl itself.
    """
    payloads_path = out_dir / "payloads.jsonl"
    lines = payloads_path.read_text().splitlines() if payloads_path.exists() else []
    confirmed = lines[:checkpoint["chunks_written"]]
    if len(confirmed) != checkpoint["chunks_written"]:
        raise RuntimeError(
            f"checkpoint claims {checkpoint['chunks_written']} chunks but "
            f"payloads.jsonl only has {len(lines)} lines; the index "
            f"directory is inconsistent and should not be resumed from."
        )
    if len(lines) != len(confirmed):
        payloads_path.write_text("\n".join(confirmed) + "\n")
    ids = [json.loads(line)["id"] for line in confirmed]
    return ids, checkpoint["shard_no"]


def build_index(
        chunks: Iterable[Chunk],
        out_dir: Path = INDEX_DIR,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        shard_size: int = SHARD_SIZE,
        device: str | None = None,
        model: SentenceTransformer | None = None,
) -> dict:
    """Embed chunks and write vectors, ids and metadata.

    Pass an already-loaded `model` to avoid a second load. The caller usually
    has one already, because chunking needs its tokenizer for length.

    Resumable: if out_dir already holds a checkpoint from an earlier,
    interrupted run over the same chunk stream (same corpus, same chunking
    config, so the Nth chunk yielded is the same chunk both times), the
    chunks already embedded are skipped rather than re-embedded. Skipping
    still means iterating them, chunking is cheap CPU work, model.encode()
    on 50,000 chunks at a time is what a restart would otherwise repeat for
    nothing, and re-chunking to skip past them costs a small fraction of
    that. A corpus-wide reindex is exactly the kind of multi-hour job worth
    protecting from having to restart at zero after a crash three shards in.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = load_model(model_name, device)
    limit = model.max_seq_length
    tok_len = token_length(model)

    checkpoint = _read_checkpoint(out_dir)
    if checkpoint is not None:
        ids, shard_no = _resume_state(out_dir, checkpoint)
        total = len(ids)
        skip = total
        logger.info("resuming from checkpoint: %d chunks already embedded "
                    "(%d shards)", total, shard_no)
    else:
        ids, shard_no, total, skip = [], 0, 0, 0

    shard: list[str] = []
    truncated = 0

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
        shard_no += 1
        shard = []
        # Written only once the vectors backing it are actually on disk, so
        # a checkpoint is never ahead of what it claims. flush() beneath the
        # payloads file handle below still has that file's own buffered
        # writes on disk first: Python flushes file objects on normal
        # interpreter exit, but not mid-run, so the payloads writer's own
        # buffer is flushed explicitly before the checkpoint is trusted.
        payloads.flush()
        (out_dir / CHECKPOINT_NAME).write_text(
            json.dumps({"chunks_written": total, "shard_no": shard_no})
        )

    # Chunk text and metadata, one JSON object per line, in the same order as
    # the vectors. Retrieval can rank by vector but has nothing to display
    # without this. Written incrementally so memory stays flat.
    mode = "a" if checkpoint is not None else "w"
    with (out_dir / "payloads.jsonl").open(mode) as payloads:
        for chunk in chunks:
            if skip > 0:
                skip -= 1
                continue
            if tok_len(chunk.text) > limit:
                truncated += 1
            ids.append(chunk.id)
            shard.append(chunk.text)
            payloads.write(json.dumps({
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "index": chunk.index,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }) + "\n")
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
    # A completed run has no more use for the checkpoint, and its presence
    # is exactly what tells the next invocation to resume instead of
    # starting fresh, so a second run against a finished index_full must not
    # find one.
    (out_dir / CHECKPOINT_NAME).unlink(missing_ok=True)

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