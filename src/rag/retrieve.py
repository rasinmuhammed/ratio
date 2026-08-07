"""Retrieval: query -> ranked chunks.

Vectors were normalized at index time, so cosine similarity reduces to a dot
product and the whole search is one matrix multiply plus a top-k selection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.embed import INDEX_DIR, MODEL_NAME, load_index, load_model

logger = logging.getLogger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_K = 5

@dataclass(frozen=True, slots=True)
class Result:
    """One retrieved chunk.

    `score` is only meaningful alongside `score_type`, because the retrievers
    return incomparable units: dense cosine sits in roughly 0.5-0.8, while
    fused RRF values are around 0.03. Naming the unit keeps it from going
    implicit, the same mistake that made OVERLAP silently mean characters.
    """

    rank: int
    score: float
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]
    score_type: str = "cosine"

def rank(query_vec: np.ndarray, vectors: np.ndarray, k: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the k best matches, best first.
    argpartition finds the k largest in O(N) """

    if k <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    scores = vectors @ query_vec
    k = min(k, len(scores))

    if k < len(scores):
        idx = np.argpartition(-scores, k)[:k]
    else:
        idx = np.arange(len(scores))

    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

class Retriever:
    def __init__(
        self, 
        index_dir: Path = INDEX_DIR,
        model: SentenceTransformer | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        if not Path(index_dir).exists():
            raise FileNotFoundError(
                f"no index at {index_dir}. Run scripts/build_index.py first."
            )

        self.vectors, self.ids, self.meta = load_index(Path(index_dir))
        self.payloads = _load_payloads(Path(index_dir), len(self.ids))

        # A mismatched model produces correctly shaped, meaningless scores, so failing here.
        if self.meta["model"] != model_name:
            raise ValueError(
                f"index was built with {self.meta['model']!r}, "
                f"but {model_name!r} was requested"
            )

        self.model = model or load_model(model_name)

        if self.model.get_embedding_dimension() != self.meta["dim"]:
            raise ValueError(
                f"index was built with {self.meta['dim']} dimensions, "
                f"but {self.model.get_embedding_dimension()} were requested"
            )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=self.meta.get("normalized", True),
            )

    def search(self, query: str, k: int = DEFAULT_K) -> list[Result]:
        if not query or not query.strip():
            raise ValueError("empty query")

        idx, scores = rank(self.embed_query(query), self.vectors, k)

        results = []
        for position, (i, score) in enumerate(zip(idx, scores), start=1):
            payload = self.payloads[int(i)]
            results.append(Result(
                rank=position,
                score=float(score),
                chunk_id=payload["id"],
                doc_id=payload["doc_id"],
                text=payload["text"],
                metadata=payload["metadata"],
            ))
        return results

    def __len__(self) -> int:
        return len(self.ids)

def _load_payloads(index_dir: Path, expected: int) -> list[dict]:
    path = index_dir / "payloads.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Rebuild the index so chunk text is stored."
        )
    with path.open() as f:
        payloads = [json.loads(line) for line in f]

    if len(payloads) != expected:
        raise ValueError(f"{path} has {len(payloads)} entries, expected {expected}")

    return payloads