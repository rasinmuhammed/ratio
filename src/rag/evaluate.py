"""Retrieval evaluation."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.generate import Searcher

LABELS_PATH = Path("data/labels.json")


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    """A query paired with the chunks that genuinely answer it.

    `kind` splits exact-match from conceptual queries. This matters more than
    it looks: hybrid weighting helped exact-match retrieval and may well have
    hurt conceptual retrieval, and a single averaged score would hide that
    completely. Reporting per kind is what turns a benchmark into a finding.
    """

    query: str
    relevant_chunk_ids: list[str]
    kind: str = "conceptual"
    note: str = ""


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: str
    kind: str
    recall: float
    precision: float
    reciprocal_rank: float
    retrieved: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """Of the chunks that should have been found, what fraction were?

    Answers "did we miss anything". A system returning every chunk in the
    corpus scores 1.0, which is why this is never reported alone.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """Of the chunks returned, what fraction should have been?

    The counterweight to recall. Returning everything tanks this, which is
    what stops recall from being gamed.
    """
    if k <= 0:
        return 0.0
    relevant = set(relevant)
    top = retrieved[:k]
    if not top:
        return 0.0
    return len([c for c in top if c in relevant]) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: Iterable[str]) -> float:
    """1/rank of the first relevant hit. Rank 1 scores 1.0, rank 5 scores 0.2.

    Unlike recall, this cares where in the list the answer landed. That is
    the right emphasis here: a user reads the first result or two, and an
    answer buried at rank 8 may as well not have been retrieved. Averaged
    over queries this is MRR.
    """
    relevant = set(relevant)
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def evaluate(searcher: Searcher, queries: list[LabelledQuery],
             k: int = 5) -> list[QueryResult]:
    """Score one retriever over a labelled query set.

    Takes anything satisfying the Searcher protocol, so dense, keyword and
    hybrid all drop in with no adapter. That protocol was written for
    generate.answer and is now paying for itself a second time.
    """
    results = []
    print(f"Evaluating {searcher.__class__.__name__} on {len(queries)} queries...")
    for i, labelled in enumerate(queries):
        if i % 5 == 0:
            print(f"  Query {i}/{len(queries)}")
        retrieved = [r.chunk_id for r in searcher.search(labelled.query, k=k)]
        results.append(QueryResult(
            query=labelled.query,
            kind=labelled.kind,
            recall=recall_at_k(retrieved, labelled.relevant_chunk_ids, k),
            precision=precision_at_k(retrieved, labelled.relevant_chunk_ids, k),
            reciprocal_rank=reciprocal_rank(retrieved, labelled.relevant_chunk_ids),
            retrieved=retrieved,
        ))
    return results


def summarise(results: list[QueryResult]) -> dict[str, dict[str, float]]:
    """Mean of each metric, overall and split by query kind."""
    def stats(rows: list[QueryResult]) -> dict[str, float]:
        if not rows:
            return {"n": 0, "recall": 0.0, "precision": 0.0, "mrr": 0.0}
        return {
            "n": len(rows),
            "recall": statistics.mean(r.recall for r in rows),
            "precision": statistics.mean(r.precision for r in rows),
            "mrr": statistics.mean(r.reciprocal_rank for r in rows),
        }

    out = {"all": stats(results)}
    for kind in sorted({r.kind for r in results}):
        out[kind] = stats([r for r in results if r.kind == kind])
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_labels(queries: list[LabelledQuery], path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"query": q.query, "relevant_chunk_ids": q.relevant_chunk_ids,
         "kind": q.kind, "note": q.note}
        for q in queries
    ]
    path.write_text(json.dumps(payload, indent=2))


def load_labels(path: Path = LABELS_PATH) -> list[LabelledQuery]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} missing. Run scripts/build_labels.py first."
        )
    raw: list[dict[str, Any]] = json.loads(Path(path).read_text())
    return [LabelledQuery(**row) for row in raw]