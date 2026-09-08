"""Contextual retrieval: situate a chunk before it gets embedded.

chunk.context_prefix() already attaches identity to every chunk, court,
title, case number, for free, mechanically, no LLM needed. What it can't
supply is substance: what is this case actually about, what issue, what
outcome. A chunk deep in a judgment ("the third contention was rejected for
want of notice") carries none of that on its own, and an embedding is
computed purely from the chunk's own text, so a query about notice
requirements in eviction proceedings has no way to find this chunk unless
the chunk happens to restate that it's an eviction case, which it usually
does not.

The fix is one LLM call per *document*, not per chunk. The published version
of this idea situates each chunk individually using the whole document as
context; at this corpus's scale, roughly 12,000 documents and 414,000
chunks, that's a 35x cost difference for a marginal gain (finer sub-argument
positioning) this corpus doesn't need as much as it needs the basic "what
is this case about" that chunk_prefix leaves out. One summary per document,
reused across all of that document's chunks, covers the real gap for a
fraction of the spend.

This changes what gets embedded, so using it means a full reindex, not a
metadata-only change. That's the real cost here, not the LLM calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol

from rag.ingest import Document

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/document_summaries.json")

# Fed the document's own text truncated to this many characters. Judgments
# front-load the caption, parties and often a synopsis; the outcome and
# central holding are usually reachable well within this window without
# needing the full document, which for some judgments runs to tens of
# thousands of characters and would cost tokens for no marginal accuracy.
MAX_CHARS = 6000

SYSTEM_PROMPT = """You summarise an Indian court judgment in 2-3 sentences \
for a search index: who the parties are (in general terms, not full names \
unless short), what court decided it, what legal question was at issue, \
and what the outcome was.

Rules:
- Plain prose, no headings, no bullet points, no preamble.
- Specific enough to distinguish this case from a similar one: name the \
statute, section or legal doctrine actually at issue where the judgment \
does.
- If the text is too short or too damaged to summarise, say so in one \
sentence rather than inventing content.
"""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def summarize_document(doc: Document, llm: LLM) -> str:
    return llm.complete(SYSTEM_PROMPT, doc.text[:MAX_CHARS]).strip()


class SummaryCache:
    """Persists doc_id -> summary to disk. At roughly 12,000 documents and
    real per-call latency, a rerun that forgot what it already paid for
    would be the expensive kind of bug. Written incrementally, one save per
    new summary, not batched at the end, so a killed run loses at most the
    document in flight, not everything since the last checkpoint.
    """

    def __init__(self, path: Path = CACHE_PATH) -> None:
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def get(self, doc_id: str) -> str | None:
        return self.data.get(doc_id)

    def set(self, doc_id: str, summary: str) -> None:
        self.data[doc_id] = summary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))

    def __len__(self) -> int:
        return len(self.data)


def summarize_with_cache(doc: Document, llm: LLM, cache: SummaryCache,
                         sleep=time.sleep, retries: int = 2) -> str:
    """Cache-first. A cache hit costs nothing, not even a network round
    trip, which matters at this document count if a run needs restarting.
    """
    cached = cache.get(doc.id)
    if cached is not None:
        return cached

    for attempt in range(retries):
        try:
            summary = summarize_document(doc, llm)
        except RuntimeError as exc:
            if attempt == retries - 1:
                logger.warning("summary failed for %s after %d attempts: %s",
                               doc.id, retries, exc)
                return ""  # empty, not a crash: chunk.context_prefix() alone still runs
            sleep(4 * (attempt + 1))
            continue
        cache.set(doc.id, summary)
        return summary
    return ""
