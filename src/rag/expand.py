"""Parent-child retrieval: match small, generate with more.

Every chunk in this system does two jobs at once: it is the unit an
embedding matches a query against, and it is the unit handed to the
generator as a source. Those jobs want opposite things. A chunk small
enough to embed as one clean idea is often too small to carry the full
holding a claim needs to cite. A chunk big enough to carry that holding
blurs the embedding across whatever else got packed in alongside it.

This module does not touch chunking, indexing or embeddings. It wraps
whatever Searcher already runs, and after that Searcher has done the actual
matching on the small, precise chunk, it widens only the *text* of each hit
to include its immediate neighbours from the same document before that text
reaches the generator. Rank and score, which reflect the precise match, are
left untouched. What grows is only what the generator gets to read.

chunk_id is `f"{doc_id}#{index}"` (see chunk.chunk_document), so a
neighbour's id is computable directly, no separate structure to build or
keep in sync with the index.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from rag.retrieve import DEFAULT_K, Result


class Searcher(Protocol):
    def search(self, query: str, k: int = DEFAULT_K, **kwargs) -> list[Result]: ...


class ParentExpandingRetriever:
    """Wraps any Searcher. Each hit's text grows to include `window` chunks
    on either side from the same document, joined in document order.

    `window=1` roughly triples the text a single source carries (three
    ~450-token chunks instead of one), which is the entire point, but it
    also means fewer distinct sources fit inside build_prompt's fixed
    CONTEXT_BUDGET. This is a real tradeoff, not a free win: expansion
    trades source *breadth* for source *sufficiency*. Left at the default
    budget it can silently starve the sources that used to fit.

    The header (court, title, case number) that chunk_document prepends to
    every chunk gets repeated once per joined neighbour, since payloads
    store only the combined text, not prefix and body separately. A few
    dozen redundant characters per extra neighbour, harmless for the
    generator to read twice, not worth the extra index-time complexity of
    storing them apart to save it.
    """

    def __init__(self, inner: Searcher, by_id: dict[str, dict], window: int = 1
                 ) -> None:
        if window < 0:
            raise ValueError(f"window ({window}) must not be negative")
        self.inner = inner
        self.by_id = by_id
        self.window = window

    @property
    def payloads(self) -> list[dict]:
        # Added when api.py started wiring this into the live retriever
        # chain (previously only scripts/audit_answers.py used it directly).
        # /health reads state.retriever.payloads for the served chunk count;
        # delegating keeps one source of truth rather than tracking a copy.
        return self.inner.payloads

    def search(self, query: str, k: int = DEFAULT_K, **kwargs) -> list[Result]:
        return [self._expand(r) for r in self.inner.search(query, k=k, **kwargs)]

    def _expand(self, result: Result) -> Result:
        doc_id, sep, index_str = result.chunk_id.rpartition("#")
        # rpartition on a doc_id lacking '#' returns ("", "", chunk_id): sep
        # is empty exactly then, and index_str is not the index at all.
        if not sep or not index_str.isdigit():
            return result

        center = int(index_str)
        pieces: list[str] = []
        for i in range(center - self.window, center + self.window + 1):
            sibling = self.by_id.get(f"{doc_id}#{i}")
            if sibling is not None:
                pieces.append(sibling["text"])

        # Only the matched chunk existed (start/end of document, or a
        # window of 0). Nothing to expand into; return the hit unchanged
        # rather than a "widened" text identical to the original.
        if len(pieces) <= 1:
            return result

        return replace(result, text="\n\n".join(pieces))
