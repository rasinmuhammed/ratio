"""Corrective retrieval: reformulate and retry when the first pass fails.

When `refused` is True:
  1. Decompose the query into 2-3 sub-questions using the LLM.
  2. Retrieve independently for each sub-question.
  3. Fuse and dedup the candidate sets.
  4. Re-run generation on the richer context.
"""
from __future__ import annotations
import logging

from rag.generate import Answer, LLM, answer
from rag.retrieve import Result

log = logging.getLogger(__name__)

class _StaticRetriever:
    """A dummy retriever that just returns the pre-calculated chunks."""
    def __init__(self, chunks: list[Result]):
        self.chunks = chunks

    def search(self, query: str, k: int = 5) -> list[Result]:
        return self.chunks[:k]

def decompose_query(query: str, llm: LLM) -> list[str]:
    """Break a complex query into simpler sub-questions."""
    system = (
        "Break this legal question into 2-3 independent sub-questions. "
        "Each must be answerable on its own. Return one per line, no bullets or numbering."
    )
    # Use a relatively low temperature for consistent structure
    reply = llm.complete(system, query)
    
    # Clean up bullets/numbers if the LLM adds them anyway
    sub_queries = []
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove leading "1. " or "- " or "* "
        import re
        line = re.sub(r"^[\d\.\-\*\s]+", "", line)
        if line:
            sub_queries.append(line)
            
    return sub_queries[:3]

def corrective_answer(
    query: str,
    retriever, # Any retriever matching the Searcher protocol
    llm: LLM,
    first_pass: Answer,
    k: int = 6,
) -> Answer | None:
    """Return a corrected Answer if the first pass refused; None otherwise."""
    if not first_pass.refused:
        return None
        
    log.info("First pass refused. Activating Corrective RAG (CRAG)...")
    
    sub_queries = decompose_query(query, llm)
    log.info(f"Decomposed into {len(sub_queries)} sub-queries:")
    for sq in sub_queries:
        log.info(f"  - {sq}")
        
    seen: set[str] = set()
    all_chunks = []
    
    # Retrieve for each sub-question
    for sub in sub_queries:
        for c in retriever.search(sub, k=k):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                all_chunks.append(c)
                
    if not all_chunks:
        log.warning("CRAG retrieved no additional chunks.")
        return None
        
    log.info(f"CRAG retrieved {len(all_chunks)} unique chunks. Regenerating answer...")
    
    # Sort them by their original retrieval scores to bring best to top
    all_chunks.sort(key=lambda x: x.score, reverse=True)
    
    return answer(query, _StaticRetriever(all_chunks), llm, k=k)
