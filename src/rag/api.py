"""FastAPI service for Ratio RAG engine.

Loads the full index into memory once at startup, eliminating the 85-second 
penalty on every CLI query. Provides synchronous and streaming (SSE) endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag.agent import LegalAgent
from rag.audit import init_db, log_request, log_correction
from rag.cache import SemanticCache
from rag.corrective import corrective_answer
from rag.generate import (
    CONTEXT_BUDGET, SYSTEM_PROMPT, answer, build_prompt, get_llm, parse_answer,
)
from rag.hybrid import HybridRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever
from rag.route import RoutedRetriever, classify
from rag.stance import classify as stance_of

# data/index-sweep-450 was a leftover from the chunk-size sweep experiment
# (2,767 chunks) rather than the real corpus, and was silently the default
# here: every live query was answered from 0.7% of the indexed corpus while
# the landing page advertised 414,122 chunks. data/index (6,476 chunks) is
# a real, complete index built the same way as the full one, just smaller.
# It is not the full corpus either — data/index-full has the vectors for
# all 414,122 chunks but is missing payloads.jsonl and cannot load yet — so
# this is the honest default until that rebuild happens, not the finished
# one. Override with RATIO_INDEX_DIR for anything else.
INDEX_DIR = Path(os.environ.get("RATIO_INDEX_DIR", "data/index"))

logger = logging.getLogger(__name__)

class State:
    """Application state loaded once at startup."""
    retriever: Any
    llm: Any
    cache: SemanticCache

state = State()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index and models."""
    logger.info("Initializing SQLite audit log...")
    init_db()

    logger.info("Loading index from %s...", INDEX_DIR)
    started = time.time()
    hybrid = HybridRetriever(INDEX_DIR)
    reranked = RerankedRetriever(hybrid, CrossEncoderReranker())
    state.retriever = RoutedRetriever(reranked, hybrid.dense.payloads)
    state.llm = get_llm()
    # hybrid.dense already carries the loaded bge-small model and its
    # normalisation settings, so the cache embeds queries the exact same
    # way retrieval does rather than loading a second copy of the model.
    state.cache = SemanticCache(hybrid.dense)
    logger.info(
        "Ready in %.0fs (%d chunks)",
        time.time() - started,
        len(hybrid.dense.payloads)
    )
    yield

app = FastAPI(title="Ratio RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    k: int = 6
    stance_notes: bool = True

class CorrectionRequest(BaseModel):
    query: str
    original_answer: str
    corrected_answer: str

@app.get("/cache_stats")
async def cache_stats() -> dict:
    """Real hit/miss counts, not a claim. Existed nowhere before this: a
    cache with no visible hit rate is indistinguishable from a cache that
    never fires at all."""
    return {
        "entries": len(state.cache),
        "hits": state.cache.stats.hits,
        "misses": state.cache.stats.misses,
        "hit_rate": state.cache.stats.hit_rate,
    }

@app.post("/submit_correction")
async def submit_correction(req: CorrectionRequest):
    """Save human feedback for Direct Preference Optimization (DPO)."""
    log_correction(req.query, req.original_answer, req.corrected_answer)
    return {"status": "success", "message": "Correction saved to training dataset."}

class SourceResponse(BaseModel):
    id: str
    rank: int
    score: float
    text: str
    court: str
    title: str
    cited_by: int
    url: str | None

class AnswerResponse(BaseModel):
    query: str
    text: str
    refused: bool
    sources: list[SourceResponse]
    cited_source_indices: list[int]
    crag_used: bool

@app.post("/query", response_model=AnswerResponse)
async def query_endpoint(req: QueryRequest):
    """Synchronous generation. Waits for the full K2-Horizon response."""
    # Keyed on query text alone, not (query, k, stance_notes): a cache hit
    # answers slightly loosely rather than never, which matches what this
    # cache is for (cutting repeat-question latency), not a strict
    # per-parameter memoizer. Two requests for the same question with
    # different k could share a cached answer generated at a different k;
    # documented here rather than silently assumed correct.
    cached = state.cache.get(req.query)
    if cached is not None:
        return cached

    start_time = time.time()
    route = classify(req.query)

    t0 = time.time()
    # 1. First Pass Retrieval & Generation
    result = answer(
        req.query, 
        state.retriever, 
        state.llm, 
        k=req.k,
        stance_notes=req.stance_notes
    )
    retriever_ms = int((time.time() - t0) * 1000)
    
    # 2. Corrective RAG (CRAG) Fallback
    crag_used = False
    t1 = time.time()
    if result.refused:
        logger.info("First pass refused. Activating CRAG...")
        crag_result = corrective_answer(req.query, state.retriever, state.llm, result, k=req.k)
        if crag_result and not crag_result.refused:
            result = crag_result
            crag_used = True
    generation_ms = int((time.time() - t1) * 1000)
    
    # 3. Format Response
    sources_resp = []
    for rank, s in enumerate(result.sources, start=1):
        sources_resp.append(SourceResponse(
            id=s.chunk_id,
            rank=rank,
            score=s.score,
            text=s.text,
            court=s.metadata.get("court", ""),
            title=s.metadata.get("title", ""),
            cited_by=s.metadata.get("cited_by", 0),
            url=s.metadata.get("url")
        ))
        
    # 4. Audit Log
    log_request(
        query=req.query,
        route=route,
        refused=result.refused,
        n_sources=len(result.sources),
        n_cited=len(result.cited),
        invalid_cites=result.invalid_citations,
        score_gap=result.score_gap,
        truncated=result.truncated,
        retriever_ms=retriever_ms,
        generation_ms=generation_ms,
        llm_model=getattr(state.llm, "model", "unknown")
    )
    
    response = AnswerResponse(
        query=req.query,
        text=result.text,
        refused=result.refused,
        sources=sources_resp,
        cited_source_indices=result.cited,
        crag_used=crag_used
    )
    state.cache.put(req.query, response)
    return response

@app.get("/stream")
async def stream_endpoint(request: Request, query: str, k: int = 6, stance_notes: bool = True):
    """Server-Sent Events (SSE) streaming endpoint."""
    async def event_generator():
        route = classify(query)
        yield {
            "event": "status",
            "data": json.dumps({"message": f"Routed as {route}. Retrieving chunks..."})
        }
        
        # Retrieval
        t0 = time.time()
        chunks = state.retriever.search(query, k=k)
        if not chunks:
            yield {"event": "status", "data": json.dumps({"message": "No chunks found."})}
            return
            
        system, used_chunks = build_prompt(
            query, chunks, budget=CONTEXT_BUDGET, stance_notes=stance_notes
        )
        retriever_ms = int((time.time() - t0) * 1000)
        
        yield {
            "event": "status",
            "data": json.dumps({"message": f"Generating answer using {len(used_chunks)} sources..."})
        }
        
        # Generation (Streaming)
        t1 = time.time()
        full_text = ""
        
        if hasattr(state.llm, "stream"):
            for token in state.llm.stream(system, query):
                if await request.is_disconnected():
                    break
                full_text += token
                yield {"event": "delta", "data": json.dumps({"token": token})}
        else:
            # Fallback for models without stream() implemented
            full_text = state.llm.complete(system, query)
            yield {"event": "delta", "data": json.dumps({"token": full_text})}
            
        generation_ms = int((time.time() - t1) * 1000)
        
        # Send Sources metadata at the end
        sources_meta = []
        for i, s in enumerate(used_chunks, start=1):
            sources_meta.append({
                "index": i,
                "court": s.metadata.get("court", ""),
                "title": s.metadata.get("title", ""),
                "url": s.metadata.get("url")
            })
            
        # Refusal check and audit log
        from rag.generate import REFUSAL
        refused = REFUSAL in full_text
        
        log_request(
            query=query,
            route=route,
            refused=refused,
            n_sources=len(used_chunks),
            n_cited=0, # Hard to calculate cleanly during stream without regex parsing here
            invalid_cites=[],
            score_gap=used_chunks[0].score - used_chunks[len(used_chunks)//2].score if used_chunks else 0.0,
            truncated=False,
            retriever_ms=retriever_ms,
            generation_ms=generation_ms,
            llm_model=getattr(state.llm, "model", "unknown")
        )

        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources_meta, "refused": refused})
        }
        
    return EventSourceResponse(event_generator())

@app.get("/agent_stream")
async def agent_stream_endpoint(request: Request, query: str, k: int = 6):
    """SSE streaming endpoint for the Autonomous LegalAgent."""
    async def event_generator():
        # A cache hit skips the agent loop entirely: no retrieval, no
        # LLM call, an SSE sequence shaped exactly like a real answer's so
        # the frontend needs no special case for it. SemanticCache routes
        # identifier queries to exact-string matching and conceptual
        # queries to embedding similarity above a conservative threshold,
        # specifically so a citation query is never served another
        # citation's cached answer just because the two embed similarly.
        cached = state.cache.get(query)
        if cached is not None:
            yield {
                "event": "status",
                "data": json.dumps({"message": "Served from cache."})
            }
            yield {
                "event": "delta",
                "data": json.dumps({"token": cached["answer"]})
            }
            yield {
                "event": "sources",
                "data": json.dumps({
                    "sources": cached["sources"],
                    "refused": cached["refused"],
                    "invalid_citations": cached["invalid_citations"],
                })
            }
            return

        agent = LegalAgent(state.llm, state.retriever, k=k)

        yield {
            "event": "status",
            "data": json.dumps({"message": "Agent initialized. Analyzing query..."})
        }

        final_answer = ""
        for event in agent.stream(query):
            if await request.is_disconnected():
                break
                
            if event["type"] == "scratchpad":
                yield {
                    "event": "scratchpad",
                    "data": json.dumps({"content": event["content"]})
                }
            elif event["type"] == "search":
                yield {
                    "event": "search",
                    "data": json.dumps({"query": event["content"]})
                }
            elif event["type"] == "final_answer":
                final_answer = event["content"]
                # In agent mode, the final answer isn't streamed token-by-token yet because
                # IFM chat doesn't support streaming tools + text natively in our wrapper.
                # So we just dump the whole answer.
                yield {
                    "event": "delta",
                    "data": json.dumps({"token": final_answer})
                }
            elif event["type"] == "error":
                yield {
                    "event": "status",
                    "data": json.dumps({"message": f"Error: {event['content']}"})
                }
                
        # parse_answer reads the same [n] markers generate.answer scores
        # citation precision with. format_search_results now numbers every
        # source by its position in agent.all_sources (see that docstring),
        # so those markers finally refer to a stable, real index instead of
        # restarting at [1] on every search call, and this is the first
        # point the agent's output has been checked against its sources at
        # all rather than trusted as-is.
        refused, cited, invalid_citations = parse_answer(
            final_answer, len(agent.all_sources))

        # Send sources so UI can render the EvidencePanel, now carrying the
        # same stance and authority signal ask.py has always printed to a
        # terminal, plus whether the model actually cited each one.
        sources_meta = []
        for i, s in enumerate(agent.all_sources.values(), start=1):
            sources_meta.append({
                "index": i,
                "court": s.metadata.get("court", ""),
                "title": s.metadata.get("title", ""),
                "url": s.metadata.get("url"),
                "cited_by": s.metadata.get("cited_by", 0),
                "stance": stance_of(s.text),
                "cited": i in cited,
            })

        # Only a real answer is cached. final_answer stays "" when the
        # agent hit the turn limit or the client disconnected mid-loop
        # (the "error" branch above never sets it), and caching an empty
        # or partial result would mean the next matching query gets
        # served that failure instead of a fresh attempt.
        if final_answer:
            state.cache.put(query, {
                "answer": final_answer,
                "sources": sources_meta,
                "refused": refused,
                "invalid_citations": invalid_citations,
            })

        yield {
            "event": "sources",
            "data": json.dumps({
                "sources": sources_meta,
                "refused": refused,
                "invalid_citations": invalid_citations,
            })
        }

    return EventSourceResponse(event_generator())
