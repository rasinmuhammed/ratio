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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rag.agent import LegalAgent
from rag.audit import init_db, log_request, log_correction
from rag.cache import SemanticCache
from rag.corrective import corrective_answer
from rag.expand import ParentExpandingRetriever
from rag.generate import (
    CONTEXT_BUDGET, SYSTEM_PROMPT, answer, build_prompt, get_llm, parse_answer,
)
from rag.hybrid import HybridRetriever
from rag.index_fetch import ensure_index
from rag.ratelimit import RateLimiter, rate_limit_dependency
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

    # A no-op locally (RATIO_INDEX_REPO unset): nothing about local
    # development changes. On a fresh deploy target with no index on
    # disk, this pulls one from a HF Hub dataset repo before
    # HybridRetriever gets a chance to raise FileNotFoundError over it.
    ensure_index(INDEX_DIR, repo=os.environ.get("RATIO_INDEX_REPO"))

    logger.info("Loading index from %s...", INDEX_DIR)
    started = time.time()
    hybrid = HybridRetriever(INDEX_DIR)
    reranked = RerankedRetriever(hybrid, CrossEncoderReranker())
    routed = RoutedRetriever(reranked, hybrid.dense.payloads)
    # Small-to-big: ranking and reranking already happened above on the
    # precise 450-token chunk; this only widens what the generator reads for
    # each already-selected source, so it sits last in the chain. window=1
    # is the same default scripts/audit_answers.py's --expand-context uses;
    # this was implemented and tested against synthetic Results (2026-09-08)
    # but never run through audit_answers.py's actual answer-quality audit
    # the way HyDE was before being wired in, so treat it as provisional
    # until that comparison exists, not as a measured win.
    state.retriever = ParentExpandingRetriever(routed, hybrid.by_id, window=1)
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

# Wildcard by default so local development (frontend on any localhost port)
# never needs configuring. Once a deploy target has a real frontend origin,
# RATIO_ALLOWED_ORIGINS should be set to it, comma-separated for more than
# one; left unset, behaviour is unchanged from before this was added.
_allowed_origins = os.environ.get("RATIO_ALLOWED_ORIGINS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins.split(",") if _allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every /query, /stream and /agent_stream call costs a real LLM call
# against a paid or rate-limited provider. One shared limiter across all
# three, rather than one each, because the actual resource being protected
# (the LLM quota) is shared too. 20 requests per 5 minutes is a starting
# number for a portfolio demo, not a load-tested ceiling.
_query_rate_limiter = RateLimiter(max_requests=20, window_seconds=300)
_rate_limit = rate_limit_dependency(_query_rate_limiter)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Every endpoint below this line had zero exception handling before
    this: a malformed request, a disconnected LLM provider, or a genuine
    bug all surfaced as FastAPI's bare default 500. This doesn't make
    those conditions not happen, it makes them return a clean, generic
    JSON error instead of leaking an internal stack trace to whoever's
    looking, which matters the moment this is public rather than a
    terminal only one person sees.
    """
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error, the team has been notified"},
    )


@app.get("/health")
async def health() -> dict:
    """Whatever host this ends up on (Spaces, Cloud Run, a real VM) needs
    a cheap way to ask "is this actually up", separate from whether a
    real query would currently succeed. FastAPI doesn't route requests
    until lifespan's startup half completes, so state.retriever is always
    set by the time this can be called."""
    return {"status": "ok", "chunks_indexed": len(state.retriever.payloads)}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=6, ge=1, le=20)
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

@app.post("/query", response_model=AnswerResponse, dependencies=[Depends(_rate_limit)])
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
    try:
        result = answer(
            req.query,
            state.retriever,
            state.llm,
            k=req.k,
            stance_notes=req.stance_notes
        )
    except RuntimeError as exc:
        # Every provider in generate.py raises RuntimeError once its own
        # retries are exhausted (rate limited, connection reset, a bad
        # status held past every retry). That is a real, expected failure
        # mode for a paid third-party API, not a bug in this code, so it
        # gets a 503 the caller can reasonably retry rather than the
        # generic 500 the exception handler above would otherwise give it.
        logger.warning("LLM provider failed for query: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="the language model provider is currently unavailable, please try again",
        ) from exc
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

@app.get("/stream", dependencies=[Depends(_rate_limit)])
async def stream_endpoint(request: Request, query: str, k: int = 6, stance_notes: bool = True):
    """Server-Sent Events (SSE) streaming endpoint."""
    # query/k arrive as raw GET params, not through a validated Pydantic
    # model the way QueryRequest is, so this endpoint had no equivalent of
    # QueryRequest's min_length/ge/le constraints until now.
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not 1 <= k <= 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

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

@app.get("/agent_stream", dependencies=[Depends(_rate_limit)])
async def agent_stream_endpoint(request: Request, query: str, k: int = 6):
    """SSE streaming endpoint for the Autonomous LegalAgent."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not 1 <= k <= 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

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
