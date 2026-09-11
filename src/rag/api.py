"""FastAPI service for Ratio RAG engine.

Loads the full index into memory once at startup, eliminating the 85-second 
penalty on every CLI query. Provides synchronous and streaming (SSE) endpoints.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag.audit import init_db, log_request
from rag.corrective import corrective_answer
from rag.generate import CONTEXT_BUDGET, SYSTEM_PROMPT, answer, build_prompt, get_llm
from rag.hybrid import HybridRetriever
from rag.rerank import CrossEncoderReranker, RerankedRetriever
from rag.route import RoutedRetriever, classify

# Default to the small sweep index for testing, but can be overridden
INDEX_DIR = Path("data/index-sweep-450")

logger = logging.getLogger(__name__)

class State:
    """Application state loaded once at startup."""
    retriever: Any
    llm: Any

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
    
    return AnswerResponse(
        query=req.query,
        text=result.text,
        refused=result.refused,
        sources=sources_resp,
        cited_source_indices=result.cited,
        crag_used=crag_used
    )

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
