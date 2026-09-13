"""API-layer tests for src/rag/api.py.

Deliberately does not trigger FastAPI's lifespan (that loads a real
embedding model, a real cross-encoder and a real index, exactly the
memory-heavy path this project spent a long night fighting locally).
TestClient(app) used without a `with` block never runs lifespan startup,
so state is populated directly with fakes instead, and every test here
runs in milliseconds with no model, no index, no network. This is the
first test coverage the API layer itself has had; agent.py's own logic
was already tested, but nothing before this exercised the actual HTTP
behaviour: status codes, validation, or what a client sees when
something fails.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rag.api as api
from rag.cache import CacheStats


class FakeCache:
    def __init__(self) -> None:
        self.stats = CacheStats()
        self.put_calls: list[tuple[str, object]] = []
        self._canned: object | None = None

    def get(self, query: str):
        if self._canned is not None:
            self.stats.hits += 1
            return self._canned
        self.stats.misses += 1
        return None

    def put(self, query: str, payload) -> None:
        self.put_calls.append((query, payload))

    def __len__(self) -> int:
        return 0


class FakeRetriever:
    payloads = [{"id": i} for i in range(6476)]

    def search(self, query: str, k: int = 6):
        return []


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply: str = "The rule is X [1].") -> None:
        self.reply = reply

    def complete(self, system: str, user: str) -> str:
        return self.reply


class ExplodingLLM:
    """Simulates every provider's own failure mode: RuntimeError once
    retries are exhausted, generate.py's documented behaviour for a rate
    limit, a connection reset, or a status held past every retry."""

    model = "exploding-llm"

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("Groq gave status 503 after all retries: ...")


@pytest.fixture(autouse=True)
def reset_shared_state():
    """The rate limiter and cache are module-level singletons in api.py,
    so without this, tests would share a rate-limit budget and leak
    between each other depending on run order."""
    api._query_rate_limiter._hits.clear()
    api.state.cache = FakeCache()
    api.state.retriever = FakeRetriever()
    api.state.llm = FakeLLM()
    yield


@pytest.fixture
def client():
    return TestClient(api.app, raise_server_exceptions=False)


def test_health_reports_chunk_count(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks_indexed": 6476}


def test_cache_stats_reflects_real_counts(client):
    api.state.cache.stats.hits = 3
    api.state.cache.stats.misses = 1
    r = client.get("/cache_stats")
    assert r.status_code == 200
    body = r.json()
    assert body["hits"] == 3
    assert body["misses"] == 1
    assert body["hit_rate"] == pytest.approx(0.75)


def test_query_rejects_empty_string():
    """QueryRequest.query has min_length=1; this used to accept an empty
    query and let it fail somewhere deeper instead of at the boundary."""
    client = TestClient(api.app, raise_server_exceptions=False)
    r = client.post("/query", json={"query": ""})
    assert r.status_code == 422


def test_query_rejects_k_out_of_range(client):
    r = client.post("/query", json={"query": "a valid question", "k": 999})
    assert r.status_code == 422


def test_stream_rejects_empty_query(client):
    r = client.get("/stream", params={"query": "   "})
    assert r.status_code == 400


def test_agent_stream_rejects_empty_query(client):
    r = client.get("/agent_stream", params={"query": ""})
    assert r.status_code == 400


def test_llm_failure_returns_503_not_a_raw_500(client):
    """Before this, generate.py's RuntimeError (every provider's own
    exhausted-retries signal) propagated as FastAPI's bare default 500
    with no clean message. This is a real, expected failure mode for a
    paid third-party API, not a bug, and should read as one."""
    api.state.llm = ExplodingLLM()
    r = client.post("/query", json={"query": "a valid question"})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


def test_query_cache_hit_skips_generation_entirely(client):
    cached_response = {
        "query": "a repeated question",
        "text": "cached answer [1].",
        "refused": False,
        "sources": [],
        "cited_source_indices": [1],
        "crag_used": False,
    }
    api.state.cache._canned = cached_response
    api.state.llm = ExplodingLLM()  # would raise if generation were reached

    r = client.post("/query", json={"query": "a repeated question"})
    assert r.status_code == 200
    assert r.json() == cached_response


def test_rate_limit_returns_429_after_the_configured_max(client):
    limiter = api._query_rate_limiter
    original_max = limiter.max_requests
    limiter.max_requests = 2
    try:
        for _ in range(2):
            r = client.post("/query", json={"query": "q"})
            assert r.status_code == 200
        r = client.post("/query", json={"query": "q"})
        assert r.status_code == 429
    finally:
        limiter.max_requests = original_max


def test_unhandled_exception_returns_clean_json_not_a_stack_trace(client):
    """A bug that isn't LLM-shaped (a genuine AttributeError, say) should
    still never leak an internal traceback to the client."""
    class BrokenRetriever:
        def search(self, query, k=6):
            raise AttributeError("something genuinely broke")

    api.state.retriever = BrokenRetriever()
    r = client.post("/query", json={"query": "a valid question"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal error, the team has been notified"}
