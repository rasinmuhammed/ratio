"""A minimal per-key sliding-window rate limiter.

Hand-rolled rather than a dependency (slowapi and friends exist) because
this project's own stance throughout has been to build the pieces it
actually needs rather than pull in a framework for something this small,
and a sliding window over a deque is maybe thirty lines. The real reason
this exists at all: every request to /query, /stream and /agent_stream
costs a real LLM call against a paid or rate-limited provider. Without
this, a public link is an open invitation to exhaust that quota for
everyone, not just slow the demo down.

Known, accepted limitation: `_hits` grows one entry per distinct key ever
seen and nothing evicts an idle key's empty deque. For a portfolio-scale
demo this is not worth the added complexity of a cleanup sweep; for
meaningfully more traffic than that, this is the first thing to revisit,
recorded here rather than silently assumed away.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


def client_key(request: Request) -> str:
    """request.client.host is the direct TCP peer, which is the load
    balancer's address rather than the real visitor behind most hosting
    platforms (Cloud Run, HF Spaces, etc. all sit behind a proxy). Prefer
    the standard forwarded-for header when present rather than rate
    limiting one shared proxy IP for every actual visitor.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_dependency(limiter: RateLimiter):
    """Returns a FastAPI dependency bound to `limiter`, so each rate-limited
    endpoint can share one limiter instance (one shared budget) or use
    separate ones, the caller's choice rather than a global singleton.
    """

    async def check(request: Request) -> None:
        key = client_key(request)
        if not limiter.allow(key):
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded: max {limiter.max_requests} "
                       f"requests per {limiter.window_seconds:.0f}s",
            )

    return check
