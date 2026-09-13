from rag.ratelimit import RateLimiter, client_key


class FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class FakeRequest:
    def __init__(self, host: str | None, forwarded_for: str | None = None) -> None:
        self.client = FakeClient(host) if host else None
        self.headers = {"x-forwarded-for": forwarded_for} if forwarded_for else {}


def test_allows_up_to_max_requests_in_window():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("ip1")
    assert limiter.allow("ip1")
    assert limiter.allow("ip1")
    assert not limiter.allow("ip1")


def test_different_keys_have_independent_budgets():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("ip1")
    assert not limiter.allow("ip1")
    assert limiter.allow("ip2")  # unaffected by ip1's budget


def test_old_hits_expire_out_of_the_window():
    # allow() calls time.time() directly, so expiry is exercised here by
    # manipulating the recorded hit timestamp rather than monkeypatching
    # time itself.
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("ip1")
    assert not limiter.allow("ip1")
    limiter._hits["ip1"][0] -= 11  # push the recorded hit outside the window
    assert limiter.allow("ip1")


def test_client_key_prefers_forwarded_for_over_direct_peer():
    request = FakeRequest(host="10.0.0.1", forwarded_for="203.0.113.5, 10.0.0.1")
    assert client_key(request) == "203.0.113.5"


def test_client_key_falls_back_to_direct_peer():
    request = FakeRequest(host="203.0.113.5")
    assert client_key(request) == "203.0.113.5"


def test_client_key_handles_missing_client():
    request = FakeRequest(host=None)
    assert client_key(request) == "unknown"
