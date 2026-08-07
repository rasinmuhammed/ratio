import pytest

from rag.generate import REFUSAL, answer, build_prompt, parse_answer
from rag.retrieve import Result


def _chunk(i: int, text: str, score: float = 0.7) -> Result:
    return Result(rank=i, score=score, chunk_id=f"d#{i}", doc_id="d",
                  text=text, metadata={})


class FakeLLM:
    """Satisfies the LLM protocol without a network call. This is the reason
    LLM is a Protocol rather than a concrete class."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_user = user
        return self.reply


class FakeSearcher:
    def __init__(self, results: list[Result]) -> None:
        self.results = results

    def search(self, query: str, k: int = 5) -> list[Result]:
        return self.results[:k]


def test_prompt_numbers_sources_from_one():
    prompt, used = build_prompt("q", [_chunk(1, "alpha"), _chunk(2, "beta")])
    assert "[1] alpha" in prompt and "[2] beta" in prompt
    assert len(used) == 2


def test_prompt_stops_at_budget():
    """Budget is measured by the injected length function, so a chunk that
    would overflow stops assembly rather than being silently truncated."""
    chunks = [_chunk(i, "x" * 100) for i in range(1, 6)]
    _, used = build_prompt("q", chunks, budget=250)
    assert 0 < len(used) < 5


def test_prompt_always_includes_at_least_one_chunk():
    """Even an oversized first chunk goes in. An empty context guarantees a
    refusal, which is worse than an over-long prompt."""
    _, used = build_prompt("q", [_chunk(1, "x" * 10_000)], budget=10)
    assert len(used) == 1


def test_prompt_handles_no_chunks():
    prompt, used = build_prompt("q", [])
    assert used == []
    assert "no sources retrieved" in prompt


def test_detects_refusal():
    refused, cited, invalid = parse_answer(REFUSAL, 3)
    assert refused and cited == [] and invalid == []


def test_extracts_citations_in_order_without_duplicates():
    refused, cited, invalid = parse_answer("A [2] and B [1], also [2].", 3)
    assert not refused
    assert cited == [2, 1]


def test_flags_hallucinated_citations():
    """Models cite [7] when given four sources. Cheap to catch, and a
    countable quality signal for M2."""
    _, cited, invalid = parse_answer("As held in [7] and [1].", 4)
    assert cited == [1]
    assert invalid == [7]

# ---------------------------------------------------------------------------
# answer(): the wiring. Untested until now, and it was broken.
# ---------------------------------------------------------------------------


def test_answer_returns_sources_actually_sent():
    """This is the test that would have caught the Answer(source=/sources=)
    mismatch. Every pure function was covered; the wiring never ran."""
    chunks = [_chunk(1, "alpha"), _chunk(2, "beta")]
    result = answer("q", FakeSearcher(chunks), FakeLLM("Held in [1]."))

    assert result.sources == chunks
    assert result.cited == [1]
    assert result.invalid_citations == []
    assert not result.refused


def test_answer_detects_refusal():
    result = answer("q", FakeSearcher([_chunk(1, "alpha")]), FakeLLM(REFUSAL))
    assert result.refused
    assert result.cited == []


def test_answer_flags_hallucinated_citation():
    result = answer("q", FakeSearcher([_chunk(1, "alpha")]),
                    FakeLLM("As held in [3]."))
    assert result.invalid_citations == [3]
    assert result.cited == []


def test_answer_handles_no_results():
    """Empty retrieval must not divide by zero computing the score gap."""
    result = answer("q", FakeSearcher([]), FakeLLM(REFUSAL))
    assert result.sources == []
    assert result.score_gap == 0.0


def test_answer_rejects_empty_query():
    with pytest.raises(ValueError, match="empty"):
        answer("   ", FakeSearcher([]), FakeLLM("x"))


def test_answer_computes_score_gap():
    """Gap is top minus median, logged as a diagnostic for M2 calibration."""
    chunks = [_chunk(1, "a", 0.8), _chunk(2, "b", 0.6), _chunk(3, "c", 0.5)]
    result = answer("q", FakeSearcher(chunks), FakeLLM("x"))
    assert result.score_gap == pytest.approx(0.2)


def test_answer_passes_numbered_sources_to_the_model():
    llm = FakeLLM("ok")
    answer("q", FakeSearcher([_chunk(1, "alpha")]), llm)
    assert "[1] alpha" in llm.last_user
