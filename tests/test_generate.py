import re

import httpx
import pytest

from rag.generate import (
    REFUSAL,
    GroqLLM,
    MercuryLLM,
    TokenRouterLLM,
    answer,
    answer_structured,
    build_prompt,
    parse_answer,
)
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
    """Numbering and text, not exact spacing. Each source also carries a stance
    note between the number and the text, so asserting the two adjacent would
    pin a formatting decision rather than the behaviour."""
    prompt, used = build_prompt("q", [_chunk(1, "alpha"), _chunk(2, "beta")])
    assert re.search(r"\[1\].*alpha", prompt)
    assert re.search(r"\[2\].*beta", prompt)
    assert len(used) == 2


def test_prompt_labels_each_source_with_its_stance():
    """A submission and a holding must not reach the model looking alike. The
    model cannot tell them apart from the text alone, and stating a party's
    contention as the law is the failure this exists to prevent."""
    prompt, _ = build_prompt("q", [
        _chunk(1, "Learned counsel submitted that the suit is barred."),
        _chunk(2, "We are of the opinion that the suit is barred."),
    ])
    assert "records a party's submission" in prompt
    assert "the court's own reasoning" in prompt


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
    assert re.search(r"\[1\].*alpha", llm.last_user)


# ---------------------------------------------------------------------------
# answer_structured: same contract as answer(), a different route to it.
# ---------------------------------------------------------------------------

class FakeStructuredLLM:
    """Satisfies StructuredLLM without a network call, same role FakeLLM
    plays for the free-text path."""

    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.last_user: str | None = None
        self.last_schema: dict | None = None

    def complete_structured(self, system: str, user: str, schema: dict) -> dict:
        self.last_user = user
        self.last_schema = schema
        return self.reply


def test_structured_reconstructs_text_in_the_same_bracket_format():
    """attribute.split_claims and everything downstream of it expects
    'text [n]'. A structured Answer has to produce exactly that shape even
    though nothing here ran the bracket regex to get it."""
    reply = {"refused": False, "claims": [
        {"text": "The suit was barred by limitation", "source_ids": [1]},
    ]}
    result = answer_structured("q", FakeSearcher([_chunk(1, "alpha")]),
                               FakeStructuredLLM(reply))
    assert result.text == "The suit was barred by limitation [1]"
    assert result.cited == [1]


def test_structured_joins_multiple_source_ids_on_one_claim():
    reply = {"refused": False, "claims": [
        {"text": "Both sources agree", "source_ids": [2, 1]},
    ]}
    result = answer_structured(
        "q", FakeSearcher([_chunk(1, "alpha"), _chunk(2, "beta")]),
        FakeStructuredLLM(reply),
    )
    assert result.text == "Both sources agree [2, 1]"
    assert result.cited == [2, 1]  # order of first appearance, same as parse_answer


def test_structured_detects_refusal():
    reply = {"refused": True, "claims": []}
    result = answer_structured("q", FakeSearcher([_chunk(1, "alpha")]),
                               FakeStructuredLLM(reply))
    assert result.refused
    assert result.text == REFUSAL
    assert result.cited == []


def test_structured_flags_hallucinated_citation():
    reply = {"refused": False, "claims": [
        {"text": "As held", "source_ids": [3]},
    ]}
    result = answer_structured("q", FakeSearcher([_chunk(1, "alpha")]),
                               FakeStructuredLLM(reply))
    assert result.invalid_citations == [3]
    assert result.cited == []


def test_structured_drops_a_claim_with_no_source_ids_rather_than_guess():
    """minItems: 1 in the schema should make this unreachable in practice,
    same spirit as strict:True for malformed JSON: if it happens anyway,
    the claim is dropped, not kept with a fabricated source."""
    reply = {"refused": False, "claims": [
        {"text": "unsupported claim", "source_ids": []},
        {"text": "supported claim", "source_ids": [1]},
    ]}
    result = answer_structured("q", FakeSearcher([_chunk(1, "alpha")]),
                               FakeStructuredLLM(reply))
    assert "unsupported claim" not in result.text
    assert result.text == "supported claim [1]"


def test_structured_handles_empty_dict_from_a_truncated_response():
    """complete_structured returns {} on truncation/invalid JSON rather than
    raising. answer_structured must not crash on a missing key, and treats
    the result as a refusal: refused=false with zero claims is exactly the
    silent non-answer state the audit found live and this collapse exists
    to remove, an empty dict is that same shape by construction."""
    result = answer_structured("q", FakeSearcher([_chunk(1, "alpha")]),
                               FakeStructuredLLM({}))
    assert result.text == REFUSAL
    assert result.refused
    assert result.cited == []


def test_structured_collapses_unrefused_empty_claims_into_a_refusal():
    """The exact live failure: refused=false, claims=[]. Found on 3 of 15
    real queries in the first full audit, a silent non-answer that isn't a
    refusal and isn't a real answer either, inflating precision/support by
    quietly leaving the denominator instead of counting as a miss."""
    reply = {"refused": False, "claims": []}
    result = answer_structured("q", FakeSearcher([_chunk(1, "alpha")]),
                               FakeStructuredLLM(reply))
    assert result.refused
    assert result.text == REFUSAL


def test_structured_rejects_empty_query():
    with pytest.raises(ValueError, match="empty"):
        answer_structured("   ", FakeSearcher([]), FakeStructuredLLM({}))


def test_structured_passes_the_schema_and_omits_the_bracket_reminder():
    """cite_reminder=False: asking for brackets in prose while also
    constraining to a JSON schema would contradict the schema, not
    reinforce it. The GROUNDED_ANSWER_SCHEMA import loop is the check that
    the right schema, not just some schema, reached the call."""
    from rag.generate import GROUNDED_ANSWER_SCHEMA
    llm = FakeStructuredLLM({"refused": False, "claims": []})
    answer_structured("q", FakeSearcher([_chunk(1, "alpha")]), llm)
    assert llm.last_schema == GROUNDED_ANSWER_SCHEMA
    assert "square brackets" not in llm.last_user


def _fake_response(status_code: int, payload: dict, headers: dict | None = None):
    class _Resp:
        def __init__(self):
            self.status_code = status_code
            self.headers = headers or {}
            self.text = str(payload)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "error", request=None, response=self)

        def json(self):
            return payload

    return _Resp()


@pytest.mark.parametrize("cls,env_key,url_attr", [
    (GroqLLM, "GROQ_API_KEY", "GROQ_URL"),
    (MercuryLLM, "INCEPTION_API_KEY", "MERCURY_URL"),
    (TokenRouterLLM, "TOKEN_ROUTER_API_KEY", "TOKEN_ROUTER_URL"),
])
def test_chat_returns_tool_calls_for_every_provider(
        monkeypatch, cls, env_key, url_attr):
    """The live bug this guards: the agent's tool-calling loop calls
    llm.chat(...) on whatever get_llm() returns, and get_llm() defaults to
    GroqLLM. Until now only IFMLLM implemented chat(), so any deployment
    that never set LLM_PROVIDER=ifm hit an AttributeError on the first
    agent request, with no test catching it. This exercises chat() against
    a mocked transport for every provider that implements the LLM Protocol,
    not just the one the agent happened to be written against.
    """
    monkeypatch.setenv(env_key, "test-key")
    tool_call_message = {
        "role": "assistant",
        "tool_calls": [{
            "id": "call_1",
            "function": {"name": "search_index",
                         "arguments": '{"query": "q", "scratchpad": "s"}'},
        }],
    }

    def fake_post(url, headers=None, json=None, timeout=None):
        assert json["tools"], "tools must be forwarded to the provider"
        return _fake_response(
            200, {"choices": [{"message": tool_call_message,
                                "finish_reason": "tool_calls"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = cls()
    message = llm.chat(
        [{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "search_index"}}],
    )
    assert message["tool_calls"][0]["function"]["name"] == "search_index"
