from rag.contextualize import (
    MAX_CHARS,
    SummaryCache,
    summarize_document,
    summarize_with_cache,
)
from rag.ingest import Document


class FakeLLM:
    def __init__(self, reply: str = "summary", raises: int = 0):
        self.reply = reply
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.raises > 0:
            self.raises -= 1
            raise RuntimeError("rate limited")
        return self.reply


def _doc(text: str, doc_id: str = "d1") -> Document:
    return Document(id=doc_id, text=text, metadata={})


def test_summarize_document_truncates_to_max_chars():
    llm = FakeLLM()
    summarize_document(_doc("x" * (MAX_CHARS * 2)), llm)
    (_, user), = llm.calls
    assert len(user) == MAX_CHARS


def test_summarize_document_strips_whitespace():
    llm = FakeLLM(reply="  a summary  \n")
    assert summarize_document(_doc("text"), llm) == "a summary"


def test_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "summaries.json"
    cache = SummaryCache(path)
    cache.set("d1", "the summary")

    reloaded = SummaryCache(path)
    assert reloaded.get("d1") == "the summary"


def test_cache_miss_returns_none(tmp_path):
    cache = SummaryCache(tmp_path / "summaries.json")
    assert cache.get("missing") is None


def test_summarize_with_cache_skips_the_llm_on_a_hit(tmp_path):
    cache = SummaryCache(tmp_path / "summaries.json")
    cache.set("d1", "cached summary")
    llm = FakeLLM(reply="should not be used")

    result = summarize_with_cache(_doc("text"), llm, cache)

    assert result == "cached summary"
    assert llm.calls == []


def test_summarize_with_cache_writes_a_miss_back(tmp_path):
    cache = SummaryCache(tmp_path / "summaries.json")
    llm = FakeLLM(reply="fresh summary")

    result = summarize_with_cache(_doc("text", "d2"), llm, cache)

    assert result == "fresh summary"
    assert cache.get("d2") == "fresh summary"


def test_summarize_with_cache_retries_before_giving_up(tmp_path):
    cache = SummaryCache(tmp_path / "summaries.json")
    llm = FakeLLM(reply="eventually", raises=1)

    result = summarize_with_cache(_doc("text", "d3"), llm, cache,
                                   sleep=lambda _: None, retries=2)

    assert result == "eventually"


def test_summarize_with_cache_returns_empty_rather_than_raise(tmp_path):
    """A document that never summarises should not crash a run 12,000
    documents long. context_prefix() alone still runs without this."""
    cache = SummaryCache(tmp_path / "summaries.json")
    llm = FakeLLM(raises=5)

    result = summarize_with_cache(_doc("text", "d4"), llm, cache,
                                   sleep=lambda _: None, retries=2)

    assert result == ""
    assert cache.get("d4") is None  # a failure is not cached as a real summary
