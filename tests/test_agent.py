import pytest

from rag.agent import LegalAgent
from rag.generate import parse_answer
from rag.retrieve import Result


def _chunk(chunk_id: str, text: str) -> Result:
    return Result(rank=1, score=0.8, chunk_id=chunk_id, doc_id=chunk_id,
                  text=text, metadata={"court": "Supreme Court",
                                        "title": "Test v. Test",
                                        "cited_by": 5},
                  score_type="cosine")


class FakeRetriever:
    """Returns a different, disjoint result set on each call, the way real
    retrieval does for two different sub-queries."""

    def __init__(self, batches: list[list[Result]]) -> None:
        self.batches = batches
        self.calls = 0

    def search(self, query: str, k: int = 6) -> list[Result]:
        batch = self.batches[self.calls]
        self.calls += 1
        return batch


class ScriptedChatLLM:
    """Satisfies the chat() half of the LLM Protocol with a fixed sequence
    of tool calls, so the agent loop can be driven without a network call."""

    def __init__(self, turns: list[dict]) -> None:
        self.turns = turns
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError("agent only uses chat()")

    def chat(self, messages: list[dict], tools=None) -> dict:
        msg = self.turns[self.calls]
        self.calls += 1
        return msg


def _tool_call(call_id: str, name: str, **arguments) -> dict:
    import json
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def test_citation_numbers_stay_stable_across_multiple_searches():
    """The bug this guards: format_search_results used to number every
    search_index call's results starting at [1], so a chunk retrieved by the
    second search and a chunk retrieved by the first could both be labelled
    [1] in what the model reads, and a citation in the final answer could
    not be mapped back to a real source once more than one search had run.
    """
    batch_one = [_chunk("doc-a#1", "first batch holding text")]
    batch_two = [_chunk("doc-b#1", "second batch holding text")]
    retriever = FakeRetriever([batch_one, batch_two])

    llm = ScriptedChatLLM([
        _tool_call("c1", "search_index", scratchpad="", query="first sub-question"),
        _tool_call("c2", "search_index", scratchpad="found doc-a", query="second sub-question"),
        _tool_call("c3", "submit_final_answer", scratchpad="done",
                   final_answer="The rule is X [1]. A separate point holds Y [2]."),
    ])

    agent = LegalAgent(llm, retriever, k=6)
    events = list(agent.stream("a complex question"))

    final = [e for e in events if e["type"] == "final_answer"][0]["content"]

    # Two searches happened, so all_sources has both chunks, in call order.
    assert list(agent.all_sources.keys()) == ["doc-a#1", "doc-b#1"]

    # [1] in the final answer must resolve to the FIRST search's chunk and
    # [2] to the SECOND search's chunk, not both collapsing to index 1
    # because each search's own formatted context restarted numbering.
    refused, cited, invalid = parse_answer(final, len(agent.all_sources))
    assert not refused
    assert cited == [1, 2]
    assert invalid == []


def test_format_search_results_numbers_by_global_position_not_call_order():
    llm = ScriptedChatLLM([])
    agent = LegalAgent(llm, retriever=FakeRetriever([]), k=6)
    first = _chunk("doc-a#1", "alpha")
    second = _chunk("doc-b#1", "beta")

    # Simulate what stream() does: insert into all_sources before formatting.
    agent.all_sources["doc-a#1"] = first
    agent.all_sources["doc-b#1"] = second

    formatted = agent.format_search_results([second])
    assert "Source [2]" in formatted
    assert "Source [1]" not in formatted
