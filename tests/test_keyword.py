import pytest

from rag.hybrid import reciprocal_rank_fusion
from rag.keyword import BM25Index, tokenize


def test_tokenize_keeps_digits():
    """Digits are the whole point of this module. Dense retrieval scored
    0.52 on 'AIR 1974' while 6 chunks contained the literal string."""
    assert tokenize("[AIR 1974 Patna 164].") == ["air", "1974", "patna", "164"]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("No.1091 of 2013") == ["no", "1091", "of", "2013"]


def test_rare_terms_outrank_common_ones():
    """idf in action: 'court' appears everywhere and carries no signal,
    'zebra' appears once and identifies the document."""
    texts = [
        "the court held",
        "the court dismissed",
        "the court allowed",
        "the court and the zebra",
    ]
    bm = BM25Index(texts, [f"d{i}" for i in range(4)])
    assert bm.search("zebra court", k=1)[0].index == 3


def test_term_frequency_saturates():
    """Eight occurrences must not score eight times one occurrence."""
    bm = BM25Index(["alpha " * 1 + "pad " * 50, "alpha " * 8 + "pad " * 43],
                   ["once", "eight"])
    once, eight = bm.search("alpha", k=2)[::-1]
    assert eight.score > once.score
    assert eight.score < once.score * 8


def test_longer_documents_are_penalised():
    """Same single occurrence, different lengths. The shorter document is
    more 'about' the term, so b>0 should rank it higher."""
    bm = BM25Index(["alpha beta", "alpha " + "filler " * 200], ["short", "long"])
    assert bm.search("alpha", k=2)[0].index == 0


def test_missing_term_returns_nothing():
    bm = BM25Index(["alpha beta"], ["d0"])
    assert bm.search("gamma") == []


def test_empty_query_returns_nothing():
    bm = BM25Index(["alpha beta"], ["d0"])
    assert bm.search("") == []
    assert bm.search("   ") == []


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="mismatch"):
        BM25Index(["a", "b"], ["only-one-id"])


def test_match_is_named():
    """Returning a NamedTuple rather than a bare tuple so callers can write
    match.index instead of match[0]."""
    bm = BM25Index(["alpha"], ["d0"])
    m = bm.search("alpha")[0]
    assert m.index == 0 and m.score > 0


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_rrf_rewards_agreement():
    """A document both systems rank moderately beats one only a single
    system loves. That is the entire point of fusing on rank."""
    fused = dict(reciprocal_rank_fusion([["x", "y", "z"], ["y", "z", "x"]]))
    assert fused["y"] > fused["x"]      # y: ranks 2 and 1. x: ranks 1 and 3.


def test_rrf_ignores_score_magnitude():
    """Only position matters. Dense scores sit in 0.5-0.8, BM25 scores are
    unbounded; fusing on rank sidesteps the incomparability entirely."""
    assert reciprocal_rank_fusion([["a"]])[0][1] == pytest.approx(1 / 61)


def test_rrf_handles_disjoint_rankings():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["c", "d"]]))
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused["a"] == fused["c"]     # both ranked first in their own list


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
