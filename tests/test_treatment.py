from rag.treatment import (
    DISTINGUISHED,
    FOLLOWED,
    OVERRULED,
    REFERRED,
    classify_treatment,
    find_citations,
)


def test_finds_an_air_citation():
    text = "The Court in AIR 1978 SC 597 held that the right applies."
    cites = find_citations(text)
    assert len(cites) == 1
    assert cites[0].key == "AIR 1978 SC 597"
    assert cites[0].raw == "AIR 1978 SC 597"


def test_finds_an_scc_citation_with_parenthesised_year():
    text = "This follows (2019) 4 SCC 736 on the point of limitation."
    cites = find_citations(text)
    assert len(cites) == 1
    assert cites[0].key == "2019 SCC 736"


def test_finds_an_scc_online_citation():
    text = "See 2020 SCC OnLine SC 123 for the current position."
    cites = find_citations(text)
    assert cites[0].key == "2020 SCC OnLine SC 123"


def test_two_mentions_of_the_same_case_normalize_to_the_same_key():
    """The whole point of keying by citation string: scraping artifacts like
    doubled whitespace, or the court abbreviation's letter case, must not
    split one case into two nodes in the graph. AIR itself is always written
    in caps in practice (it is a reporter name, not prose), so that part of
    the format is not a real source of variation, unlike spacing and the
    court code."""
    a = find_citations("AIR 1978 sc 597 is the leading case.")[0]
    b = find_citations("As held in AIR   1978   SC   597.")[0]
    assert a.key == b.key


def test_citations_are_returned_in_order_of_appearance():
    text = "First AIR 1978 SC 597, then (2019) 4 SCC 736 later."
    cites = find_citations(text)
    assert [c.key for c in cites] == ["AIR 1978 SC 597", "2019 SCC 736"]


def test_no_citation_shaped_text_returns_nothing():
    assert find_citations("The petition is dismissed with costs.") == []


def test_overruled_signal_within_the_window_is_detected():
    text = "AIR 1978 SC 597 has been overruled by a later Constitution Bench."
    cite = find_citations(text)[0]
    assert classify_treatment(text, cite) == OVERRULED


def test_distinguished_signal_within_the_window_is_detected():
    text = "Learned counsel relied on AIR 1978 SC 597, but it is distinguishable on facts."
    cite = find_citations(text)[0]
    assert classify_treatment(text, cite) == DISTINGUISHED


def test_followed_signal_within_the_window_is_detected():
    text = "The ratio in AIR 1978 SC 597 was followed by this Court."
    cite = find_citations(text)[0]
    assert classify_treatment(text, cite) == FOLLOWED


def test_bare_citation_with_no_signal_word_is_referred():
    """The honest default: most citations sit in a list of authorities with
    no individual discussion, and guessing a treatment for those would be
    worse than saying plainly that none was stated."""
    text = "See also AIR 1978 SC 597, AIR 1982 SC 12 and AIR 1990 SC 88."
    cite = find_citations(text)[0]
    assert classify_treatment(text, cite) == REFERRED


def test_a_signal_word_far_outside_the_window_does_not_attach():
    """'Distinguished' three paragraphs later, about a different citation, is
    not a treatment of this one; only what is actually near the mention
    counts."""
    from rag.treatment import WINDOW_CHARS
    far_away = "distinguished on facts"
    padding = "x" * (WINDOW_CHARS + 100)
    text = f"AIR 1978 SC 597 is cited here. {padding} {far_away}"
    cite = find_citations(text)[0]
    assert classify_treatment(text, cite) == REFERRED


def test_overruled_wins_when_multiple_signals_are_present():
    """Overruled is the strongest, most specific claim; a passage that
    mentions both should not be reported as merely distinguished."""
    text = "AIR 1978 SC 597 was distinguished below, but is now overruled outright."
    cite = find_citations(text)[0]
    assert classify_treatment(text, cite) == OVERRULED
