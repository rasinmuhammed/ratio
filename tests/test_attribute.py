from rag.attribute import MIN_CLAIM_CHARS, split_claims


def cited(text):
    return [(c.text, c.cited) for c in split_claims(text)]


def test_each_span_takes_the_citation_that_follows_it():
    assert cited("The suit was barred [1]. Costs were awarded [2].") == [
        ("The suit was barred", [1]),
        (". Costs were awarded", [2]),
    ]


def test_a_sentence_with_two_citations_becomes_two_claims():
    """The reason this splits on citations rather than sentences. Checking the
    whole sentence against [1] would fail on account of the half that [1] was
    never offered for."""
    claims = split_claims("Limitation applies [1], and costs follow the event [2].")
    assert len(claims) == 2
    assert claims[0].cited == [1]
    assert claims[1].cited == [2]


def test_grouped_citations_stay_together():
    assert split_claims("Delay defeats equity [1, 3].")[0].cited == [1, 3]
    assert split_claims("Delay defeats equity [1,3].")[0].cited == [1, 3]


def test_adjacent_citation_runs_attach_to_the_same_claim():
    """'[1][2]' means both sources support the sentence, not that there is a
    second claim whose entire body is empty."""
    assert split_claims("Delay defeats equity [1][2].")[0].cited == [1, 2]
    assert split_claims("Delay defeats equity [1] [2].")[0].cited == [1, 2]


def test_a_connective_fragment_merges_into_the_previous_claim():
    """', and' is not a proposition, and asking whether a source supports it
    is not a question. Its citation belongs to the claim before it."""
    claims = split_claims("The appeal was dismissed for delay [1], and [2].")
    assert len(claims) == 1
    assert claims[0].cited == [1, 2]


def test_a_leading_citation_carries_forward():
    """'According to [1], ...' has nothing before the citation to attach it
    to. Dropping it would understate how much of the answer was cited."""
    claims = split_claims("According to [1], the suit was barred by limitation.")
    assert len(claims) == 1
    assert claims[0].cited == [1]


def test_uncited_trailing_text_is_kept_as_a_claim():
    """The most dangerous kind. Discarding it would raise every score computed
    afterwards, by removing exactly the claims that have no support."""
    claims = split_claims("The suit was barred [1]. This depends on the facts.")
    assert claims[-1].uncited
    assert claims[-1].text == ". This depends on the facts."


def test_an_answer_with_no_citations_is_one_uncited_claim():
    claims = split_claims("The position is unsettled and varies by High Court.")
    assert len(claims) == 1
    assert claims[0].uncited


def test_repeated_citations_are_deduplicated_in_order():
    assert split_claims("Delay defeats equity [2, 1, 2].")[0].cited == [2, 1]


def test_whitespace_and_newlines_are_normalised():
    claims = split_claims("The suit\n   was\n\nbarred [1].")
    assert claims[0].text == "The suit was barred"


def test_start_offsets_locate_the_claim_in_the_answer():
    text = "The suit was barred [1]. Costs were awarded [2]."
    for claim in split_claims(text):
        # The claim text is whitespace-normalised, so the comparison has to be
        # too. Comparing against the raw slice fails on the newlines the model
        # actually produces.
        assert " ".join(text[claim.start:].split()).startswith(claim.text)


def test_empty_answer_yields_no_claims():
    assert split_claims("") == []
    assert split_claims("   \n  ") == []


def test_short_spans_between_citations_do_not_become_claims():
    """Mid-answer fragments merge rather than standing alone. The separate
    case of an answer that is *entirely* short is covered below, where
    returning nothing would be worse."""
    claims = split_claims("The suit was barred by limitation [1], so [2].")
    assert len(claims) == 1
    assert len(claims[0].text) >= MIN_CLAIM_CHARS


def test_law_report_brackets_are_not_source_citations():
    """Indian citations are written "[1998] 2 SCC 341" and the model quotes
    judgment language. An unbounded \\d+ read 1998 as a source number, ate the
    words before it, and reported a hallucinated citation that never happened,
    in the metric whose job is to detect hallucinated citations."""
    claims = split_claims(
        "The court in [1958] SCR 1226 held that delay defeats equity [2]."
    )
    assert len(claims) == 1
    assert claims[0].cited == [2]
    assert claims[0].text.startswith("The court in [1958] SCR 1226")


def test_a_plausible_hallucinated_source_number_is_still_captured():
    """The three-digit bound must not hide the thing it sits next to. A model
    inventing a source cites something near the range it was given."""
    assert split_claims("The suit was barred by limitation [9].")[0].cited == [9]


def test_a_short_answer_still_produces_a_claim():
    """Every span under the threshold used to return no claims at all, so a
    short answer dropped out of the aggregate rather than counting against
    it."""
    claims = split_claims("Yes [1].")
    assert len(claims) == 1
    assert claims[0].cited == [1]


def test_an_answer_that_is_only_a_citation_has_no_claim():
    """The counterpart. There is genuinely no proposition here, so inventing
    one would be worse than returning nothing."""
    assert split_claims("[1][2]") == []
