from rag.attribute import MIN_CLAIM_CHARS, Claim, split_claims


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


def test_short_spans_do_not_become_claims():
    assert all(len(c.text) >= MIN_CLAIM_CHARS for c in split_claims("Yes [1]."))
