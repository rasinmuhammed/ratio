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


# ---------------------------------------------------------------------------
# Step 2: support checking
# ---------------------------------------------------------------------------

from rag.attribute import (  # noqa: E402
    MISSING,
    SUPPORTED,
    UNCLEAR,
    UNSUPPORTED,
    check_claims,
    summarise,
)


class FakeJudge:
    """Decides on a keyword in the passage, so the fake controls the verdict
    rather than leaving it to chance. Also counts calls, because not calling
    the judge is part of the contract in two cases below."""

    def __init__(self, verdicts=None, raises=0):
        self.verdicts = verdicts or {}
        self.raises = raises
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.raises:
            self.raises -= 1
            raise RuntimeError("rate limited")
        for key, verdict in self.verdicts.items():
            if key in user:
                return verdict
        return UNSUPPORTED


class Src:
    def __init__(self, text):
        self.text = text


def test_each_claim_is_judged_against_every_source_it_cites():
    """One claim citing two sources makes two assertions about what those
    sources say. Collapsing them to one verdict hides one of them."""
    claims = split_claims("Delay defeats equity [1, 2].")
    checks = check_claims(claims, [Src("alpha"), Src("beta")], FakeJudge())
    assert [c.source for c in checks] == [1, 2]


def test_uncited_claims_are_never_judged():
    """There is nothing to judge them against, and inventing an unsupported
    verdict would conflate 'cited badly' with 'never cited at all'."""
    judge = FakeJudge()
    claims = split_claims("This depends entirely on the facts of the case.")
    assert check_claims(claims, [Src("alpha")], judge) == []
    assert judge.calls == 0


def test_a_citation_to_a_nonexistent_source_is_not_sent_to_the_judge():
    judge = FakeJudge()
    claims = split_claims("The suit was barred by limitation [9].")
    checks = check_claims(claims, [Src("alpha")], judge)
    assert [c.verdict for c in checks] == [MISSING]
    assert judge.calls == 0


def test_an_unparseable_reply_counts_as_unclear_not_supported():
    """A judge that will not answer is not evidence of support. Defaulting the
    other way would inflate every score in the harness."""
    claims = split_claims("The suit was barred by limitation [1].")
    checks = check_claims(claims, [Src("alpha")], FakeJudge({"alpha": "maybe?"}))
    assert checks[0].verdict == UNCLEAR


def test_rate_limits_are_retried_rather_than_swallowed():
    """Giving up on a 429 would silently shrink the sample, and a shrunken
    sample of the easy cases reads better than the truth."""
    claims = split_claims("The suit was barred by limitation [1].")
    judge = FakeJudge({"alpha": SUPPORTED}, raises=2)
    checks = check_claims(claims, [Src("alpha")], judge, sleep=lambda _: None)
    assert checks[0].verdict == SUPPORTED
    assert judge.calls == 3


def test_summarise_reports_citation_and_claim_level_separately():
    """They answer different questions. Citation level measures how honestly
    the model cites; claim level measures how much of the answer holds up, and
    is more forgiving because one good source rescues a claim."""
    claims = split_claims(
        "The suit was barred [1]. Costs followed the event [2]. "
        "This depends on the facts of each case."
    )
    judge = FakeJudge({"limitation": SUPPORTED})
    checks = check_claims(
        claims, [Src("barred by limitation"), Src("silent on costs")], judge
    )
    summary = summarise(claims, checks)
    assert summary["claims"] == 3
    assert summary["uncited_claims"] == 1
    assert summary["citation_precision"] == 0.5
    assert summary["claim_support"] == 0.5


def test_a_claim_survives_on_one_good_source_out_of_two():
    """The reason claim level and citation level diverge."""
    claims = split_claims("Delay defeats equity in these circumstances [1, 2].")
    judge = FakeJudge({"good": SUPPORTED})
    checks = check_claims(claims, [Src("good passage"), Src("bad passage")], judge)
    summary = summarise(claims, checks)
    assert summary["citation_precision"] == 0.5
    assert summary["claim_support"] == 1.0


def test_missing_sources_are_excluded_from_precision_but_still_counted():
    """A citation to a source that was never given is a real failure, but it
    is not a judgement about a passage, so averaging it into precision would
    mix two different things."""
    claims = split_claims("The suit was barred by limitation [1, 9].")
    judge = FakeJudge({"barred": SUPPORTED})
    checks = check_claims(claims, [Src("barred by limitation")], judge)
    summary = summarise(claims, checks)
    assert summary["missing_sources"] == 1
    assert summary["citation_precision"] == 1.0


def test_an_empty_answer_does_not_divide_by_zero():
    assert summarise([], [])["citation_precision"] == 0.0
