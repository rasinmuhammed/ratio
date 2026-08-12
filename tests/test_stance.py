import pytest

from rag.stance import (
    ARGUMENT,
    BOTH,
    HOLDING,
    NEITHER,
    classify,
    describe,
    priority,
)

ARGUMENTS = [
    "Learned counsel for the petitioner submitted that the suit is barred.",
    "It was contended that no notice under Section 80 was served.",
    "On behalf of the respondent it was urged that the delay is fatal.",
    "The appellant contends that the finding is perverse.",
    "Learned senior counsel argued that possession was never delivered.",
]

HOLDINGS = [
    "We are of the opinion that the suit is barred by limitation.",
    "In our considered view the finding cannot be sustained.",
    "It is well settled that a writ does not lie against a private body.",
    "This Court holds that the notification is invalid.",
    "We are unable to accept the contention regarding delay.",
]

NARRATIVE = [
    "The appeal was filed on 3 March 2011 before this Court.",
    "The plaintiff purchased the land in 1974 under a registered sale deed.",
    "Section 9 of the Act provides for the framing of a scheme.",
]


@pytest.mark.parametrize("text", ARGUMENTS)
def test_reported_positions_are_argument(text):
    assert classify(text) == ARGUMENT


@pytest.mark.parametrize("text", HOLDINGS)
def test_the_courts_own_voice_is_holding(text):
    assert classify(text) == HOLDING


@pytest.mark.parametrize("text", NARRATIVE)
def test_narrative_is_not_guessed_at(text):
    """Roughly 70% of the corpus lands here and that is the honest answer.
    Recitals of fact and statutory quotation carry no stance, and inventing
    one would be worse than declining."""
    assert classify(text) == NEITHER


def test_a_submission_and_its_rejection_is_both():
    """The most useful passage in the corpus, and it must not be demoted for
    containing the word 'submitted'."""
    text = ("It was submitted that limitation does not apply. "
            "We are unable to accept that contention.")
    assert classify(text) == BOTH


def test_both_outranks_holding_which_outranks_argument():
    assert priority("It was submitted that limitation applies. We hold it does.") < \
           priority("We hold that limitation applies.")
    assert priority("We hold that limitation applies.") < \
           priority("It was submitted that limitation applies.")


def test_argument_ranks_last():
    """The whole point. A passage recording only what a party wanted is the
    weakest evidence of what the law is."""
    texts = ARGUMENTS[:1] + HOLDINGS[:1] + NARRATIVE[:1]
    assert max(texts, key=priority) == ARGUMENTS[0]


def test_classification_is_case_insensitive():
    assert classify("LEARNED COUNSEL SUBMITTED THAT THE CLAIM FAILS") == ARGUMENT
    assert classify("WE ARE OF THE OPINION THAT THE CLAIM FAILS") == HOLDING


def test_describe_returns_prose_for_the_prompt():
    """The model has to act on this, so it is a phrase and not a score."""
    assert describe(ARGUMENTS[0]) == "records a party's submission, not the court's finding"
    assert describe(HOLDINGS[0]) == "the court's own reasoning"


def test_empty_text_does_not_raise():
    assert classify("") == NEITHER


def test_substrings_do_not_trigger_a_match():
    """'submitted' inside another word, and 'we find' as part of an ordinary
    sentence about locating something, should not read as a holding."""
    assert classify("The resubmitted application was rejected.") == NEITHER
