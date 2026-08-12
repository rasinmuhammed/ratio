"""Whose voice is this passage in?

A judgment is not uniform text. It recites facts, records what each side
submitted, and states what the court decided, and the first two routinely
contradict the third. "Learned counsel submitted that limitation does not
apply" is a proposition that may be rejected two paragraphs later.

Nothing in cosine similarity or BM25 can see that. Both score a passage on
whether it discusses limitation, not on whether the court agreed. Handed to a
model as an undifferentiated source, an argument passage produces a fluent
statement of the opposite of the law with a correct citation attached, which is
the worst failure available here: wrong and credible.

How often that happens is smaller than it first looked. Measured on the full
index the corpus is 13.3% argument against 12.9% holding, close to even, and
retrieval already prefers holdings by two to three times: dense returns 5.3%
argument against 16.0% holding, BM25 10.7% against 20.7%. So this is a hazard
of roughly one source in ten to twenty rather than a systemic bias.

An earlier version of these patterns reported 21% argument against 5.8%
holding, and that gap was an artifact: "it was held" did not match, which is
the most common way an Indian judgment states law. The number said the corpus
was argument-heavy by 3.6 to 1 when it is closer to even.

This does not filter anything. A contention is often exactly what a user asked
about, and a passage carrying an argument together with its rejection is the
most useful text in the corpus. The point is to stop presenting every source as
if it carried equal authority.

The classifier is deliberately shallow. It matches the reporting formulas that
Indian judgments use consistently, and it declines to guess anywhere else:
roughly 70% of chunks come back `neither`, which is the honest answer for
recitals of fact, statutory quotation and procedural history. A model could do
better, but not 414,122 times, and the label has to exist before retrieval
ranks anything rather than after.
"""

from __future__ import annotations

import re

ARGUMENT = "argument"
HOLDING = "holding"
BOTH = "both"
NEITHER = "neither"

# What a party said. A reporting verb is required in every branch: the earlier
# version accepted a bare "learned counsel", which fires on the appearance
# block at the head of a judgment ("Shri S.D. Shukla, learned counsel for the
# applicants") where nobody has argued anything yet. Validation against a
# language model put that row at 32% agreement, and the appearance lines were
# a large part of why.
_SPEAKER = r"(?:learned\s+(?:senior\s+|standing\s+|additional\s+)?(?:counsel|advocate|" \
           r"government\s+pleader)|the\s+(?:petitioner|respondent|appellant|applicant)s?)"
_ARGUES = r"(?:submits?|submitted|contends?|contended|urges?|urged|argues?|argued|" \
          r"cont(?:e|a)nds?|pleaded)"

_ARGUMENT = re.compile(
    r"\b(?:"
    # A speaker and a reporting verb, with room for an intervening phrase such
    # as "for the petitioner" or "appearing on behalf of the State".
    rf"{_SPEAKER}(?:\W+\w+){{0,8}}?\W+{_ARGUES}"
    r"|it\s+(?:was|is|has\s+been)\s+(?:submitted|contended|urged|argued)"
    # "submitted by" is deliberately absent. In "returns submitted by the
    # dealers" the verb means filed, not argued, and it was a false positive.
    r"|(?:contended|urged|argued|submitted)\s+that"
    r"|on\s+behalf\s+of\s+the\s+(?:petitioner|respondent|appellant|applicant)s?"
    rf"(?:\W+\w+){{0,8}}?\W+{_ARGUES}"
    r")",
    re.I,
)

# What the court decided, including law it adopts from earlier authority.
#
# The costly omission in the first version was "it was held". Indian judgments
# state law that way constantly, far more often than "it is held", so the
# pattern missed the most common holding formula in the corpus and made the
# whole corpus look argument-heavy. Held-in-the-past is still a statement of
# law, not a submission, whether the court is holding or quoting a holding.
_HOLDING = re.compile(
    r"\b(?:"
    r"we\s+(?:are\s+of\s+the\s+(?:opinion|view)|hold|held|find|"
    r"are\s+satisfied|conclude|are\s+unable\s+to\s+accept)"
    r"|in\s+our\s+(?:considered\s+)?(?:opinion|view|judgment)"
    r"|it\s+(?:is|was|has\s+been)\s+(?:held|well\s+settled|settled)"
    r"|(?:court|bench|judge)\s+(?:has\s+)?held\s+that"
    r"|(?:this|the)\s+court\s+(?:holds|is\s+of\s+the\s+(?:opinion|view))"
    r"|the\s+law\s+is\s+(?:well\s+)?settled"
    r"|(?:having\s+)?considered\s+the\s+(?:rival\s+)?(?:submissions|contentions)"
    r")",
    re.I,
)

# Shown to the model beside each source. Written as plain description rather
# than as a score, because the model has to act on it and "0.3" would mean
# nothing to it.
DESCRIPTIONS = {
    ARGUMENT: "records a party's submission, not the court's finding",
    HOLDING: "the court's own reasoning",
    BOTH: "contains both a party's submission and the court's own reasoning",
    NEITHER: "narrative or procedural, no stated position",
}

# Ordering for reranking. A passage in the court's voice outranks one that only
# reports an argument. `both` sits at the top because a contention shown
# alongside its disposition is the most informative passage available.
PRIORITY = {BOTH: 0, HOLDING: 1, NEITHER: 2, ARGUMENT: 3}


def classify(text: str) -> str:
    """Which voices appear in this passage.

    Coarse on purpose. A 450 token chunk can hold both voices, and how often it
    does is part of the finding: a chunk carrying an argument and its rejection
    is safe, one carrying only the argument is not.

    >>> classify("Learned counsel for the petitioner submitted that the suit is barred.")
    'argument'
    >>> classify("We are of the opinion that the suit is barred by limitation.")
    'holding'
    >>> classify("The appeal was filed on 3 March 2011 before this Court.")
    'neither'
    """
    argument = bool(_ARGUMENT.search(text))
    holding = bool(_HOLDING.search(text))
    if argument and holding:
        return BOTH
    if argument:
        return ARGUMENT
    if holding:
        return HOLDING
    return NEITHER


def describe(text: str) -> str:
    """The stance label as a phrase to put in front of a language model.

    >>> describe("It was submitted that no notice was served.")
    "records a party's submission, not the court's finding"
    """
    return DESCRIPTIONS[classify(text)]


def priority(text: str) -> int:
    """Sort key. Lower is more authoritative.

    >>> priority("We hold that the appeal fails.") < priority("It was urged that the appeal fails.")
    True
    """
    return PRIORITY[classify(text)]
