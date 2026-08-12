"""Whose voice is this passage in?

A judgment is not uniform text. It recites facts, records what each side
submitted, and states what the court decided, and the first two routinely
contradict the third. "Learned counsel submitted that limitation does not
apply" is a proposition that may be rejected two paragraphs later.

Nothing in cosine similarity or BM25 can see that. Both score a passage on
whether it discusses limitation, not on whether the court agreed. Measured on
the full index, 10% to 17% of retrieved chunks record a party's position with
no sign of the court's own view, and 21% of the corpus does. Handed to a model
as an undifferentiated source, such a chunk produces a fluent statement of the
opposite of the law with a correct citation attached, which is the worst
failure available here: wrong and credible.

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

# What a party said. These are reporting verbs: the judgment is describing a
# position rather than adopting one.
_ARGUMENT = re.compile(
    r"\b(?:"
    r"learned\s+(?:counsel|advocate|senior\s+counsel|standing\s+counsel)"
    r"|it\s+(?:was|is)\s+(?:submitted|contended|urged|argued)"
    r"|(?:submitted|contended|urged|argued)\s+(?:that|by)"
    r"|on\s+behalf\s+of\s+the\s+(?:petitioner|respondent|appellant|applicant)"
    r"|the\s+(?:petitioner|respondent|appellant|applicant)\s+"
    r"(?:submits|contends|urges|argues|submitted|contended)"
    r")\b",
    re.I,
)

# What the court decided. First person plural and decisional verbs, which a
# judgment reserves for its own voice.
_HOLDING = re.compile(
    r"\b(?:"
    r"we\s+(?:are\s+of\s+the\s+(?:opinion|view)|hold|find|are\s+satisfied|conclude)"
    r"|in\s+our\s+(?:considered\s+)?(?:opinion|view|judgment)"
    r"|it\s+is\s+(?:held|well\s+settled)"
    r"|this\s+court\s+(?:holds|is\s+of\s+the\s+view)"
    r"|the\s+law\s+is\s+(?:well\s+)?settled"
    r"|we\s+are\s+unable\s+to\s+accept"
    r")\b",
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
