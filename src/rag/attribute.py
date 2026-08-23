"""Is the answer actually supported by the sources it cites?

Retrieval is measured to death in this repo and generation is not measured at
all. That is the wrong way round: retrieval failures are recoverable, because a
missing chunk produces a vague answer or a refusal, but an attribution failure
produces a confident sentence with a citation attached to a source that does
not say it. A reader who checks the citation is reassured by the number and
never reads the passage.

The question is narrower than relevance and, unlike relevance, it is decidable
from the material already on hand. "Does chunk 7 answer this query" needs a
human with legal judgment. "Does chunk 7 say what this sentence claims it
says" needs only the sentence and the chunk, both of which are in the Answer.

Four steps. This module is the first:

1. split the answer into claims, each with the sources it cites
2. check each claim against the text of the sources it cites
3. report citation precision, unsupported rate, uncited rate
4. validate the checker by hand on a sample, the way stance.py was

Nothing here calls a model. Segmentation is a pure function over the answer
text so it can be tested exhaustively and cheaply, and so a mistake in it shows
up as a failing test rather than as a wrong percentage at the end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

# A run of one or more bracketed citations. Matching a run rather than a single
# bracket means "[1][2]" and "[1] [2]" attach to the same preceding text, which
# is what they mean, instead of producing a claim whose entire body is a space.
_CITATION = re.compile(r"(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*)+")
_NUMBER = re.compile(r"\d+")

# Below this, a span is connective tissue rather than a proposition: ", and",
# "However", "such as". Checking whether a source supports ", and" is not a
# question, so those spans are merged instead of judged.
MIN_CLAIM_CHARS = 15


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion from an answer, with the sources offered for it.

    `cited` is empty for a claim the model stated without any citation, and
    those are kept rather than discarded. An uncited claim is the most
    dangerous kind: it has no support by construction, and dropping it would
    quietly raise every score computed afterwards.
    """

    text: str
    cited: list[int]
    start: int          # offset into the answer, so a claim can be located again

    @property
    def uncited(self) -> bool:
        return not self.cited


def _dedup(numbers: list[int]) -> list[int]:
    """Order-preserving. Repeated citations of one source are one claim about
    that source, but the order carries the model's own emphasis."""
    seen: list[int] = []
    for n in numbers:
        if n not in seen:
            seen.append(n)
    return seen


def split_claims(text: str, min_chars: int = MIN_CLAIM_CHARS) -> list[Claim]:
    """Split an answer at its citations.

    Sentence splitting would be the obvious approach and it is wrong here. A
    single sentence routinely carries two citations for two different halves,
    as in "X [1], and Y [2]", and checking that whole sentence against source
    [1] fails on account of Y, which [1] never claimed to support. Splitting at
    the citations attaches each span to the sources actually offered for it.

    The cost is fragments. ", and the court has consistently held that ..." is
    not a well-formed sentence, but it is a checkable proposition, which is the
    only property that matters downstream.

    A refused answer should not be passed here. It has no claims, and the
    caller knows about refusal while this module deliberately does not.

    >>> [(c.text, c.cited) for c in split_claims("The suit was barred [1]. Costs followed [2].")]
    [('The suit was barred', [1]), ('. Costs followed', [2])]
    >>> split_claims("Delay defeats equity [1, 3].")[0].cited
    [1, 3]
    >>> split_claims("It depends on the circumstances of the case.")[0].uncited
    True
    """
    claims: list[Claim] = []
    pending: list[int] = []
    pos = 0

    for run in _CITATION.finditer(text):
        start, raw = pos, text[pos:run.start()]
        pos = run.end()
        span = " ".join(raw.split())
        numbers = [int(n) for n in _NUMBER.findall(run.group(0))]

        if len(span) >= min_chars:
            claims.append(Claim(span, _dedup(pending + numbers), start))
            pending = []
        elif claims:
            # A fragment between two citations belongs to the claim before it.
            claims[-1] = replace(
                claims[-1], cited=_dedup(claims[-1].cited + numbers)
            )
        else:
            # Nothing to attach to yet, as in "According to [1], the court
            # held X". Carry the citation forward to the first real claim
            # rather than losing it.
            pending.extend(numbers)

    tail = " ".join(text[pos:].split())
    if len(tail) >= min_chars:
        claims.append(Claim(tail, _dedup(pending), pos))
    elif pending and claims:
        claims[-1] = replace(claims[-1], cited=_dedup(claims[-1].cited + pending))

    return claims
