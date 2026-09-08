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
import time
from dataclasses import dataclass, replace

# A run of one or more bracketed citations. Matching a run rather than a single
# bracket means "[1][2]" and "[1] [2]" attach to the same preceding text, which
# is what they mean, instead of producing a claim whose entire body is a space.
#
# Three digits maximum, and that bound is load-bearing. Indian law reports are
# written "[1998] 2 SCC 341" and the model quotes judgment language, so an
# unbounded \d+ turns a quoted case into a citation of source 1998, eats the
# words before it, and reports a hallucinated citation that never happened.
_CITATION = re.compile(r"(?:\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]\s*)+")
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

    if not claims:
        # Every span fell under the threshold, as in "Yes [1]." Returning
        # nothing would drop the answer out of the aggregate entirely, and a
        # short answer is not an answer without claims, it is a short claim.
        whole = " ".join(_CITATION.sub(" ", text).split())
        if whole:
            claims.append(Claim(whole, _dedup(pending), 0))

    return claims


# ---------------------------------------------------------------------------
# Step 2: does the cited source actually say it
# ---------------------------------------------------------------------------

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
UNCLEAR = "unclear"
MISSING = "missing_source"

VERDICTS = (SUPPORTED, UNSUPPORTED, UNCLEAR)

# A source chunk is around 450 tokens. Capped anyway, because a claim that
# cannot be judged from 2,500 characters will not become judgeable at 5,000,
# and the cost is paid once per claim-source pair.
EXCERPT = 2500

JUDGE_SYSTEM = """You check whether a passage from an Indian court judgment \
supports a specific claim.

Answer with exactly one word:

supported    - the passage states the claim, or the claim follows directly from it
unsupported  - the passage does not state this, or states something different
unclear      - the claim is a sentence fragment or too vague to check

Judge only against the passage in front of you. Do not use outside knowledge of \
Indian law. A claim can be perfectly true in general and still unsupported by \
this particular passage, and that is "unsupported".

One word. No punctuation, no explanation."""


@dataclass(frozen=True, slots=True)
class Check:
    """One claim judged against one source it cited.

    The unit is the pair rather than the claim, because a claim citing three
    sources makes three separate assertions about what those sources say, and
    collapsing them would hide two of them.
    """

    claim: Claim
    source: int          # 1-based, as the model wrote it
    verdict: str


def _ask(llm, passage: str, claim: str, attempts: int = 3,
         sleep=time.sleep) -> str:
    """One verdict. Anything the model will not answer cleanly is `unclear`,
    which is honest: an unparseable reply is not evidence of support.

    attempts dropped from 4 to 3 on 2026-09-07. llm.complete() already
    retries internally (now 4 attempts, up to 30s wait, for both 429s and
    transport errors), so this outer loop was multiplying an already-bounded
    retry into an unbounded-feeling one: 4 outer x 6 inner meant one
    consistently unlucky claim could burn 30+ minutes before giving up, and a
    few of those in one query is how a run goes silent for two hours without
    ever actually being stuck. 3 keeps test_rate_limits_are_retried's
    guarantee (survives 2 consecutive failures, succeeds on the 3rd) while
    still cutting the worst case from hours to well under 30 minutes."""
    prompt = f"Passage:\n{passage[:EXCERPT]}\n\nClaim:\n{claim}\n\nVerdict:"
    for attempt in range(attempts):
        try:
            reply = llm.complete(JUDGE_SYSTEM, prompt).strip().lower()
        except RuntimeError:
            # Rate limited. Backing off is the only correct response; giving
            # up would silently shrink the sample and flatter the result.
            sleep(4 * (attempt + 1))
            continue
        word = reply.split()[0].strip(".,:;") if reply.split() else ""
        return word if word in VERDICTS else UNCLEAR
    return UNCLEAR


def check_claims(claims: list[Claim], sources: list, llm,
                 sleep=time.sleep) -> list[Check]:
    """Judge every claim against every source it cited.

    Uncited claims produce no Check at all. That is deliberate: there is
    nothing to judge them against, and inventing an `unsupported` verdict for
    them would conflate two different failures. They are counted separately in
    `summarise`, where the distinction survives.

    `sleep` is injected so the retry path can be tested without a test suite
    that actually waits twelve seconds. A slow suite stops being run.
    """
    checks: list[Check] = []
    for claim in claims:
        for number in claim.cited:
            if not 1 <= number <= len(sources):
                # The model cited a source that was never given to it. No
                # point asking a judge about a passage that does not exist.
                checks.append(Check(claim, number, MISSING))
                continue
            verdict = _ask(llm, sources[number - 1].text, claim.text,
                           sleep=sleep)
            checks.append(Check(claim, number, verdict))
    return checks


def summarise(claims: list[Claim], checks: list[Check]) -> dict[str, float]:
    """Two units, reported side by side, because they answer different
    questions.

    Citation level asks how often the model's citing is honest. Claim level
    asks how much of the answer a reader can trust, which is what a reader
    actually wants and is the more forgiving of the two: a claim citing three
    sources needs only one of them to hold up.
    """
    judged = [c for c in checks if c.verdict != MISSING]
    supported = [c for c in judged if c.verdict == SUPPORTED]

    cited_claims = [c for c in claims if not c.uncited]
    # Keyed by start offset, which is unique within an answer. Claim holds a
    # list so it is unhashable, and id() would silently stop working the moment
    # a caller rebuilt the claims between checking and summarising.
    backed = {c.claim.start for c in checks if c.verdict == SUPPORTED}

    return {
        "claims": len(claims),
        "uncited_claims": sum(1 for c in claims if c.uncited),
        "uncited_rate": _ratio(sum(1 for c in claims if c.uncited), len(claims)),
        "citations": len(checks),
        "missing_sources": sum(1 for c in checks if c.verdict == MISSING),
        "unclear": sum(1 for c in judged if c.verdict == UNCLEAR),
        # Of the citations the model made, how many point at a passage that
        # actually says the thing.
        "citation_precision": _ratio(len(supported), len(judged)),
        # Of the claims that cited anything, how many have at least one source
        # that holds up.
        "claim_support": _ratio(
            sum(1 for c in cited_claims if c.start in backed), len(cited_claims)
        ),
    }


def _ratio(numerator: int, denominator: int) -> float:
    """Zero rather than a ZeroDivisionError. An answer with no claims has no
    failure rate, and crashing the harness over it would lose the run."""
    return numerator / denominator if denominator else 0.0
