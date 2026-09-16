"""Has this precedent held up?

A citation is not a fact about the world, it is a claim by whoever cited it.
"Followed in AIR 1978 SC 597" and "AIR 1978 SC 597 was distinguished on
facts" cite the same case to opposite effect, and nothing about retrieval,
dense or BM25, can see that difference: both passages score on whether they
mention the citation, not on what the citing court did with it.

This does not try to resolve a citation to a specific document in this
corpus. A first attempt did: match "AIR 1978 SC 597" against a document that
states that citation about itself, and build doc-to-doc edges. Measured
against a 4,000-document sample, self-citations appear in roughly 2% of
documents (most of this corpus is unreported first-instance orders that
never receive an AIR/SCC citation for themselves), so a graph keyed that way
would resolve almost nothing. What is common, 25,000+ citation-shaped
mentions in that same sample, is a *reference* to a citation string,
regardless of whether the referenced case is itself in the corpus. So the
graph here is keyed by the citation string as text: every mention of
"AIR 1978 SC 597" anywhere in the corpus, with a shallow read of how each
citing passage treated it, whether or not that case was ever scraped.

Two reporters are covered: AIR and SCC/SCC OnLine. Between them they cover
the large majority of citations to reported Indian case law; a citation in
any other reporter, or a bare party-name citation ("Kesavananda Bharati"),
is invisible to this and stays invisible rather than being guessed at.

The treatment classifier is the same kind of deliberately shallow rule set
as stance.py: it matches the handful of reporting formulas Indian judgments
actually use for negative treatment, and defaults to "referred" (mentioned,
no stated treatment) for everything else, which is the honest answer for a
citation dropped into a string of authorities with no individual discussion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OVERRULED = "overruled"
DISTINGUISHED = "distinguished"
FOLLOWED = "followed"
REFERRED = "referred"

# Both patterns are deliberately loose on the court abbreviation (SC, Ker,
# Bom, All, ...) and reporter-part number, because getting the citation
# string exactly byte-for-byte right does not matter here, two mentions of
# "the same" case only need to normalize to the same key consistently, and a
# citation is never resolved against an external table anyway.
_AIR = re.compile(
    r"\bAIR\s+(\d{4})\s+([A-Za-z]{2,8})\s+(\d{1,5})\b"
)
_SCC = re.compile(
    r"\(?(\d{4})\)?\s*(\d{1,2})?\s*SCC(\s+OnLine)?\s+([A-Za-z]{0,4})\s*(\d{1,5})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Citation:
    key: str          # normalized, for grouping mentions of "the same" citation
    raw: str          # exactly as it appeared, for display
    start: int
    end: int


def find_citations(text: str) -> list[Citation]:
    """All AIR/SCC-shaped citation mentions in `text`, in order of appearance."""
    out: list[Citation] = []
    for m in _AIR.finditer(text):
        year, court, number = m.group(1), m.group(2).upper(), m.group(3)
        out.append(Citation(
            key=f"AIR {year} {court} {number}",
            raw=m.group(0), start=m.start(), end=m.end(),
        ))
    for m in _SCC.finditer(text):
        year, online, court, number = m.group(1), m.group(3), m.group(4), m.group(5)
        online_part = " OnLine" if online else ""
        court_part = f" {court.upper()}" if court else ""
        out.append(Citation(
            key=f"{year} SCC{online_part}{court_part} {number}",
            raw=m.group(0), start=m.start(), end=m.end(),
        ))
    out.sort(key=lambda c: c.start)
    return out


# Signal words are matched in a fixed window around the citation rather than
# anywhere in the document: "distinguished" appearing three paragraphs later,
# about a different case, is not a treatment of this one. The window is
# characters, not tokens, to match how the citation's own position is
# reported and to stay independent of any tokenizer.
WINDOW_CHARS = 200

_OVERRULED = re.compile(
    r"\b(?:overrul(?:ed|ing)|no\s+longer\s+(?:good\s+law|holds?\s+the\s+field)|"
    r"stands?\s+overruled)\b", re.IGNORECASE,
)
_DISTINGUISHED = re.compile(
    r"\b(?:distinguish(?:ed|able)|(?:not|in)applicable\s+(?:to|on)\s+the\s+"
    r"(?:facts|present\s+case)|does\s+not\s+apply\s+(?:to|here|in\s+the\s+"
    r"present\s+case))\b", re.IGNORECASE,
)
_FOLLOWED = re.compile(
    r"\b(?:followed|affirmed|approved|reiterated|relied\s+(?:upon|on)|"
    r"applies\s+(?:squarely|with\s+full\s+force)|squarely\s+covers?)\b",
    re.IGNORECASE,
)


def classify_treatment(text: str, citation: Citation) -> str:
    """How the passage around one citation treats it.

    Checked in this order because a passage can plausibly contain more than
    one signal ("distinguished, and in any event no longer good law after
    X"), and overruled is the strongest, most specific claim among the three
    when several appear together.
    """
    window = text[max(0, citation.start - WINDOW_CHARS):citation.end + WINDOW_CHARS]
    if _OVERRULED.search(window):
        return OVERRULED
    if _DISTINGUISHED.search(window):
        return DISTINGUISHED
    if _FOLLOWED.search(window):
        return FOLLOWED
    return REFERRED

# Ordering for "is there anything worth warning about": referred is the
# common, uninteresting case, followed is reassuring rather than a warning,
# and the two negative treatments are what a reader would actually want
# surfaced before relying on a citation.
SEVERITY = {OVERRULED: 3, DISTINGUISHED: 2, FOLLOWED: 1, REFERRED: 0}
