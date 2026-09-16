import pytest

from rag.chunk import (
    MIN_CHUNK_SIZE,
    OVERLAP_CHARS,
    TARGET_SIZE,
    _hard_split,
    _overlap_tail,
    _split_numbered,
    chunk_document,
    context_prefix,
)
from rag.ingest import Document


def test_never_exceeds_target():
    doc = Document(id="d", text="word " * 4000,
                   metadata={"court": "Patna High Court", "title": "A vs B"})
    assert all(len(c.text) <= 1200 for c in chunk_document(doc))


def test_hard_split_respects_budget_in_the_measured_unit():
    """The hard split slices characters but must respect a budget measured by
    `length`. With a length function that is not len(), a naive
    text[i:i+size] silently produces the wrong size. Here 1 unit = 4 chars.
    """
    def four_chars_per_unit(s: str) -> int:
        return len(s) // 4 + 1

    text = "abcd" * 500
    pieces = _hard_split(text, size=20, length=four_chars_per_unit)

    assert all(four_chars_per_unit(p) <= 20 for p in pieces)
    assert "".join(pieces) == text          # nothing lost or duplicated
    assert all(p for p in pieces)           # no empty pieces, loop advances


def test_hard_split_terminates_on_pathological_input():
    """A string with no whitespace and a length function that reports every
    string as oversized must still terminate rather than loop forever."""
    pieces = _hard_split("x" * 100, size=1, length=lambda s: len(s))
    assert len(pieces) == 100


def test_overlap_is_measured_in_characters():
    """OVERLAP_CHARS is characters because slicing is character-indexed.
    Renaming it was the fix; this asserts the unit stays honest."""
    body = "the quick brown fox jumps over the lazy dog " * 20
    tail = _overlap_tail(body, OVERLAP_CHARS)
    assert len(tail) <= OVERLAP_CHARS
    assert len(tail) > OVERLAP_CHARS - 30   # snapping costs a word, not more


def test_numbered_split_needs_space_not_tab():
    text = "\n".join(f"{i}. Paragraph {i} text goes here." for i in range(1, 8))
    assert _split_numbered(text) is not None


def test_rejects_non_ascending_numbering():
    assert _split_numbered("1. A\n1985. B\n3. C\n2. D\n7. E\n") is None


def test_overlap_starts_at_word_boundary():
    assert not _overlap_tail("the Competent Authority acts", 12).startswith("ct")


def test_chunk_is_immutable():
    doc = Document(id="d", text="x " * 2000, metadata={"court": "C", "title": "T"})
    chunk = next(chunk_document(doc))
    try:
        chunk.text = "mutated"
    except AttributeError:
        return
    raise AssertionError("Chunk should be frozen")

def test_oversized_summary_falls_back_to_header_only_prefix():
    """Measured on the full corpus: a document whose own text is mostly a
    list of citations (a consolidated cause-list order) produces a SAC
    summary that is itself citation-dense and can overflow the chunk budget
    on its own, even though the base header (court, title, case number) is
    always short. Dropping just the summary should let chunking succeed
    rather than crashing a corpus-wide reindex over one atypical source."""
    doc = Document(
        id="d", text="word " * 2000,
        metadata={"court": "C", "title": "T",
                  "doc_summary": "x" * (TARGET_SIZE * 2)},
    )
    chunks = list(chunk_document(doc))
    assert chunks
    assert "Summary:" not in chunks[0].text


def test_summary_is_used_when_it_actually_fits():
    doc = Document(
        id="d", text="word " * 2000,
        metadata={"court": "C", "title": "T", "doc_summary": "A short summary."},
    )
    chunks = list(chunk_document(doc))
    assert "Summary: A short summary." in chunks[0].text


def test_pathologically_long_header_with_no_summary_still_raises():
    """The fallback only has a header to fall back to; if that alone
    overflows the budget there is nothing left to drop, and raising is the
    correct, honest failure rather than silently truncating case metadata."""
    doc = Document(
        id="d", text="word " * 2000,
        metadata={"court": "C" * (TARGET_SIZE * 2), "title": "T"},
    )
    with pytest.raises(ValueError, match="Context prefix too long"):
        list(chunk_document(doc))


def test_context_prefix_include_summary_false_omits_it():
    doc = Document(id="d", text="text",
                    metadata={"court": "C", "title": "T", "doc_summary": "S"})
    assert "Summary" not in context_prefix(doc, include_summary=False)
    assert "Summary: S" in context_prefix(doc)
