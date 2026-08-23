from rag.chunk import (
    OVERLAP_CHARS,
    _hard_split,
    _overlap_tail,
    _split_numbered,
    chunk_document,
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