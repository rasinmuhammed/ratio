from rag.ingest import Document
from rag.chunk import chunk_document, _split_numbered, _overlap_tail


def test_never_exceeds_target():
    doc = Document(id="d", text="word " * 4000,
                   metadata={"court": "Patna High Court", "title": "A vs B"})
    assert all(len(c.text) <= 1200 for c in chunk_document(doc))


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