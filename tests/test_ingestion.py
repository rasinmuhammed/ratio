from rag.ingest import exclusion_reason, normalize_text


def test_collapses_space_runs():
    assert normalize_text("a     b") == "a b"


def test_preserves_newlines():
    assert normalize_text("a\nb") == "a\nb"


def test_collapses_excess_blank_lines():
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


def test_excludes_empty_text():
    assert exclusion_reason({"Doc_url": "u", "Text": "   "}) == "empty_text"


def test_accepts_valid_row():
    assert exclusion_reason({"Doc_url": "u", "Text": "real"}) is None