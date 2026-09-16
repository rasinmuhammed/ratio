from rag.treatment import DISTINGUISHED, FOLLOWED, OVERRULED, REFERRED
from rag.treatment_db import mentions_for, record_mentions, stats, worst_treatment


def test_fresh_db_path_is_created_on_first_write(tmp_path):
    """No setup step is required first, matching audit.py's own lesson:
    every entry point creates its directory and schema itself."""
    db_path = tmp_path / "nested" / "treatment.db"
    record_mentions([("AIR 1978 SC 597", "doc1", REFERRED, "AIR 1978 SC 597", "...")],
                     db_path=db_path)
    assert db_path.exists()


def test_mentions_for_an_unseen_citation_is_empty(tmp_path):
    db_path = tmp_path / "treatment.db"
    assert mentions_for("AIR 1999 SC 1", db_path=db_path) == []


def test_worst_treatment_is_none_for_an_unmentioned_citation(tmp_path):
    db_path = tmp_path / "treatment.db"
    assert worst_treatment("AIR 1999 SC 1", db_path=db_path) is None


def test_worst_treatment_picks_the_most_serious_of_several_mentions(tmp_path):
    """A case followed nine times and overruled once is not safe to cite
    without the warning; the single worst mention is what a reader needs to
    see, not the majority vote."""
    db_path = tmp_path / "treatment.db"
    record_mentions([
        ("AIR 1978 SC 597", "doc1", FOLLOWED, "AIR 1978 SC 597", "..."),
        ("AIR 1978 SC 597", "doc2", FOLLOWED, "AIR 1978 SC 597", "..."),
        ("AIR 1978 SC 597", "doc3", OVERRULED, "AIR 1978 SC 597", "..."),
    ], db_path=db_path)
    assert worst_treatment("AIR 1978 SC 597", db_path=db_path) == OVERRULED


def test_distinguished_outranks_followed_but_not_overruled(tmp_path):
    db_path = tmp_path / "treatment.db"
    record_mentions([
        ("X", "doc1", FOLLOWED, "X", "..."),
        ("X", "doc2", DISTINGUISHED, "X", "..."),
    ], db_path=db_path)
    assert worst_treatment("X", db_path=db_path) == DISTINGUISHED


def test_mentions_are_scoped_to_their_own_citation_key(tmp_path):
    db_path = tmp_path / "treatment.db"
    record_mentions([
        ("A", "doc1", OVERRULED, "A", "..."),
        ("B", "doc2", REFERRED, "B", "..."),
    ], db_path=db_path)
    assert [m.source_doc_id for m in mentions_for("A", db_path=db_path)] == ["doc1"]


def test_stats_summarizes_totals_and_treatment_counts(tmp_path):
    db_path = tmp_path / "treatment.db"
    record_mentions([
        ("A", "doc1", OVERRULED, "A", "..."),
        ("A", "doc2", OVERRULED, "A", "..."),
        ("B", "doc3", REFERRED, "B", "..."),
    ], db_path=db_path)
    result = stats(db_path=db_path)
    assert result["total_mentions"] == 3
    assert result["distinct_citations"] == 2
    assert result["by_treatment"][OVERRULED] == 2
    assert result["by_treatment"][REFERRED] == 1
