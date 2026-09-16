"""Storage and lookup for the citation-treatment graph built by
scripts/build_treatment_graph.py.

Every function that opens the database creates the table first, following
the same lesson audit.py's history already paid for once: a query function
that assumes some other function's init already ran breaks the moment it is
called from a fresh checkout or a different process, so every connection
point is self-sufficient instead.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path("data/treatment.db")


@dataclass(frozen=True, slots=True)
class Mention:
    source_doc_id: str
    treatment: str
    raw: str
    snippet: str


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS mentions ("
        " citation_key TEXT NOT NULL,"
        " source_doc_id TEXT NOT NULL,"
        " treatment TEXT NOT NULL,"
        " raw TEXT NOT NULL,"
        " snippet TEXT NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mentions_key ON mentions(citation_key)"
    )
    return conn


def record_mentions(
    rows: list[tuple[str, str, str, str, str]], db_path: Path = DB_PATH,
) -> None:
    """Bulk-insert (citation_key, source_doc_id, treatment, raw, snippet) rows."""
    conn = _connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO mentions VALUES (?, ?, ?, ?, ?)", rows,
        )
        conn.commit()
    finally:
        conn.close()


def mentions_for(citation_key: str, db_path: Path = DB_PATH) -> list[Mention]:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT source_doc_id, treatment, raw, snippet FROM mentions "
            "WHERE citation_key = ?",
            (citation_key,),
        )
        return [Mention(*row) for row in cur.fetchall()]
    finally:
        conn.close()


def worst_treatment(citation_key: str, db_path: Path = DB_PATH) -> str | None:
    """The single most serious treatment recorded anywhere in the corpus for
    this citation, or None if it was never mentioned. Used to decide whether
    a passage citing this case is worth flagging at all, one query per
    citation rather than pulling every mention just to find the max.
    """
    from rag.treatment import SEVERITY

    mentions = mentions_for(citation_key, db_path)
    if not mentions:
        return None
    return max((m.treatment for m in mentions), key=lambda t: SEVERITY[t])


def stats(db_path: Path = DB_PATH) -> dict:
    conn = _connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        distinct = conn.execute(
            "SELECT COUNT(DISTINCT citation_key) FROM mentions"
        ).fetchone()[0]
        by_treatment = dict(conn.execute(
            "SELECT treatment, COUNT(*) FROM mentions GROUP BY treatment"
        ).fetchall())
        return {"total_mentions": total, "distinct_citations": distinct,
                "by_treatment": by_treatment}
    finally:
        conn.close()
