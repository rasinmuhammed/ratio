"""Audit logging: Persist request metadata to SQLite."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/audit.db")

def init_db() -> None:
    """Create the audit table if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_log (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                query TEXT NOT NULL,
                route TEXT,
                refused INTEGER,
                n_sources INTEGER,
                n_cited INTEGER,
                invalid_cites TEXT,
                score_gap REAL,
                truncated INTEGER,
                retriever_ms INTEGER,
                generation_ms INTEGER,
                llm_model TEXT
            )
        """)

def log_request(
    query: str,
    route: str,
    refused: bool,
    n_sources: int,
    n_cited: int,
    invalid_cites: list[int],
    score_gap: float,
    truncated: bool,
    retriever_ms: int,
    generation_ms: int,
    llm_model: str,
) -> None:
    """Append a single request log to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO request_log (
                ts, query, route, refused, n_sources, n_cited,
                invalid_cites, score_gap, truncated, retriever_ms,
                generation_ms, llm_model
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            query,
            route,
            int(refused),
            n_sources,
            n_cited,
            str(invalid_cites) if invalid_cites else None,
            score_gap,
            int(truncated),
            retriever_ms,
            generation_ms,
            llm_model
        ))
