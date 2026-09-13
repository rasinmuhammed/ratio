"""Audit logging: Persist request metadata to SQLite."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/audit.db")


def _connect() -> sqlite3.Connection:
    """Every function below used to call sqlite3.connect(DB_PATH) directly,
    which only worked because data/ already existed on whatever machine
    happened to run this first. init_db() creates the directory, but
    nothing enforced init_db() running before log_request, log_correction,
    save_session or load_session, and a fresh checkout with no data/ at
    all (a CI runner, a first-time clone) hit sqlite3.OperationalError:
    unable to open database file the first time any of them ran without
    init_db() having been called first. Centralising the connection here
    means the directory exists regardless of call order.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the audit table if it does not exist."""
    with _connect() as conn:
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dpo_feedback (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                original_answer TEXT NOT NULL,
                corrected_answer TEXT NOT NULL,
                ts TEXT NOT NULL
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
    with _connect() as conn:
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

def save_session(session_id: str, messages: list[dict]) -> None:
    """Save an agent conversation session to the database."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO agent_sessions (session_id, messages, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                messages=excluded.messages,
                updated_at=excluded.updated_at
        """, (
            session_id,
            json.dumps(messages),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ))

def load_session(session_id: str) -> list[dict] | None:
    """Load an agent conversation session from the database."""
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT messages FROM agent_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

def log_correction(query: str, original_answer: str, corrected_answer: str) -> None:
    """Save human-in-the-loop feedback for Direct Preference Optimization (DPO)."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO dpo_feedback (query, original_answer, corrected_answer, ts)
            VALUES (?, ?, ?, ?)
        """, (
            query,
            original_answer,
            corrected_answer,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ))
