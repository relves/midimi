"""Persistent Sequence entity: DB schema, CRUD, and revision storage."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

_JSON = json  # alias to avoid shadowing

# Injected at startup (same DB_PATH used by server.py)
_db_path: Path | None = None


def init_model(db_path: Path) -> None:
    global _db_path
    _db_path = db_path
    _create_tables()


def _conn() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("model not initialized — call init_model(db_path) first")
    return sqlite3.connect(_db_path)


def _create_tables() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sequences (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                title TEXT,
                abc TEXT,
                tempo_bpm REAL,
                time_signature TEXT,
                key TEXT,
                source TEXT,
                created_at INTEGER,
                modified_at INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sequence_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id TEXT,
                abc TEXT,
                created_at INTEGER
            )
        """)
        # Phase 4: raw_events column for recordings
        try:
            conn.execute("ALTER TABLE sequences ADD COLUMN raw_events TEXT")
        except Exception:
            pass
        conn.commit()


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_sequence(
    *,
    title: str,
    abc: str,
    session_id: str | None = None,
    tempo_bpm: float = 120.0,
    time_signature: str = "4/4",
    key: str = "C",
    source: str = "agent",
    raw_events: list | None = None,
) -> str:
    """Insert a new sequence, store initial revision, return its id."""
    seq_id = str(uuid.uuid4())[:8]
    now = int(time.time())
    raw_json = _JSON.dumps(raw_events) if raw_events is not None else None
    with _conn() as conn:
        conn.execute(
            """INSERT INTO sequences
               (id, session_id, title, abc, tempo_bpm, time_signature, key, source, raw_events, created_at, modified_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (seq_id, session_id, title, abc, tempo_bpm, time_signature, key, source, raw_json, now, now),
        )
        conn.execute(
            "INSERT INTO sequence_revisions (sequence_id, abc, created_at) VALUES (?,?,?)",
            (seq_id, abc, now),
        )
        conn.commit()
    return seq_id


def get_sequence(seq_id: str) -> dict[str, Any] | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, session_id, title, abc, tempo_bpm, time_signature, key, source, raw_events, created_at, modified_at "
            "FROM sequences WHERE id=?",
            (seq_id,),
        ).fetchone()
    if not row:
        return None
    raw_events = _JSON.loads(row[8]) if row[8] else None
    return {
        "id": row[0], "session_id": row[1], "title": row[2], "abc": row[3],
        "tempo_bpm": row[4], "time_signature": row[5], "key": row[6],
        "source": row[7], "raw_events": raw_events, "created_at": row[9], "modified_at": row[10],
    }


def update_sequence(seq_id: str, *, abc: str) -> bool:
    """Replace the ABC for a sequence, append a revision. Returns False if not found."""
    now = int(time.time())
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE sequences SET abc=?, modified_at=? WHERE id=?",
            (abc, now, seq_id),
        )
        if cur.rowcount == 0:
            return False
        conn.execute(
            "INSERT INTO sequence_revisions (sequence_id, abc, created_at) VALUES (?,?,?)",
            (seq_id, abc, now),
        )
        conn.commit()
    return True


def list_sequences(session_id: str | None = None) -> list[dict[str, Any]]:
    with _conn() as conn:
        if session_id is not None:
            rows = conn.execute(
                "SELECT id, title, time_signature, tempo_bpm, source, modified_at "
                "FROM sequences WHERE session_id=? ORDER BY modified_at DESC",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title, time_signature, tempo_bpm, source, modified_at "
                "FROM sequences ORDER BY modified_at DESC"
            ).fetchall()
    return [
        {"id": r[0], "title": r[1], "time_signature": r[2],
         "tempo_bpm": r[3], "source": r[4], "modified_at": r[5]}
        for r in rows
    ]


def get_revisions(seq_id: str) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, abc, created_at FROM sequence_revisions WHERE sequence_id=? ORDER BY id",
            (seq_id,),
        ).fetchall()
    return [{"id": r[0], "abc": r[1], "created_at": r[2]} for r in rows]
