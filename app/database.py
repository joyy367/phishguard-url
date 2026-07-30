"""
database.py
-----------
SQLite-backed, browser-session-isolated scan history with privacy redaction.
Credentials, query strings and fragments are stripped before storage.
"""

import os
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_ROOT, "reports", "scan_history.db")


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------

def _redact_url(url: str) -> str:
    """
    Remove credentials, query string and fragment from a URL before storage.
    e.g. https://user:pass@example.com/path?token=abc#frag
         → https://example.com/path
    """
    try:
        p = urlsplit(url)
        # Strip credentials from netloc. Bracket IPv6 literals when rebuilt.
        hostname = p.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        try:
            port = p.port
        except ValueError:
            port = None
        if port:
            netloc = f"{netloc}:{port}"
        cleaned = urlunsplit((p.scheme, netloc, p.path, "", ""))
        return cleaned
    except Exception:
        # Never return the original string because it may contain credentials.
        return "[unparseable URL]"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the scans table if it does not exist."""
    conn = _get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                session_id    TEXT    NOT NULL DEFAULT 'local',
                display_url   TEXT    NOT NULL,
                prediction    TEXT    NOT NULL,
                phish_prob    REAL    NOT NULL,
                risk_level    TEXT    NOT NULL,
                model_name    TEXT    NOT NULL
            )
        """)
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(scans)").fetchall()
        }
        if "session_id" not in columns:
            conn.execute(
                "ALTER TABLE scans ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy'"
            )
    conn.close()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def save_scan(url: str, prediction: str, phish_prob: float,
              risk_level: str, model_name: str, session_id: str = "local"):
    """Persist a redacted scan record."""
    init_db()
    display_url = _redact_url(url)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO scans (timestamp, session_id, display_url, prediction,
                               phish_prob, risk_level, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                session_id,
                display_url,
                prediction,
                phish_prob,
                risk_level,
                model_name,
            ),
        )
    conn.close()


def load_scans(limit: int = 200, session_id: str = "local") -> list:
    """Return recent scans belonging only to *session_id*."""
    init_db()
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM scans WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_summary(session_id: str = "local") -> dict:
    """Return aggregate counts for one browser session."""
    init_db()
    conn = _get_connection()
    row = conn.execute("""
        SELECT
            COUNT(*)                                          AS total,
            COALESCE(SUM(CASE WHEN prediction='Phishing'   THEN 1 ELSE 0 END), 0) AS phishing,
            COALESCE(SUM(CASE WHEN prediction='Legitimate' THEN 1 ELSE 0 END), 0) AS legitimate,
            COALESCE(SUM(CASE WHEN risk_level='High'       THEN 1 ELSE 0 END), 0) AS high_risk,
            COALESCE(SUM(CASE WHEN risk_level='Medium'     THEN 1 ELSE 0 END), 0) AS medium_risk,
            COALESCE(SUM(CASE WHEN risk_level='Low'        THEN 1 ELSE 0 END), 0) AS low_risk
        FROM scans
        WHERE session_id=?
    """, (session_id,)).fetchone()
    conn.close()
    return dict(row)


def clear_history(session_id: str = "local"):
    """Delete scan records for one browser session only."""
    init_db()
    conn = _get_connection()
    with conn:
        conn.execute("DELETE FROM scans WHERE session_id=?", (session_id,))
    conn.close()
