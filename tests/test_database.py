"""
test_database.py
----------------
Tests for URL redaction, save/load/clear lifecycle, and summary counts.
Uses a temporary database path so tests do not affect the real history file.
"""

import os
import sqlite3
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Patch the DB path to a temp file for all tests in this module
# Use tempfile.mkdtemp() instead of tmp_path to avoid Windows ACL issues
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    """Redirect all DB operations to a temporary file in a local temp dir."""
    import app.database as db_module
    tmpdir = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(__file__), "..", "reports"))
    temp_db_path = os.path.join(tmpdir, "test_scan_history.db")
    monkeypatch.setattr(db_module, "_DB_PATH", temp_db_path)
    yield temp_db_path
    # Cleanup
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)
    try:
        os.rmdir(tmpdir)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

class TestRedactUrl:

    def test_removes_query_string(self):
        from app.database import _redact_url
        result = _redact_url("https://example.com/path?token=abc&user=xyz")
        assert "token" not in result
        assert "user" not in result

    def test_removes_fragment(self):
        from app.database import _redact_url
        result = _redact_url("https://example.com/page#section")
        assert "#" not in result

    def test_removes_credentials(self):
        from app.database import _redact_url
        result = _redact_url("https://user:password@example.com/path")
        assert "password" not in result
        assert "user" not in result

    def test_malformed_port_never_leaks_credentials(self):
        from app.database import _redact_url
        result = _redact_url("https://user:password@example.com:bad/path?token=abc")
        assert "password" not in result
        assert "user" not in result
        assert "token" not in result

    def test_preserves_scheme_and_host(self):
        from app.database import _redact_url
        result = _redact_url("https://example.com/path?q=1")
        assert "https" in result
        assert "example.com" in result

    def test_preserves_path(self):
        from app.database import _redact_url
        result = _redact_url("https://example.com/some/path?q=1")
        assert "/some/path" in result


# ---------------------------------------------------------------------------
# Save / Load / Clear lifecycle
# ---------------------------------------------------------------------------

class TestDatabaseLifecycle:

    def test_save_and_load(self):
        from app.database import save_scan, load_scans
        save_scan(
            url="https://example.com/page",
            prediction="Legitimate",
            phish_prob=0.05,
            risk_level="Low",
            model_name="Random Forest",
        )
        scans = load_scans()
        assert len(scans) == 1
        assert scans[0]["prediction"] == "Legitimate"

    def test_multiple_saves(self):
        from app.database import save_scan, load_scans
        for i in range(5):
            save_scan(
                url=f"https://example{i}.com",
                prediction="Phishing",
                phish_prob=0.9,
                risk_level="High",
                model_name="Random Forest",
            )
        scans = load_scans()
        assert len(scans) == 5

    def test_load_order_newest_first(self):
        from app.database import save_scan, load_scans
        save_scan("https://first.com", "Legitimate", 0.1, "Low", "RF")
        save_scan("https://second.com", "Phishing", 0.9, "High", "RF")
        scans = load_scans()
        assert scans[0]["display_url"] == "https://second.com"

    def test_clear_history(self):
        from app.database import save_scan, load_scans, clear_history
        save_scan("https://example.com", "Legitimate", 0.05, "Low", "RF")
        clear_history()
        assert load_scans() == []

    def test_query_string_redacted_in_storage(self):
        from app.database import save_scan, load_scans
        save_scan(
            url="https://example.com/page?token=secret123",
            prediction="Legitimate",
            phish_prob=0.1,
            risk_level="Low",
            model_name="Random Forest",
        )
        scans = load_scans()
        assert "secret123" not in scans[0]["display_url"]

    def test_load_returns_list_of_dicts(self):
        from app.database import save_scan, load_scans
        save_scan("https://example.com", "Phishing", 0.8, "High", "RF")
        scans = load_scans()
        assert isinstance(scans, list)
        assert isinstance(scans[0], dict)

    def test_sessions_are_isolated(self):
        from app.database import save_scan, load_scans, clear_history
        save_scan(
            "https://a.com", "Phishing", 0.9, "High", "RF",
            session_id="session-a",
        )
        save_scan(
            "https://b.com", "Legitimate", 0.1, "Low", "RF",
            session_id="session-b",
        )
        assert len(load_scans(session_id="session-a")) == 1
        assert load_scans(session_id="session-a")[0]["display_url"] == "https://a.com"
        clear_history(session_id="session-a")
        assert load_scans(session_id="session-a") == []
        assert len(load_scans(session_id="session-b")) == 1


# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------

class TestGetSummary:

    def test_empty_summary(self):
        from app.database import get_summary
        s = get_summary()
        assert s["total"] == 0
        assert s["phishing"] == 0
        assert s["legitimate"] == 0

    def test_summary_counts(self):
        from app.database import save_scan, get_summary
        save_scan("https://a.com", "Phishing", 0.9, "High", "RF")
        save_scan("https://b.com", "Phishing", 0.7, "Medium", "RF")
        save_scan("https://c.com", "Legitimate", 0.1, "Low", "RF")
        s = get_summary()
        assert s["total"] == 3
        assert s["phishing"] == 2
        assert s["legitimate"] == 1
        assert s["high_risk"] == 1
        assert s["medium_risk"] == 1
        assert s["low_risk"] == 1
