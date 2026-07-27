import sqlite3
from pathlib import Path

import src.reporting.history_db as history_db
from src.reporting.evidence_snapshot import REVIEW_REQUIRED, build_evidence_bundle


def _test_db(name: str) -> Path:
    path = Path("tests_tmp") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return path


def test_old_schema_is_migrated_without_losing_rows(monkeypatch) -> None:
    db_path = _test_db("reporting_old_schema.db")
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                scanned_at TEXT NOT NULL,
                scan_type TEXT NOT NULL,
                source_name TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_name TEXT,
                preview TEXT,
                flags TEXT,
                explanation TEXT,
                raw_input TEXT,
                report_note TEXT,
                source_fingerprint TEXT UNIQUE NOT NULL
            );
            INSERT INTO scan_history (
                session_id, scanned_at, scan_type, prediction, confidence, source_fingerprint
            ) VALUES (
                'local-capstone-demo', '2026-01-01T00:00:00', 'Audio',
                'Uploaded recording chunk analysis', 99, 'legacy-one'
            );
            """
        )

    monkeypatch.setattr(history_db, "DB_PATH", db_path)
    monkeypatch.setattr(history_db, "_LEGACY_MIGRATED", True)
    history_db.init_db()
    rows = history_db.query_history()

    assert len(rows) == 1
    assert rows[0]["native_prediction"] == "Uploaded recording chunk analysis"
    assert rows[0]["action_status"] == REVIEW_REQUIRED
    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(scan_history)")}
    assert {"native_prediction", "action_status", "evidence_bundle", "provenance"} <= columns


def test_new_snapshot_round_trips_through_sqlite(monkeypatch) -> None:
    db_path = _test_db("reporting_snapshot.db")
    monkeypatch.setattr(history_db, "DB_PATH", db_path)
    monkeypatch.setattr(history_db, "_LEGACY_MIGRATED", True)
    bundle = build_evidence_bundle(
        evidence_type="Email",
        source_input={"source_name": "message.eml"},
        dashboard_summary={"final_verdict": "Suspicious"},
    )

    scan_id = history_db.insert_scan(
        scan_type="Email",
        prediction="Suspicious",
        native_prediction="Suspicious",
        confidence=91,
        concern_score=76,
        score_label="Average suspicious risk",
        score_available=True,
        action_status="Immediate Action Required",
        raw_input="urgent transfer",
        evidence_bundle=bundle,
    )
    row = history_db.query_history()[0]

    assert scan_id > 0
    assert row["concern_score"] == 76
    assert row["action_status"] == "Immediate Action Required"
    assert '"schema_version": "1.0"' in row["evidence_bundle"]
