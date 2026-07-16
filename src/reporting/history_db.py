"""SQLite-backed scan history for AI-FDS reports.

This module gives the report generator a durable evidence store while keeping
the current Streamlit scan pages simple. Existing pages can continue appending
dicts to ``st.session_state.history``; the report page syncs those dicts into
SQLite when it opens. Future pages can call ``insert_scan`` directly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.utils.time_utils import now_for_app


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "session_history.db"
LEGACY_DB_PATH = ROOT / "src" / "data" / "session_history.db"
DEFAULT_SESSION_ID = "local-capstone-demo"
_LEGACY_MIGRATED = False


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scan_history (
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

CREATE INDEX IF NOT EXISTS idx_scan_history_session_time
    ON scan_history (session_id, scanned_at DESC);

CREATE TABLE IF NOT EXISTS report_exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exported_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    format TEXT NOT NULL,
    scan_ids TEXT NOT NULL,
    filename TEXT NOT NULL
);
"""


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)


def _migrate_legacy_history() -> None:
    global _LEGACY_MIGRATED
    if _LEGACY_MIGRATED:
        return
    _LEGACY_MIGRATED = True

    if not LEGACY_DB_PATH.exists() or LEGACY_DB_PATH.resolve() == DB_PATH.resolve():
        return

    try:
        with sqlite3.connect(LEGACY_DB_PATH) as legacy, _connect() as target:
            legacy.row_factory = sqlite3.Row
            tables = {
                row["name"]
                for row in legacy.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "scan_history" not in tables:
                return

            _ensure_schema(target)
            columns = [
                "session_id",
                "scanned_at",
                "scan_type",
                "source_name",
                "prediction",
                "confidence",
                "model_name",
                "preview",
                "flags",
                "explanation",
                "raw_input",
                "report_note",
                "source_fingerprint",
            ]
            column_sql = ", ".join(columns)
            placeholders = ", ".join("?" for _ in columns)
            for row in legacy.execute(f"SELECT {column_sql} FROM scan_history"):
                target.execute(
                    f"INSERT OR IGNORE INTO scan_history ({column_sql}) VALUES ({placeholders})",
                    [row[column] for column in columns],
                )
    except sqlite3.Error:
        return


def init_db() -> None:
    """Create the report history tables if they do not already exist."""

    with _connect() as connection:
        _ensure_schema(connection)
    _migrate_legacy_history()


def _string(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _normalise_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")

    text = _string(value)
    if not text:
        return now_for_app().replace(microsecond=0).isoformat()

    # Current app timestamps may be formatted as:
    # "2026-06-08 14:30:00 (Asia/Kuala_Lumpur, GMT+08:00)".
    cleaned = text.split(" (", 1)[0].strip().replace("Z", "+00:00")
    for candidate in (cleaned, cleaned[:19]):
        try:
            return datetime.fromisoformat(candidate.replace(" ", "T")).replace(microsecond=0).isoformat()
        except ValueError:
            continue

    return now_for_app().replace(microsecond=0).isoformat()


def _normalise_confidence(value: object) -> float:
    try:
        confidence = float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        confidence = 0.0
    if 0 < confidence <= 1:
        confidence *= 100
    return max(0.0, min(100.0, confidence))


def _normalise_flags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return _normalise_flags(decoded)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
    if isinstance(value, dict):
        phrase = value.get("phrase") or value.get("label") or value.get("category") or value.get("reason")
        return [_string(phrase)] if phrase else []
    if isinstance(value, Iterable):
        flags: list[str] = []
        for item in value:
            flags.extend(_normalise_flags(item))
        return [flag for flag in flags if flag]
    return [_string(value)]


def normalise_history_item(item: dict[str, object], session_id: str = DEFAULT_SESSION_ID) -> dict[str, object]:
    """Convert a loose Streamlit history entry into the database schema."""

    scanned_at = _normalise_timestamp(item.get("scanned_at") or item.get("time"))
    scan_type = _string(item.get("scan_type") or item.get("type"), "Unknown")
    preview = _string(item.get("preview") or item.get("source_name"), "")[:800]
    model_name = _string(item.get("model_name") or item.get("model"), "")
    flags = _normalise_flags(item.get("flags") or item.get("findings"))

    row = {
        "session_id": _string(item.get("session_id"), session_id),
        "scanned_at": scanned_at,
        "scan_type": scan_type,
        "source_name": _string(item.get("source_name") or item.get("filename") or preview[:80], ""),
        "prediction": _string(item.get("prediction") or item.get("label_name"), "Unknown"),
        "confidence": _normalise_confidence(item.get("confidence")),
        "model_name": model_name,
        "preview": preview,
        "flags": flags,
        "explanation": _string(item.get("explanation") or item.get("summary"), ""),
        "raw_input": _string(item.get("raw_input") or item.get("input_text") or item.get("text"), ""),
        "report_note": _string(item.get("report_note"), ""),
    }
    row["source_fingerprint"] = _string(item.get("source_fingerprint")) or history_fingerprint(row)
    return row


def history_fingerprint(item: dict[str, object]) -> str:
    """Build a stable fingerprint to avoid duplicate report history rows."""

    payload = {
        "scanned_at": _normalise_timestamp(item.get("scanned_at") or item.get("time")),
        "scan_type": _string(item.get("scan_type") or item.get("type")),
        "prediction": _string(item.get("prediction") or item.get("label_name")),
        "confidence": _normalise_confidence(item.get("confidence")),
        "model_name": _string(item.get("model_name") or item.get("model")),
        "preview": _string(item.get("preview")),
        "chunks": _string(item.get("chunks")),
    }
    packed = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def insert_scan(
    *,
    scan_type: str,
    prediction: str,
    confidence: float,
    session_id: str = DEFAULT_SESSION_ID,
    scanned_at: str | None = None,
    source_name: str = "",
    model_name: str = "",
    preview: str = "",
    flags: list[str] | None = None,
    explanation: str = "",
    raw_input: str = "",
    report_note: str = "",
    source_fingerprint: str | None = None,
) -> int:
    """Insert one scan result and return its row id."""

    init_db()
    row = {
        "session_id": session_id,
        "scanned_at": _normalise_timestamp(scanned_at),
        "scan_type": scan_type,
        "source_name": source_name,
        "prediction": prediction,
        "confidence": _normalise_confidence(confidence),
        "model_name": model_name,
        "preview": preview[:800],
        "flags": flags or [],
        "explanation": explanation,
        "raw_input": raw_input,
        "report_note": report_note,
    }
    fingerprint = source_fingerprint or history_fingerprint(row)

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO scan_history (
                session_id, scanned_at, scan_type, source_name, prediction,
                confidence, model_name, preview, flags, explanation, raw_input,
                report_note, source_fingerprint
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["session_id"],
                row["scanned_at"],
                row["scan_type"],
                row["source_name"],
                row["prediction"],
                row["confidence"],
                row["model_name"],
                row["preview"],
                json.dumps(row["flags"], ensure_ascii=True),
                row["explanation"],
                row["raw_input"],
                row["report_note"],
                fingerprint,
            ),
        )
        if cursor.rowcount:
            return int(cursor.lastrowid)
        existing = connection.execute(
            "SELECT id FROM scan_history WHERE source_fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return int(existing["id"]) if existing else 0


def sync_session_history(history: list[dict[str, object]], session_id: str = DEFAULT_SESSION_ID) -> int:
    """Persist loose Streamlit history entries into SQLite."""

    init_db()
    inserted = 0
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        row = normalise_history_item(item, session_id=session_id)
        before = query_by_fingerprint(str(row["source_fingerprint"]))
        insert_scan(
            session_id=str(row["session_id"]),
            scanned_at=str(row["scanned_at"]),
            scan_type=str(row["scan_type"]),
            source_name=str(row["source_name"]),
            prediction=str(row["prediction"]),
            confidence=float(row["confidence"]),
            model_name=str(row["model_name"]),
            preview=str(row["preview"]),
            flags=list(row["flags"]),
            explanation=str(row["explanation"]),
            raw_input=str(row["raw_input"]),
            report_note=str(row["report_note"]),
            source_fingerprint=str(row["source_fingerprint"]),
        )
        if before is None:
            inserted += 1
    return inserted


def record_history_item(
    history: list[dict[str, object]],
    item: dict[str, object],
    session_id: str = DEFAULT_SESSION_ID,
) -> int:
    """Persist one scan result immediately and mirror it in Streamlit session history."""

    row = normalise_history_item(item, session_id=session_id)
    item["source_fingerprint"] = row["source_fingerprint"]
    history.insert(0, item)
    return insert_scan(
        session_id=str(row["session_id"]),
        scanned_at=str(row["scanned_at"]),
        scan_type=str(row["scan_type"]),
        source_name=str(row["source_name"]),
        prediction=str(row["prediction"]),
        confidence=float(row["confidence"]),
        model_name=str(row["model_name"]),
        preview=str(row["preview"]),
        flags=list(row["flags"]),
        explanation=str(row["explanation"]),
        raw_input=str(row["raw_input"]),
        report_note=str(row["report_note"]),
        source_fingerprint=str(row["source_fingerprint"]),
    )


def query_by_fingerprint(source_fingerprint: str) -> dict[str, object] | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM scan_history WHERE source_fingerprint = ?",
            (source_fingerprint,),
        ).fetchone()
    return dict(row) if row else None


def query_history(
    *,
    session_id: str = DEFAULT_SESSION_ID,
    date_from: str | None = None,
    date_to: str | None = None,
    scan_types: list[str] | None = None,
    predictions: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Return filtered report history rows."""

    init_db()
    clauses = ["session_id = ?"]
    params: list[object] = [session_id]
    if date_from:
        clauses.append("scanned_at >= ?")
        params.append(f"{date_from}T00:00:00")
    if date_to:
        clauses.append("scanned_at <= ?")
        params.append(f"{date_to}T23:59:59")
    if scan_types:
        placeholders = ",".join("?" for _ in scan_types)
        clauses.append(f"scan_type IN ({placeholders})")
        params.extend(scan_types)
    if predictions:
        placeholders = ",".join("?" for _ in predictions)
        clauses.append(f"prediction IN ({placeholders})")
        params.extend(predictions)

    sql = f"SELECT * FROM scan_history WHERE {' AND '.join(clauses)} ORDER BY scanned_at DESC, id DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    with _connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def delete_selected(scan_ids: list[int]) -> int:
    """Delete selected scan history rows."""

    if not scan_ids:
        return 0
    init_db()
    placeholders = ",".join("?" for _ in scan_ids)
    with _connect() as connection:
        cursor = connection.execute(
            f"DELETE FROM scan_history WHERE id IN ({placeholders})",
            [int(scan_id) for scan_id in scan_ids],
        )
        return int(cursor.rowcount)


def delete_all_history(session_id: str = DEFAULT_SESSION_ID) -> int:
    """Delete all scan and export rows for the current local capstone session."""

    init_db()
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM scan_history WHERE session_id = ?", (session_id,))
        connection.execute("DELETE FROM report_exports WHERE session_id = ?", (session_id,))
        return int(cursor.rowcount)


def log_export(
    *,
    report_format: str,
    scan_ids: list[int],
    filename: str,
    session_id: str = DEFAULT_SESSION_ID,
) -> int:
    """Record that a report export was generated."""

    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO report_exports (exported_at, session_id, format, scan_ids, filename)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                now_for_app().replace(microsecond=0).isoformat(),
                session_id,
                report_format.upper(),
                json.dumps(scan_ids, ensure_ascii=True),
                filename,
            ),
        )
        return int(cursor.lastrowid)
