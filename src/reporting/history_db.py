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

from src.reporting.evidence_snapshot import (
    decode_json_object,
    derive_action_status,
    provenance_record,
)
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
    source_fingerprint TEXT UNIQUE NOT NULL,
    native_prediction TEXT,
    action_status TEXT,
    concern_score REAL,
    score_label TEXT,
    score_available INTEGER,
    evidence_bundle TEXT,
    provenance TEXT
);

CREATE INDEX IF NOT EXISTS idx_scan_history_session_time
    ON scan_history (session_id, scanned_at DESC);

CREATE TABLE IF NOT EXISTS report_exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exported_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    format TEXT NOT NULL,
    scan_ids TEXT NOT NULL,
    filename TEXT NOT NULL,
    report_sha256 TEXT,
    manifest TEXT
);
"""


SCAN_HISTORY_ADDITIVE_COLUMNS = {
    "native_prediction": "TEXT",
    "action_status": "TEXT",
    "concern_score": "REAL",
    "score_label": "TEXT",
    "score_available": "INTEGER",
    "evidence_bundle": "TEXT",
    "provenance": "TEXT",
}

REPORT_EXPORT_ADDITIVE_COLUMNS = {
    "report_sha256": "TEXT",
    "manifest": "TEXT",
}


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    _ensure_columns(connection, "scan_history", SCAN_HISTORY_ADDITIVE_COLUMNS)
    _ensure_columns(connection, "report_exports", REPORT_EXPORT_ADDITIVE_COLUMNS)


def _ensure_columns(
    connection: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    existing = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})")
    }
    for column_name, column_type in columns.items():
        if column_name not in existing:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )


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


def _json_text(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps(value, ensure_ascii=True)
    return json.dumps(value, ensure_ascii=True, default=str)


def _explicit_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return bool(value)


def _phone_snapshot_fields(raw_input: object) -> tuple[float | None, bool | None]:
    data = decode_json_object(raw_input)
    assessment = data.get("assessment", {})
    coverage = data.get("coverage", {})
    score = assessment.get("score") if isinstance(assessment, dict) else None
    score_value = _normalise_confidence(score) if isinstance(score, (int, float)) else None
    complete = None
    if isinstance(coverage, dict) and coverage:
        complete = (
            coverage.get("veriphone") == "success"
            and coverage.get("penipumy") in {"success", "no_match"}
        )
    return score_value, complete


def normalise_history_item(item: dict[str, object], session_id: str = DEFAULT_SESSION_ID) -> dict[str, object]:
    """Convert a loose Streamlit history entry into the database schema."""

    scanned_at = _normalise_timestamp(item.get("scanned_at") or item.get("time"))
    scan_type = _string(item.get("scan_type") or item.get("type"), "Unknown")
    preview = _string(item.get("preview") or item.get("source_name"), "")[:800]
    model_name = _string(item.get("model_name") or item.get("model"), "")
    flags = _normalise_flags(item.get("flags") or item.get("findings"))
    native_prediction = _string(
        item.get("native_prediction") or item.get("source_verdict") or item.get("prediction") or item.get("label_name"),
        "Unknown",
    )
    scan_family = scan_type.casefold()
    concern_value = item.get("concern_score")
    if concern_value is None:
        concern_value = item.get("risk_score")
    evidence_complete = _explicit_bool(item.get("evidence_complete"))
    if concern_value is None and "phone" in scan_family:
        concern_value, phone_complete = _phone_snapshot_fields(item.get("raw_input"))
        if evidence_complete is None:
            evidence_complete = phone_complete
    concern_score = _normalise_confidence(concern_value) if concern_value is not None else None
    score_available = _explicit_bool(item.get("score_available"))
    if score_available is None:
        score_available = concern_score is not None
        if concern_score is None and "transcript" in scan_family:
            concern_score = _normalise_confidence(item.get("confidence"))
            score_available = True
    action_status = _string(item.get("action_status"), "")
    if not action_status:
        action_status = derive_action_status(
            native_prediction=native_prediction,
            evidence_type=scan_type,
            concern_score=concern_score,
            score_available=score_available,
            model_agreement=item.get("model_agreement"),
            evidence_complete=evidence_complete,
            direct_exposure=bool(item.get("direct_exposure")),
        )
    evidence_bundle = _json_text(item.get("evidence_bundle"))
    provenance = _json_text(item.get("provenance"))
    raw_input = _string(item.get("raw_input") or item.get("input_text") or item.get("text"), "")
    if not provenance and raw_input:
        provenance = _json_text(
            provenance_record(
                raw_input,
                source_name=_string(item.get("source_name") or item.get("filename") or preview[:80], ""),
                source_kind=scan_type,
                captured_at=scanned_at,
            )
        )

    row = {
        "session_id": _string(item.get("session_id"), session_id),
        "scanned_at": scanned_at,
        "scan_type": scan_type,
        "source_name": _string(item.get("source_name") or item.get("filename") or preview[:80], ""),
        "prediction": _string(item.get("prediction") or item.get("label_name"), native_prediction),
        "confidence": _normalise_confidence(item.get("confidence")),
        "model_name": model_name,
        "preview": preview,
        "flags": flags,
        "explanation": _string(item.get("explanation") or item.get("summary"), ""),
        "raw_input": raw_input,
        "report_note": _string(item.get("report_note"), ""),
        "native_prediction": native_prediction,
        "action_status": action_status,
        "concern_score": concern_score,
        "score_label": _string(item.get("score_label"), "Concern score"),
        "score_available": bool(score_available),
        "evidence_bundle": evidence_bundle,
        "provenance": provenance,
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
        "raw_input_sha256": hashlib.sha256(
            _string(item.get("raw_input") or item.get("input_text") or item.get("text")).encode("utf-8")
        ).hexdigest(),
        "evidence_bundle": _string(item.get("evidence_bundle")),
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
    native_prediction: str = "",
    action_status: str = "",
    concern_score: float | None = None,
    score_label: str = "Concern score",
    score_available: bool | None = None,
    evidence_bundle: dict[str, object] | str | None = None,
    provenance: dict[str, object] | str | None = None,
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
        "native_prediction": native_prediction or prediction,
        "action_status": action_status,
        "concern_score": concern_score,
        "score_label": score_label,
        "score_available": score_available,
        "evidence_bundle": evidence_bundle or "",
        "provenance": provenance or "",
    }
    row = normalise_history_item(row, session_id=session_id)
    fingerprint = source_fingerprint or history_fingerprint(row)

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO scan_history (
                session_id, scanned_at, scan_type, source_name, prediction,
                confidence, model_name, preview, flags, explanation, raw_input,
                report_note, source_fingerprint, native_prediction, action_status,
                concern_score, score_label, score_available, evidence_bundle, provenance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                row["native_prediction"],
                row["action_status"],
                row["concern_score"],
                row["score_label"],
                1 if row["score_available"] else 0,
                row["evidence_bundle"],
                row["provenance"],
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
            native_prediction=str(row["native_prediction"]),
            action_status=str(row["action_status"]),
            concern_score=float(row["concern_score"]) if row["concern_score"] is not None else None,
            score_label=str(row["score_label"]),
            score_available=bool(row["score_available"]),
            evidence_bundle=str(row["evidence_bundle"]),
            provenance=str(row["provenance"]),
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
        native_prediction=str(row["native_prediction"]),
        action_status=str(row["action_status"]),
        concern_score=float(row["concern_score"]) if row["concern_score"] is not None else None,
        score_label=str(row["score_label"]),
        score_available=bool(row["score_available"]),
        evidence_bundle=str(row["evidence_bundle"]),
        provenance=str(row["provenance"]),
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
    action_statuses: list[str] | None = None,
    native_predictions: list[str] | None = None,
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
    if native_predictions:
        placeholders = ",".join("?" for _ in native_predictions)
        clauses.append(f"COALESCE(native_prediction, prediction) IN ({placeholders})")
        params.extend(native_predictions)

    sql = f"SELECT * FROM scan_history WHERE {' AND '.join(clauses)} ORDER BY scanned_at DESC, id DESC"
    if limit and not action_statuses:
        sql += " LIMIT ?"
        params.append(limit)

    with _connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    hydrated = [_hydrate_query_row(dict(row)) for row in rows]
    if action_statuses:
        allowed = set(action_statuses)
        hydrated = [row for row in hydrated if str(row.get("action_status")) in allowed]
    return hydrated[:limit] if limit else hydrated


def _hydrate_query_row(row: dict[str, object]) -> dict[str, object]:
    native_prediction = _string(row.get("native_prediction") or row.get("prediction"), "Unknown")
    row["native_prediction"] = native_prediction
    score_available = _explicit_bool(row.get("score_available"))
    concern_score = row.get("concern_score")
    evidence_complete = None
    if concern_score is None and "phone" in _string(row.get("scan_type")).casefold():
        concern_score, evidence_complete = _phone_snapshot_fields(row.get("raw_input"))
    if score_available is None:
        score_available = concern_score is not None
        if concern_score is None and "transcript" in _string(row.get("scan_type")).casefold():
            concern_score = _normalise_confidence(row.get("confidence"))
            score_available = True
    row["concern_score"] = concern_score
    row["score_available"] = bool(score_available)
    row["score_label"] = _string(row.get("score_label"), "Concern score")
    if not _string(row.get("action_status"), ""):
        row["action_status"] = derive_action_status(
            native_prediction=native_prediction,
            evidence_type=row.get("scan_type"),
            concern_score=concern_score,
            score_available=score_available,
            evidence_complete=evidence_complete,
        )
    return row


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
    report_sha256: str = "",
    manifest: dict[str, object] | None = None,
) -> int:
    """Record that a report export was generated."""

    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO report_exports (
                exported_at, session_id, format, scan_ids, filename, report_sha256, manifest
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_for_app().replace(microsecond=0).isoformat(),
                session_id,
                report_format.upper(),
                json.dumps(scan_ids, ensure_ascii=True),
                filename,
                report_sha256,
                json.dumps(manifest or {}, ensure_ascii=True, default=str),
            ),
        )
        return int(cursor.lastrowid)
