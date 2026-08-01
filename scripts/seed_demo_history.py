"""Sanitize and seed the report-history database for lecturer demos.

The seeded examples use negative internal row IDs so real user-created scans
can start at ID 1 after the database is reset. The report page displays seeded
rows as demo ID 0 while keeping the hidden internal key for export selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.evidence_snapshot import (  # noqa: E402
    IMMEDIATE_ACTION,
    NO_IMMEDIATE_ACTION,
    build_evidence_bundle,
    provenance_record,
    remediation_plan,
    table_artifact,
    text_artifact,
    xai_record,
)
from src.reporting.history_db import (  # noqa: E402
    DEFAULT_SESSION_ID,
    SCAN_HISTORY_ADDITIVE_COLUMNS,
    DB_PATH,
    init_db,
)


SEED_SCOPE = "seed_demo"
BANNED_PHONE_HASHES = (
    "50376e07fa500adb049a79e6d8c4359daf92dd25a9dc39d3dc912655be60bbf5",
    "0ecfc65e966372cc3e8d856fdb65af727d8d8ea6abe71263628495c01a6765db",
    "145af77fd97f15c058e7bfcd05f908936d9d502ca4b3d4a423a51ee0343b9174",
    "c32d585bbfe166859057607587b0d0643aa2b2babbb9e55c69ff7a9611bbb520",
    "a8ac92960383de7dca28b438d72c038aa078ba721953eed281886368e1003ff7",
    "1d8127eeddb1130c5f010a39875533b7889e8a39af22ab2a5d555eff34ee2841",
)
BANNED_PROVIDER_TERMS = ("omkar",)
PHONE_CANDIDATE_PATTERN = re.compile(r"(?<!\d)(?:\+?60|0)1\d[\s().-]?\d{3,4}[\s().-]?\d{4}(?!\d)")
SENSITIVE_SCAN_COLUMNS = (
    "source_name",
    "preview",
    "raw_input",
    "model_name",
    "evidence_bundle",
    "provenance",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


def _ensure_history_columns(connection: sqlite3.Connection) -> None:
    existing = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(scan_history)")
    }
    for column_name, column_type in SCAN_HISTORY_ADDITIVE_COLUMNS.items():
        if column_name not in existing:
            connection.execute(
                f"ALTER TABLE scan_history ADD COLUMN {column_name} {column_type}"
            )


def _backup_database(db_path: Path) -> Path | None:
    if not db_path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}_{stamp}.db.bak")
    shutil.copy2(db_path, backup_path)
    return backup_path


def _normalise_phone_candidate(value: str) -> str:
    return re.sub(r"\D", "", value)


def _phone_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_banned_phone(value: object) -> bool:
    text = str(value or "")
    for candidate in PHONE_CANDIDATE_PATTERN.findall(text):
        if _phone_hash(_normalise_phone_candidate(candidate)) in BANNED_PHONE_HASHES:
            return True
    return False


def _contains_banned_provider(value: object) -> bool:
    text = str(value or "").casefold()
    return any(term in text for term in BANNED_PROVIDER_TERMS)


def _count_targeted_sensitive_rows(connection: sqlite3.Connection) -> int:
    column_sql = ", ".join(SENSITIVE_SCAN_COLUMNS)
    count = 0
    for row in connection.execute(f"SELECT {column_sql} FROM scan_history"):
        if any(_contains_banned_phone(value) or _contains_banned_provider(value) for value in row):
            count += 1
    return count


def _email_seed(
    *,
    seed_id: int,
    sample_id: str,
    scanned_at: str,
    source_name: str,
    text: str,
    prediction: str,
    confidence: float,
    concern_score: float,
    action_status: str,
    flags: list[str],
    explanation: str,
) -> dict[str, object]:
    bundle = build_evidence_bundle(
        evidence_type="Email",
        captured_at=scanned_at,
        source_input={
            "sample_id": sample_id,
            "source_name": source_name,
            "demo_file": "data/demo/demo_emails.csv",
        },
        dashboard_summary={
            "final_verdict": prediction,
            "action_status": action_status,
            "expected_result": prediction,
            "expected_report_use": "Selectable evidence for TXT/PDF/DOCX report export",
        },
        findings=flags,
        artifacts=[
            text_artifact(
                "Demo email body",
                text,
                description="Upload-ready email text used for the seeded result.",
                source="data/demo/demo_emails.csv",
            ),
            table_artifact(
                "Expected output",
                [
                    {
                        "sample_id": sample_id,
                        "expected_prediction": prediction,
                        "expected_action_status": action_status,
                        "expected_concern_score": concern_score,
                    }
                ],
                description="Curated expected output for lecturer demo checks.",
                source="data/demo/demo_emails.csv",
            ),
        ],
        xai=xai_record(
            method="Curated keyword and model-output explanation",
            factors=[
                {"factor": flag, "effect": "Raises concern" if action_status == IMMEDIATE_ACTION else "Reduces concern"}
                for flag in flags
            ],
            explanation=explanation,
            limitations=[
                "Demo evidence is curated for repeatable dashboard walkthroughs.",
                "Predictions remain educational signals and require human review.",
            ],
        ),
        remediation=remediation_plan(
            evidence_type="Email",
            action_status=action_status,
            findings=flags,
        ),
    )
    return {
        "id": seed_id,
        "session_id": DEFAULT_SESSION_ID,
        "scanned_at": scanned_at,
        "scan_type": "Email",
        "source_name": source_name,
        "prediction": prediction,
        "confidence": confidence,
        "model_name": "Email TF-IDF Ensemble",
        "preview": " ".join(text.split())[:800],
        "flags": _json(flags),
        "explanation": explanation,
        "raw_input": text,
        "report_note": "Seed demo email evidence. Visible demo ID: 0.",
        "source_fingerprint": f"seed-demo-email-{sample_id.casefold()}",
        "native_prediction": prediction,
        "action_status": action_status,
        "concern_score": concern_score,
        "score_label": "Curated expected concern",
        "score_available": 1,
        "evidence_bundle": _json(bundle),
        "provenance": _json(
            provenance_record(
                text,
                source_name=source_name,
                source_kind="Email demo upload",
                captured_at=scanned_at,
                extra={"demo_scope": SEED_SCOPE, "demo_file": "data/demo/demo_emails.csv"},
            )
        ),
        "record_scope": SEED_SCOPE,
    }


def _transcript_seed(
    *,
    seed_id: int,
    sample_id: str,
    scanned_at: str,
    source_name: str,
    transcript: str,
    prediction: str,
    confidence: float,
    concern_score: float,
    action_status: str,
    flags: list[str],
    explanation: str,
) -> dict[str, object]:
    bundle = build_evidence_bundle(
        evidence_type="Transcript",
        captured_at=scanned_at,
        source_input={
            "sample_id": sample_id,
            "source_name": source_name,
            "demo_file": "data/demo/demo_transcripts.csv",
        },
        dashboard_summary={
            "final_verdict": prediction,
            "action_status": action_status,
            "expected_result": prediction,
            "expected_report_use": "Selectable evidence for TXT/PDF/DOCX report export",
        },
        findings=flags,
        artifacts=[
            text_artifact(
                "Demo call transcript",
                transcript,
                description="Upload-ready transcript text used for the seeded result.",
                source="data/demo/demo_transcripts.csv",
            ),
            table_artifact(
                "Expected output",
                [
                    {
                        "sample_id": sample_id,
                        "expected_prediction": prediction,
                        "expected_action_status": action_status,
                        "expected_concern_score": concern_score,
                    }
                ],
                description="Curated expected output for lecturer demo checks.",
                source="data/demo/demo_transcripts.csv",
            ),
        ],
        xai=xai_record(
            method="Curated transcript risk explanation",
            factors=[
                {"factor": flag, "effect": "Raises concern" if action_status == IMMEDIATE_ACTION else "Reduces concern"}
                for flag in flags
            ],
            explanation=explanation,
            limitations=[
                "Transcript examples are text fixtures; separate audio authenticity is not claimed.",
                "Predictions remain educational signals and require human review.",
            ],
        ),
        remediation=remediation_plan(
            evidence_type="Transcript",
            action_status=action_status,
            findings=flags,
        ),
    )
    return {
        "id": seed_id,
        "session_id": DEFAULT_SESSION_ID,
        "scanned_at": scanned_at,
        "scan_type": "Transcript",
        "source_name": source_name,
        "prediction": prediction,
        "confidence": confidence,
        "model_name": "Transcript DistilBERT/TF-IDF Ensemble",
        "preview": " ".join(transcript.split())[:800],
        "flags": _json(flags),
        "explanation": explanation,
        "raw_input": transcript,
        "report_note": "Seed demo transcript evidence. Visible demo ID: 0.",
        "source_fingerprint": f"seed-demo-transcript-{sample_id.casefold()}",
        "native_prediction": prediction,
        "action_status": action_status,
        "concern_score": concern_score,
        "score_label": "Curated expected concern",
        "score_available": 1,
        "evidence_bundle": _json(bundle),
        "provenance": _json(
            provenance_record(
                transcript,
                source_name=source_name,
                source_kind="Transcript demo upload",
                captured_at=scanned_at,
                extra={"demo_scope": SEED_SCOPE, "demo_file": "data/demo/demo_transcripts.csv"},
            )
        ),
        "record_scope": SEED_SCOPE,
    }


def _phone_seed(
    *,
    seed_id: int,
    sample_id: str,
    scanned_at: str,
    phone_number: str,
    prediction: str,
    confidence: float,
    concern_score: float,
    action_status: str,
    flags: list[str],
    explanation: str,
    reports: int,
    tag: str,
) -> dict[str, object]:
    raw_input = {
        "phone": phone_number,
        "demo_notice": "Reserved non-personal phone number for capstone demonstration.",
        "assessment": {
            "label": prediction,
            "score": concern_score,
            "reports": reports,
            "tag": tag,
        },
        "coverage": {
            "veriphone": "demo_seed",
            "penipumy": "demo_seed",
        },
        "providers": [
            {
                "name": "Reserved demo fixture",
                "status": "curated",
                "detail": "No live API lookup or real caller identity is claimed.",
            }
        ],
    }
    bundle = build_evidence_bundle(
        evidence_type="Phone",
        captured_at=scanned_at,
        source_input={
            "sample_id": sample_id,
            "source_name": phone_number,
            "demo_file": "data/demo/demo_phone_numbers.csv",
            "number_type": "reserved drama/demo number",
        },
        dashboard_summary={
            "final_verdict": prediction,
            "action_status": action_status,
            "expected_result": prediction,
            "expected_report_use": "Seeded report evidence only; not a live provider lookup.",
        },
        findings=flags,
        artifacts=[
            table_artifact(
                "Phone demo reputation fixture",
                [
                    {
                        "sample_id": sample_id,
                        "phone_number": phone_number,
                        "reports": reports,
                        "tag": tag,
                        "expected_prediction": prediction,
                        "expected_action_status": action_status,
                        "expected_concern_score": concern_score,
                    }
                ],
                description="Reserved-number fixture for repeatable lecturer report generation.",
                source="data/demo/demo_phone_numbers.csv",
            )
        ],
        xai=xai_record(
            method="Curated phone reputation fixture explanation",
            factors=[
                {"factor": flag, "effect": "Raises concern" if action_status == IMMEDIATE_ACTION else "Reduces concern"}
                for flag in flags
            ],
            explanation=explanation,
            limitations=[
                "Reserved demo numbers do not represent real phone owners.",
                "Live provider output may differ if the number is manually searched online.",
            ],
        ),
        remediation=remediation_plan(
            evidence_type="Phone",
            action_status=action_status,
            findings=flags,
        ),
    )
    return {
        "id": seed_id,
        "session_id": DEFAULT_SESSION_ID,
        "scanned_at": scanned_at,
        "scan_type": "Phone",
        "source_name": phone_number,
        "prediction": prediction,
        "confidence": confidence,
        "model_name": "Curated Phone Demo Fixture",
        "preview": f"{phone_number} | {tag}",
        "flags": _json(flags),
        "explanation": explanation,
        "raw_input": _json(raw_input),
        "report_note": "Seed demo phone evidence. Visible demo ID: 0.",
        "source_fingerprint": f"seed-demo-phone-{sample_id.casefold()}",
        "native_prediction": prediction,
        "action_status": action_status,
        "concern_score": concern_score,
        "score_label": "Curated expected concern",
        "score_available": 1,
        "evidence_bundle": _json(bundle),
        "provenance": _json(
            provenance_record(
                raw_input,
                source_name=phone_number,
                source_kind="Phone demo fixture",
                captured_at=scanned_at,
                extra={
                    "demo_scope": SEED_SCOPE,
                    "demo_file": "data/demo/demo_phone_numbers.csv",
                    "provider_coverage": "Curated fixture only; no live provider key or API result stored.",
                },
            )
        ),
        "record_scope": SEED_SCOPE,
    }


def _seed_rows() -> list[dict[str, object]]:
    return [
        _email_seed(
            seed_id=-1,
            sample_id="DEMO-EMAIL-SCAM-001",
            scanned_at="2026-08-01T09:00:00",
            source_name="demo_email_urgent_mailbox.txt",
            text=(
                "From: University IT Helpdesk <notice@example.edu>\n"
                "Subject: Action required: mailbox suspension\n\n"
                "Your student mailbox will be suspended in two hours. Confirm your password and OTP at "
                "https://student-mail-verify.example.invalid to keep access."
            ),
            prediction="Suspicious",
            confidence=93.0,
            concern_score=88.0,
            action_status=IMMEDIATE_ACTION,
            flags=["urgent deadline", "password request", "OTP request", "untrusted verification link"],
            explanation="Urgency, credential collection, OTP request, and a non-official verification URL align with phishing indicators.",
        ),
        _email_seed(
            seed_id=-2,
            sample_id="DEMO-EMAIL-LEGIT-001",
            scanned_at="2026-08-01T09:01:00",
            source_name="demo_email_library_notice.txt",
            text=(
                "From: Library Services <library@example.edu>\n"
                "Subject: Requested book is ready\n\n"
                "The book you requested is ready for collection at the circulation desk. Please bring your student card."
            ),
            prediction="Legitimate",
            confidence=89.0,
            concern_score=10.0,
            action_status=NO_IMMEDIATE_ACTION,
            flags=["routine campus notice", "no payment request", "no credential request"],
            explanation="The message is routine and does not ask for money, OTPs, passwords, secrecy, or off-platform verification.",
        ),
        _email_seed(
            seed_id=-3,
            sample_id="DEMO-EMAIL-SCAM-002",
            scanned_at="2026-08-01T09:02:00",
            source_name="demo_email_scholarship_fee.txt",
            text=(
                "Subject: Scholarship selected notice\n\n"
                "Congratulations, you were selected for a student grant. Pay the RM150 processing fee today by instant transfer "
                "or your award will be cancelled."
            ),
            prediction="Suspicious",
            confidence=91.0,
            concern_score=84.0,
            action_status=IMMEDIATE_ACTION,
            flags=["unexpected reward", "processing fee", "same-day pressure", "bank transfer request"],
            explanation="Unexpected scholarship reward plus urgent processing-fee pressure is a common student-targeted scam pattern.",
        ),
        _transcript_seed(
            seed_id=-4,
            sample_id="DEMO-TRANSCRIPT-SCAM-001",
            scanned_at="2026-08-01T09:03:00",
            source_name="demo_transcript_bank_otp.txt",
            transcript=(
                "Caller: I am from the bank fraud team. Your account is frozen. Do not end this call. "
                "Read me the TAC and OTP now so I can cancel the suspicious transfer."
            ),
            prediction="Scam",
            confidence=94.0,
            concern_score=92.0,
            action_status=IMMEDIATE_ACTION,
            flags=["bank impersonation", "OTP request", "pressure to stay on call", "transaction threat"],
            explanation="The caller impersonates a bank and requests TAC/OTP codes under immediate pressure.",
        ),
        _transcript_seed(
            seed_id=-5,
            sample_id="DEMO-TRANSCRIPT-LEGIT-001",
            scanned_at="2026-08-01T09:04:00",
            source_name="demo_transcript_advisor_portal.txt",
            transcript=(
                "Advisor: Your course registration form is due Friday. Please submit it through the official student portal. "
                "Email me if you have questions about the module list."
            ),
            prediction="Non-scam",
            confidence=87.0,
            concern_score=14.0,
            action_status=NO_IMMEDIATE_ACTION,
            flags=["official portal", "routine academic process", "no secrecy request"],
            explanation="The call points to an official portal and contains no money, credential, secrecy, or threat indicators.",
        ),
        _transcript_seed(
            seed_id=-6,
            sample_id="DEMO-TRANSCRIPT-SCAM-002",
            scanned_at="2026-08-01T09:05:00",
            source_name="demo_transcript_police_case.txt",
            transcript=(
                "Caller: Your parcel is connected to a police case. Keep this confidential. "
                "Transfer RM500 to the secure account today so we can clear your name."
            ),
            prediction="Scam",
            confidence=92.0,
            concern_score=90.0,
            action_status=IMMEDIATE_ACTION,
            flags=["authority impersonation", "secrecy request", "money transfer pressure", "legal threat"],
            explanation="Authority impersonation, secrecy, and urgent transfer instructions indicate high-risk scam behavior.",
        ),
        _phone_seed(
            seed_id=-7,
            sample_id="DEMO-PHONE-SCAM-001",
            scanned_at="2026-08-01T09:06:00",
            phone_number="+447700900101",
            prediction="High concern",
            confidence=86.0,
            concern_score=86.0,
            action_status=IMMEDIATE_ACTION,
            flags=["reserved demo number", "bank impersonation tag", "31 curated reports"],
            explanation="The seeded fixture represents a high-concern phone reputation scenario without storing a real person's number.",
            reports=31,
            tag="bank impersonation",
        ),
        _phone_seed(
            seed_id=-8,
            sample_id="DEMO-PHONE-LEGIT-001",
            scanned_at="2026-08-01T09:07:00",
            phone_number="+447700900102",
            prediction="Lower concern",
            confidence=82.0,
            concern_score=8.0,
            action_status=NO_IMMEDIATE_ACTION,
            flags=["reserved demo number", "no curated scam reports", "routine campus contact"],
            explanation="The seeded fixture represents a lower-concern phone scenario and does not identify a real caller.",
            reports=0,
            tag="campus office",
        ),
    ]


def _replace_history(connection: sqlite3.Connection, rows: list[dict[str, object]]) -> tuple[int, int]:
    existing_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM scan_history WHERE session_id = ?",
            (DEFAULT_SESSION_ID,),
        ).fetchone()[0]
    )
    connection.execute("DELETE FROM scan_history WHERE session_id = ?", (DEFAULT_SESSION_ID,))
    connection.execute("DELETE FROM report_exports WHERE session_id = ?", (DEFAULT_SESSION_ID,))

    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    for row in rows:
        connection.execute(
            f"INSERT INTO scan_history ({column_sql}) VALUES ({placeholders})",
            [row[column] for column in columns],
        )

    positive_max = connection.execute(
        "SELECT MAX(id) FROM scan_history WHERE id > 0"
    ).fetchone()[0]
    if positive_max is None:
        connection.execute("DELETE FROM sqlite_sequence WHERE name = 'scan_history'")
    else:
        cursor = connection.execute(
            "UPDATE sqlite_sequence SET seq = ? WHERE name = 'scan_history'",
            (int(positive_max),),
        )
        if cursor.rowcount == 0:
            connection.execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES ('scan_history', ?)",
                (int(positive_max),),
            )
    connection.execute("DELETE FROM sqlite_sequence WHERE name = 'report_exports'")
    return existing_count, len(rows)


def seed_demo_history(*, dry_run: bool = False, backup: bool = True) -> dict[str, object]:
    init_db()
    rows = _seed_rows()
    backup_path = _backup_database(DB_PATH) if backup and not dry_run else None
    with sqlite3.connect(DB_PATH) as connection:
        _ensure_history_columns(connection)
        sensitive_count = _count_targeted_sensitive_rows(connection)
        existing_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM scan_history WHERE session_id = ?",
                (DEFAULT_SESSION_ID,),
            ).fetchone()[0]
        )
        if dry_run:
            return {
                "dry_run": True,
                "database": str(DB_PATH),
                "existing_default_session_rows": existing_count,
                "targeted_sensitive_or_outdated_rows": sensitive_count,
                "seed_rows_to_insert": len(rows),
                "backup_path": str(backup_path) if backup_path else "",
            }

        replaced_count, inserted_count = _replace_history(connection, rows)
        connection.commit()

    return {
        "dry_run": False,
        "database": str(DB_PATH),
        "backup_path": str(backup_path) if backup_path else "",
        "replaced_default_session_rows": replaced_count,
        "targeted_sensitive_or_outdated_rows_before_reset": sensitive_count,
        "seed_rows_inserted": inserted_count,
        "visible_demo_id": 0,
        "next_user_scan_id": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show planned DB changes without writing.")
    parser.add_argument("--no-backup", action="store_true", help="Skip the timestamped .db.bak copy.")
    args = parser.parse_args()
    result = seed_demo_history(dry_run=args.dry_run, backup=not args.no_backup)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
