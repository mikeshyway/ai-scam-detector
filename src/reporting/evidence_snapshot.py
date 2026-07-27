"""Shared evidence snapshots, action status normalization, and remediation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.utils.time_utils import now_for_app


SNAPSHOT_SCHEMA_VERSION = "1.0"

IMMEDIATE_ACTION = "Immediate Action Required"
REVIEW_REQUIRED = "Review Required"
NO_IMMEDIATE_ACTION = "No Immediate Action"
ACTION_STATUSES = [IMMEDIATE_ACTION, REVIEW_REQUIRED, NO_IMMEDIATE_ACTION]


def json_safe(value: object) -> object:
    """Convert common scientific/UI values to JSON-compatible Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict("records"))
        except TypeError:
            try:
                return json_safe(to_dict())
            except Exception:
                pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(tolist())
        except Exception:
            pass
    return str(value)


def canonical_json(value: object) -> str:
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_text(value: object) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 0 < number <= 1:
        number *= 100
    return max(0.0, min(100.0, number))


def _agreement_ratio(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ratio = float(value)
        return ratio if 0 <= ratio <= 1 else max(0.0, min(1.0, ratio / 100.0))
    text = str(value).strip()
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            numerator = float(left)
            denominator = float(right)
            return numerator / denominator if denominator > 0 else None
        except ValueError:
            return None
    try:
        return _agreement_ratio(float(text.replace("%", "")))
    except ValueError:
        return None


def derive_action_status(
    *,
    native_prediction: object,
    evidence_type: object = "",
    concern_score: object = None,
    score_available: bool | None = None,
    model_agreement: object = None,
    evidence_complete: bool | None = None,
    direct_exposure: bool = False,
) -> str:
    """Map source-native results to one of three user-action labels.

    The source verdict is never replaced. This is a separate triage label, and
    unavailable or incomplete evidence is intentionally routed to review.
    """

    prediction = str(native_prediction or "Unknown").strip().casefold()
    family = str(evidence_type or "").strip().casefold()
    score = _number(concern_score)
    available = bool(score is not None) if score_available is None else bool(score_available)
    agreement = _agreement_ratio(model_agreement)

    if direct_exposure:
        return IMMEDIATE_ACTION
    if any(term in prediction for term in ("critical review", "high concern", "high risk", "confirmed scam")):
        return IMMEDIATE_ACTION
    if any(term in prediction for term in ("unknown", "unavailable", "not found", "insufficient")):
        return REVIEW_REQUIRED
    if evidence_complete is False:
        return REVIEW_REQUIRED
    if any(term in prediction for term in ("medium risk", "needs verification", "needs review")):
        return REVIEW_REQUIRED

    suspicious = any(
        term in prediction
        for term in ("suspicious", "phishing", "scam", "ai-generated", "deepfake")
    )
    if suspicious:
        if available and score is not None and score >= 70 and (agreement is None or agreement >= 2 / 3):
            return IMMEDIATE_ACTION
        return REVIEW_REQUIRED

    lower = any(
        term in prediction
        for term in ("legitimate", "lower concern", "lower risk", "low risk", "real human", "benign", "safe")
    )
    if lower:
        if "phone" in family and evidence_complete is not True:
            return REVIEW_REQUIRED
        return NO_IMMEDIATE_ACTION

    # Old process labels (for example "uploaded recording chunk analysis") are
    # not threat findings and must not be promoted by keyword matching.
    return REVIEW_REQUIRED


def provenance_record(
    raw_input: object,
    *,
    source_name: str = "",
    source_kind: str = "application input",
    captured_at: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    captured = captured_at or now_for_app().replace(microsecond=0).isoformat()
    record: dict[str, object] = {
        "captured_at": captured,
        "source_name": source_name or "Unspecified source",
        "source_kind": source_kind,
        "sha256": sha256_text(raw_input),
        "hash_scope": (
            "UTF-8 input text"
            if isinstance(raw_input, str)
            else "Raw binary input bytes"
            if isinstance(raw_input, bytes)
            else "Canonical JSON evidence object"
        ),
        "integrity_note": "The digest identifies the input captured when the investigation was saved.",
    }
    if extra:
        record.update(dict(json_safe(extra)))
    return record


def table_artifact(
    title: str,
    data: object,
    *,
    description: str = "",
    source: str = "Dashboard",
) -> dict[str, object]:
    rows = json_safe(data)
    if not isinstance(rows, list):
        rows = [rows]
    columns: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            for key in row:
                if key not in columns:
                    columns.append(key)
    return {
        "kind": "table",
        "title": title,
        "description": description,
        "source": source,
        "columns": columns,
        "rows": rows,
    }


def chart_artifact(
    title: str,
    figure: object | None,
    *,
    description: str = "",
    data: object | None = None,
    source: str = "Dashboard",
) -> dict[str, object]:
    figure_json = ""
    if figure is not None:
        to_json = getattr(figure, "to_json", None)
        if callable(to_json):
            try:
                figure_json = to_json()
            except Exception:
                figure_json = ""
    return {
        "kind": "chart",
        "title": title,
        "description": description,
        "source": source,
        "figure_json": figure_json,
        "figure_sha256": sha256_text(figure_json) if figure_json else "",
        "data": json_safe(data) if data is not None else None,
    }


def text_artifact(
    title: str,
    body: object,
    *,
    description: str = "",
    source: str = "Dashboard",
) -> dict[str, object]:
    return {
        "kind": "text",
        "title": title,
        "description": description,
        "source": source,
        "body": str(body or ""),
    }


def xai_record(
    *,
    method: str,
    factors: Iterable[dict[str, object]] | None = None,
    explanation: str = "",
    scope: str = "local",
    limitations: Iterable[str] | None = None,
) -> dict[str, object]:
    return {
        "method": method,
        "scope": scope,
        "explanation": explanation,
        "factors": list(json_safe(list(factors or []))),
        "limitations": [str(item) for item in (limitations or []) if str(item).strip()],
    }


def remediation_plan(
    *,
    evidence_type: str,
    action_status: str,
    findings: Iterable[object] | None = None,
    incident_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """Create trigger-based, educational actions without inventing exposure."""

    family = evidence_type.strip().casefold()
    finding_text = " ".join(str(item) for item in (findings or [])).casefold()
    context = {str(key): bool(value) for key, value in (incident_context or {}).items()}
    items: list[dict[str, str]] = []

    def add(priority: str, trigger: str, action: str, reason: str) -> None:
        candidate = {"priority": priority, "trigger": trigger, "action": action, "reason": reason}
        if candidate not in items:
            items.append(candidate)

    if action_status == IMMEDIATE_ACTION:
        add(
            "Immediate",
            "Evidence requires immediate attention",
            "Pause contact and do not send money, credentials, OTPs, files, or remote access.",
            "Stopping the interaction limits further exposure while the evidence is verified.",
        )
    elif action_status == REVIEW_REQUIRED:
        add(
            "Prompt review",
            "Evidence is suspicious, incomplete, unknown, or conflicting",
            "Verify the person or organization through a separately obtained official channel.",
            "An independent channel avoids relying on details supplied by the same contact.",
        )
    else:
        add(
            "Monitor",
            "No immediate action is indicated by the available evidence",
            "Keep normal caution and re-check any later request involving money, identity, credentials, or urgency.",
            "A lower-concern result is not a guarantee that future contact is safe.",
        )

    if "email" in family:
        add(
            "Preserve",
            "Email or message evidence",
            "Retain the original message, complete headers, attachments, URLs, and timestamps.",
            "Original metadata supports later sender, routing, and domain investigation.",
        )
    if "transcript" in family or "audio" in family:
        add(
            "Preserve",
            "Call, transcript, or voice evidence",
            "Retain the earliest available recording, transcript, timestamps, and unedited working copy.",
            "Voice familiarity and detector scores alone cannot establish speaker identity or authenticity.",
        )
    if "phone" in family:
        add(
            "Verify",
            "Phone carrier or reputation lookup",
            "Call the organization using a number from its official website or trusted statement.",
            "Carrier metadata and absence of community reports do not verify the current caller.",
        )

    if any(term in finding_text for term in ("otp", "password", "credential", "login")):
        add(
            "Immediate",
            "Credential or OTP request detected",
            "Do not disclose the code or password; change exposed credentials and contact the account provider.",
            "OTPs and passwords can enable account takeover.",
        )
    if any(term in finding_text for term in ("payment", "bank", "transfer", "crypto", "gift card")):
        add(
            "Immediate",
            "Payment pressure detected",
            "Stop payment and contact the bank or payment provider through its official fraud channel.",
            "Fast reporting may improve the chance of limiting or tracing a transfer.",
        )
    if any(term in finding_text for term in ("remote access", "install software", "anydesk", "teamviewer")):
        add(
            "Immediate",
            "Remote-access request detected",
            "Disconnect the device from the network, remove unauthorized remote-access tools, and seek technical review.",
            "Remote access can expose accounts, files, and active sessions.",
        )

    exposure_actions = {
        "link_clicked": (
            "A suspicious link was opened",
            "Close the page, avoid entering information, scan the device, and review account sessions.",
            "A visited site may capture credentials or attempt malware delivery.",
        ),
        "credentials_shared": (
            "Credentials were shared",
            "Change the password from a trusted device, revoke sessions, and enable multi-factor authentication.",
            "Shared credentials should be treated as compromised.",
        ),
        "otp_shared": (
            "An OTP or verification code was shared",
            "Contact the affected provider immediately and review account and transaction activity.",
            "A one-time code may authorize account access or a transaction.",
        ),
        "software_installed": (
            "Software or remote access was installed",
            "Disconnect the device, preserve relevant logs, and obtain a security review before reuse.",
            "Installed software may provide persistent access.",
        ),
        "funds_transferred": (
            "Funds were transferred",
            "Contact the bank or payment provider immediately, preserve receipts, and file an official report.",
            "Immediate reporting may help recovery and creates an investigation record.",
        ),
        "interaction_ongoing": (
            "The contact is still ongoing",
            "Pause the interaction and verify the person or organization through an independent official channel.",
            "Continuing the same channel can create further pressure or exposure before identity is verified.",
        ),
    }
    for key, (trigger, action, reason) in exposure_actions.items():
        if context.get(key):
            add("Immediate", trigger, action, reason)

    return items


def build_evidence_bundle(
    *,
    evidence_type: str,
    source_input: dict[str, object],
    dashboard_summary: dict[str, object],
    artifacts: Iterable[dict[str, object]] | None = None,
    findings: Iterable[object] | None = None,
    xai: dict[str, object] | None = None,
    limitations: Iterable[str] | None = None,
    remediation: Iterable[dict[str, str]] | None = None,
    captured_at: str | None = None,
) -> dict[str, object]:
    captured = captured_at or now_for_app().replace(microsecond=0).isoformat()
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "captured_at": captured,
        "evidence_type": evidence_type,
        "source_input": dict(json_safe(source_input)),
        "dashboard_summary": dict(json_safe(dashboard_summary)),
        "findings": list(json_safe(list(findings or []))),
        "xai": dict(json_safe(xai or {})),
        "artifacts": list(json_safe(list(artifacts or []))),
        "limitations": [str(item) for item in (limitations or []) if str(item).strip()],
        "remediation": list(json_safe(list(remediation or []))),
    }


def decode_json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}
