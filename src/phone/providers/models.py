"""Shared provider result models for phone-number investigations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def is_populated(value: object) -> bool:
    return value not in (None, "", [], {}, "N/A", "Not returned")


def top_level_field_counts(payload: dict[str, Any]) -> tuple[int, int]:
    returned = len(payload)
    populated = sum(1 for value in payload.values() if is_populated(value))
    return returned, populated


def safe_rate_limit_text(rate_limit: dict[str, Any] | None, *, status_code: int | None = None) -> str:
    if status_code == 429:
        return "Reached"

    values = [
        f"{key}={value}"
        for key, value in dict(rate_limit or {}).items()
        if is_populated(value)
    ]
    return ", ".join(values) if values else "Not returned"


def normalize_error_code(status_code: int | None, error_message: object) -> str:
    text = str(error_message or "").strip().lower()

    if "missing" in text or "not configured" in text:
        return "missing_key"
    if status_code in {401, 403} or "rejected" in text or "invalid key" in text:
        return "authentication_failed"
    if status_code == 429 or "rate limit" in text:
        return "rate_limited"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "network" in text or "connection" in text:
        return "connection_failed"
    if "phone number" in text or "invalid number" in text:
        return "invalid_number"
    if "json" in text or "response format" in text:
        return "invalid_response"
    if status_code in {500, 502, 503}:
        return "provider_error"
    if text:
        return "provider_error"
    return "none"


def authentication_status(configured: bool, status_code: int | None, error_code: str) -> str:
    if not configured:
        return "No"
    if error_code == "authentication_failed" or status_code in {401, 403}:
        return "Rejected"
    if status_code is None:
        return "Not evaluated"
    return "Not rejected"


def provider_reachable(configured: bool, status_code: int | None, error_code: str) -> bool:
    if not configured:
        return False
    if error_code in {"timeout", "connection_failed"}:
        return False
    return status_code is not None


def request_accepted(status_code: int | None, success: bool, error_code: str) -> str:
    if status_code is None:
        return "Not evaluated"
    if success:
        return "Yes"
    return "No" if error_code != "none" else "Not returned"


@dataclass
class ProviderDiagnosticResult:
    provider_id: str
    provider_name: str
    configured: bool
    key_detected: bool
    key_source: str
    key_variable: str
    reachable: bool
    authentication_status: str
    request_accepted: str
    http_status: int | None
    provider_success: bool | None
    response_time_ms: float
    fields_returned: int
    fields_populated: int
    rate_limit: str
    fallback_used: bool
    request_id: str
    error_code: str
    error_message: str
    raw_field_names: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=utc_timestamp)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhoneProviderResult:
    provider_id: str
    provider_name: str
    channel: str
    status: str
    success: bool
    normalized_number: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    response_time_ms: float = 0.0
    rate_limit: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    checked_at: str = field(default_factory=utc_timestamp)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnostic_rows(diagnostic: ProviderDiagnosticResult | dict[str, Any]) -> list[dict[str, str]]:
    data = diagnostic.as_dict() if isinstance(diagnostic, ProviderDiagnosticResult) else dict(diagnostic)

    http_status = data.get("http_status")
    live_detail = data.get("provider_name", "")
    return [
        {"Check": "Live provider", "Result": str(data.get("provider_name", "")), "Detail": str(live_detail)},
        {
            "Check": "API key detected",
            "Result": "Yes" if data.get("key_detected") else "No",
            "Detail": str(data.get("key_source") or "Not configured"),
        },
        {
            "Check": "Key variable",
            "Result": str(data.get("key_variable") or "-"),
            "Detail": "Value hidden",
        },
        {
            "Check": "Provider reachable",
            "Result": "Yes" if data.get("reachable") else "No",
            "Detail": f"Status {http_status if http_status is not None else 'N/A'}",
        },
        {
            "Check": "Authentication",
            "Result": str(data.get("authentication_status") or "Not evaluated"),
            "Detail": (
                "No 401/403 authentication error returned"
                if data.get("authentication_status") == "Not rejected"
                else str(data.get("error_code") or "-")
            ),
        },
        {
            "Check": "Request accepted",
            "Result": str(data.get("request_accepted") or "Not evaluated"),
            "Detail": str(data.get("error_code") or "-"),
        },
        {
            "Check": "HTTP status",
            "Result": str(http_status if http_status is not None else "N/A"),
            "Detail": "Live request" if float(data.get("response_time_ms") or 0) else "Not tested",
        },
        {
            "Check": "Provider success",
            "Result": (
                "Yes"
                if data.get("provider_success") is True
                else "No"
                if data.get("provider_success") is False
                else "Not returned"
            ),
            "Detail": "Provider response interpretation",
        },
        {
            "Check": "Response time",
            "Result": f"{float(data.get('response_time_ms') or 0):.0f} ms",
            "Detail": "Live request",
        },
        {
            "Check": "Fields returned",
            "Result": str(data.get("fields_returned", 0)),
            "Detail": "Top-level response fields",
        },
        {
            "Check": "Fields populated",
            "Result": str(data.get("fields_populated", 0)),
            "Detail": "Non-empty top-level fields",
        },
        {
            "Check": "Rate limit",
            "Result": str(data.get("rate_limit") or "Not returned"),
            "Detail": "Provider headers",
        },
        {
            "Check": "Fallback used",
            "Result": "Yes" if data.get("fallback_used") else "No",
            "Detail": "Connection check only",
        },
        {
            "Check": "Request ID",
            "Result": str(data.get("request_id") or "Not returned"),
            "Detail": "-",
        },
        {
            "Check": "Error",
            "Result": str(data.get("error_code") or "none"),
            "Detail": str(data.get("error_message") or "-"),
        },
    ]
