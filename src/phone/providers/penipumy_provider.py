"""PenipuMY reputation provider adapter."""

from __future__ import annotations

import time
from typing import Any

from src.phone.penipumy_client import PenipuApiError, PenipuClientError, fetch_phone_reputation
from src.phone.phone_number import format_phone_for_penipumy, normalise_phone_query, validate_phone_query
from src.phone.providers.models import (
    PhoneProviderResult,
    ProviderDiagnosticResult,
    authentication_status,
    normalize_error_code,
    provider_reachable,
    request_accepted,
    safe_rate_limit_text,
    top_level_field_counts,
)


PROVIDER_ID = "penipumy"
PROVIDER_NAME = "PenipuMY"


def _request_id(payload: dict[str, Any]) -> str:
    return str(payload.get("request_id") or payload.get("id") or payload.get("uuid") or "")


def _status_from_error(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


def test_penipumy_connection(
    test_number: str,
    api_key: str,
    *,
    key_source: str,
    key_variable: str,
    timeout: float = 10.0,
) -> ProviderDiagnosticResult:
    api_key = str(api_key or "").strip()
    configured = bool(api_key)
    started = time.perf_counter()

    if not configured:
        return ProviderDiagnosticResult(
            provider_id=PROVIDER_ID,
            provider_name=PROVIDER_NAME,
            configured=False,
            key_detected=False,
            key_source="Not configured",
            key_variable="-",
            reachable=False,
            authentication_status="No",
            request_accepted="Not evaluated",
            http_status=None,
            provider_success=None,
            response_time_ms=0.0,
            fields_returned=0,
            fields_populated=0,
            rate_limit="Not returned",
            fallback_used=False,
            request_id="",
            error_code="missing_key",
            error_message="PenipuMY API key is not configured.",
        )

    ok, message = validate_phone_query(test_number)
    if not ok:
        return ProviderDiagnosticResult(
            provider_id=PROVIDER_ID,
            provider_name=PROVIDER_NAME,
            configured=True,
            key_detected=True,
            key_source=key_source,
            key_variable=key_variable,
            reachable=False,
            authentication_status="Not evaluated",
            request_accepted="Not evaluated",
            http_status=None,
            provider_success=None,
            response_time_ms=0.0,
            fields_returned=0,
            fields_populated=0,
            rate_limit="Not returned",
            fallback_used=False,
            request_id="",
            error_code="invalid_number",
            error_message=message,
        )

    try:
        live = fetch_phone_reputation(
            format_phone_for_penipumy(test_number),
            api_key,
            timeout_seconds=int(timeout),
        )
    except (PenipuApiError, PenipuClientError) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        status_code = _status_from_error(exc)
        error_message = str(exc)
        error_code = normalize_error_code(status_code, error_message)
        return ProviderDiagnosticResult(
            provider_id=PROVIDER_ID,
            provider_name=PROVIDER_NAME,
            configured=True,
            key_detected=True,
            key_source=key_source,
            key_variable=key_variable,
            reachable=provider_reachable(True, status_code, error_code),
            authentication_status=authentication_status(True, status_code, error_code),
            request_accepted=request_accepted(status_code, False, error_code),
            http_status=status_code,
            provider_success=False if status_code is not None else None,
            response_time_ms=elapsed_ms,
            fields_returned=0,
            fields_populated=0,
            rate_limit=safe_rate_limit_text({}, status_code=status_code),
            fallback_used=False,
            request_id="",
            error_code=error_code,
            error_message=error_message,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = dict(live.data)
    fields_returned, fields_populated = top_level_field_counts(payload)

    return ProviderDiagnosticResult(
        provider_id=PROVIDER_ID,
        provider_name=PROVIDER_NAME,
        configured=True,
        key_detected=True,
        key_source=key_source,
        key_variable=key_variable,
        reachable=True,
        authentication_status="Not rejected",
        request_accepted="Yes",
        http_status=live.status_code,
        provider_success=True,
        response_time_ms=elapsed_ms,
        fields_returned=fields_returned,
        fields_populated=fields_populated,
        rate_limit=safe_rate_limit_text(live.rate_limit, status_code=live.status_code),
        fallback_used=False,
        request_id=_request_id(payload),
        error_code="none",
        error_message="",
        raw_field_names=list(payload.keys()),
    )


def lookup_penipumy_reputation(number: str, api_key: str, *, timeout: float = 10.0) -> PhoneProviderResult:
    normalized = normalise_phone_query(number)
    api_key = str(api_key or "").strip()

    if not api_key:
        return PhoneProviderResult(
            provider_id=PROVIDER_ID,
            provider_name=PROVIDER_NAME,
            channel="reputation",
            status="not_configured",
            success=False,
            normalized_number=normalized,
            error_code="missing_key",
            error_message="PenipuMY API key is not configured.",
        )

    started = time.perf_counter()
    try:
        live = fetch_phone_reputation(
            format_phone_for_penipumy(normalized),
            api_key,
            timeout_seconds=int(timeout),
        )
    except (PenipuApiError, PenipuClientError) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        status_code = _status_from_error(exc)
        return PhoneProviderResult(
            provider_id=PROVIDER_ID,
            provider_name=PROVIDER_NAME,
            channel="reputation",
            status="error",
            success=False,
            normalized_number=normalized,
            error_code=normalize_error_code(status_code, str(exc)),
            error_message=str(exc),
            response_time_ms=elapsed_ms,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = dict(live.data)
    status = "success" if payload else "no_match"
    return PhoneProviderResult(
        provider_id=PROVIDER_ID,
        provider_name=PROVIDER_NAME,
        channel="reputation",
        status=status,
        success=True,
        normalized_number=normalized,
        data=payload,
        response_time_ms=elapsed_ms,
        rate_limit=live.rate_limit,
        request_id=_request_id(payload) or None,
    )
