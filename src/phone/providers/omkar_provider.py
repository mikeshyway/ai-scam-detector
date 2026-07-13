"""Omkar Carrier Lookup provider adapter."""

from __future__ import annotations

import time
from typing import Any

from src.phone.omkar_client import lookup_omkar_phone
from src.phone.phone_number import format_phone_for_omkar, normalise_phone_query, validate_phone_query
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


PROVIDER_ID = "omkar_carrier_lookup"
PROVIDER_NAME = "Omkar Carrier Lookup"


def _request_id(payload: dict[str, Any]) -> str:
    return str(payload.get("request_id") or payload.get("id") or "")


def test_omkar_connection(
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
            error_message="Omkar Carrier Lookup API key is not configured.",
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

    live = lookup_omkar_phone(format_phone_for_omkar(test_number), api_key, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = dict(live.get("record", {})) if isinstance(live.get("record"), dict) else {}
    fields_returned, fields_populated = top_level_field_counts(payload)
    status_code = live.get("status_code")
    status_code = int(status_code) if isinstance(status_code, int) else status_code
    error_message = str(live.get("error") or "")
    success = bool(live.get("ok"))
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
        request_accepted=request_accepted(status_code, success, error_code),
        http_status=status_code,
        provider_success=success,
        response_time_ms=elapsed_ms,
        fields_returned=fields_returned,
        fields_populated=fields_populated,
        rate_limit=safe_rate_limit_text(dict(live.get("rate_limit", {})), status_code=status_code),
        fallback_used=False,
        request_id=_request_id(payload),
        error_code=error_code,
        error_message=error_message,
        raw_field_names=list(payload.keys()),
    )


def lookup_omkar_metadata(number: str, api_key: str, *, timeout: float = 10.0) -> PhoneProviderResult:
    normalized = normalise_phone_query(number)
    api_key = str(api_key or "").strip()

    if not api_key:
        return PhoneProviderResult(
            provider_id=PROVIDER_ID,
            provider_name=PROVIDER_NAME,
            channel="metadata",
            status="not_configured",
            success=False,
            normalized_number=normalized,
            error_code="missing_key",
            error_message="Omkar Carrier Lookup API key is not configured.",
        )

    started = time.perf_counter()
    live = lookup_omkar_phone(format_phone_for_omkar(normalized), api_key, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = dict(live.get("record", {})) if isinstance(live.get("record"), dict) else {}
    status_code = live.get("status_code")
    error_message = str(live.get("error") or "")

    if bool(live.get("ok")):
        return PhoneProviderResult(
            provider_id=PROVIDER_ID,
            provider_name=PROVIDER_NAME,
            channel="metadata",
            status="success",
            success=True,
            normalized_number=normalized,
            data=payload,
            response_time_ms=elapsed_ms,
            rate_limit=dict(live.get("rate_limit", {})),
            request_id=_request_id(payload) or None,
        )

    return PhoneProviderResult(
        provider_id=PROVIDER_ID,
        provider_name=PROVIDER_NAME,
        channel="metadata",
        status="error",
        success=False,
        normalized_number=normalized,
        data=payload,
        error_code=normalize_error_code(status_code if isinstance(status_code, int) else None, error_message),
        error_message=error_message or "Omkar Carrier Lookup unavailable.",
        response_time_ms=elapsed_ms,
        rate_limit=dict(live.get("rate_limit", {})),
        request_id=_request_id(payload) or None,
    )
