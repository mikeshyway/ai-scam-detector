"""Veriphone.io metadata provider adapter."""

from __future__ import annotations

import time
from typing import Any

from src.phone.phone_number import format_phone_for_veriphone, normalise_phone_query, validate_phone_query
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
from src.phone.veriphone_client import lookup_veriphone_phone


PROVIDER_ID = "veriphone"
PROVIDER_NAME = "Veriphone.io Carrier Lookup"
PRIMARY_TIMEOUT_SECONDS = 10.0
RETRY_TIMEOUT_SECONDS = 20.0
DEFAULT_LOOKUP_MODE = "static"


def _request_id(payload: dict[str, Any]) -> str:
    return str(payload.get("request_id") or payload.get("id") or "")


def _mcc_mnc_parts(value: object) -> tuple[str | None, str | None]:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) < 5:
        return None, None
    return digits[:3], digits[3:]


def _normalise_line_type(value: object) -> str:
    return str(value or "").strip().replace("_", " ")


def _normalise_veriphone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mcc, mnc = _mcc_mnc_parts(payload.get("current_mccmnc"))
    timezone = payload.get("timezone")
    if isinstance(timezone, list):
        timezone_value: object = ", ".join(str(item) for item in timezone if str(item).strip())
    else:
        timezone_value = timezone

    carrier = payload.get("current_carrier") or payload.get("carrier")
    line_type = payload.get("current_line_type") or payload.get("phone_type")
    phone_number = payload.get("e164") or payload.get("phone")

    return {
        "is_valid_number": payload.get("phone_valid"),
        "phone_number": phone_number,
        "national_format": payload.get("local_number"),
        "international_number": payload.get("international_number"),
        "country": payload.get("country"),
        "country_code": payload.get("country_code"),
        "calling_country_code": payload.get("country_prefix"),
        "carrier": carrier,
        "line_type": _normalise_line_type(line_type),
        "mobile_country_code": mcc,
        "mobile_network_code": mnc,
        "phone_region": payload.get("phone_region"),
        "timezone": timezone_value,
        "geographical": payload.get("geographical"),
        "mode": payload.get("mode") or DEFAULT_LOOKUP_MODE,
        "ported": payload.get("ported"),
        "carrier_data_source": payload.get("carrier_data_source"),
        "reason": payload.get("reason"),
        "raw_veriphone": payload,
    }


def _lookup_with_timeout_retry(
    number: str,
    api_key: str,
    *,
    mode: str = DEFAULT_LOOKUP_MODE,
    timeout: float = PRIMARY_TIMEOUT_SECONDS,
    retry_timeout: float = RETRY_TIMEOUT_SECONDS,
) -> tuple[dict[str, object], bool]:
    result = lookup_veriphone_phone(number, api_key, mode=mode, timeout=timeout)
    status_code = result.get("status_code")
    error_code = normalize_error_code(
        status_code if isinstance(status_code, int) else None,
        result.get("error"),
    )
    if error_code != "timeout":
        return result, False

    retry_result = lookup_veriphone_phone(number, api_key, mode=mode, timeout=retry_timeout)
    retry_status_code = retry_result.get("status_code")
    retry_error_code = normalize_error_code(
        retry_status_code if isinstance(retry_status_code, int) else None,
        retry_result.get("error"),
    )
    if retry_error_code == "timeout":
        retry_result = dict(retry_result)
        retry_result["error"] = f"Veriphone.io timed out after retry ({int(timeout)}s then {int(retry_timeout)}s)."
    return retry_result, True


def test_veriphone_connection(
    test_number: str,
    api_key: str,
    *,
    key_source: str,
    key_variable: str,
    timeout: float = PRIMARY_TIMEOUT_SECONDS,
    mode: str = DEFAULT_LOOKUP_MODE,
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
            error_message="Veriphone.io API key is not configured.",
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

    live, retry_attempted = _lookup_with_timeout_retry(
        format_phone_for_veriphone(test_number),
        api_key,
        mode=mode,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = dict(live.get("record", {})) if isinstance(live.get("record"), dict) else {}
    fields_returned, fields_populated = top_level_field_counts(payload)
    status_code = live.get("status_code")
    status_code = int(status_code) if isinstance(status_code, int) else status_code
    error_message = str(live.get("error") or "")
    success = bool(live.get("ok"))
    error_code = normalize_error_code(status_code if isinstance(status_code, int) else None, error_message)

    return ProviderDiagnosticResult(
        provider_id=PROVIDER_ID,
        provider_name=PROVIDER_NAME,
        configured=True,
        key_detected=True,
        key_source=key_source,
        key_variable=key_variable,
        reachable=provider_reachable(True, status_code if isinstance(status_code, int) else None, error_code),
        authentication_status=authentication_status(
            True,
            status_code if isinstance(status_code, int) else None,
            error_code,
        ),
        request_accepted=request_accepted(status_code if isinstance(status_code, int) else None, success, error_code),
        http_status=status_code if isinstance(status_code, int) else None,
        provider_success=success,
        response_time_ms=elapsed_ms,
        fields_returned=fields_returned,
        fields_populated=fields_populated,
        rate_limit=safe_rate_limit_text(dict(live.get("rate_limit", {})), status_code=status_code if isinstance(status_code, int) else None),
        fallback_used=False,
        request_id=_request_id(payload),
        error_code=error_code,
        error_message=error_message,
        raw_field_names=list(payload.keys()),
        retry_attempted=retry_attempted,
    )


def lookup_veriphone_metadata(
    number: str,
    api_key: str,
    *,
    timeout: float = PRIMARY_TIMEOUT_SECONDS,
    mode: str = DEFAULT_LOOKUP_MODE,
) -> PhoneProviderResult:
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
            error_message="Veriphone.io API key is not configured.",
        )

    started = time.perf_counter()
    live, retry_attempted = _lookup_with_timeout_retry(
        format_phone_for_veriphone(normalized),
        api_key,
        mode=mode,
        timeout=timeout,
    )
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
            data=_normalise_veriphone_payload(payload),
            response_time_ms=elapsed_ms,
            rate_limit=dict(live.get("rate_limit", {})),
            request_id=_request_id(payload) or None,
            retry_attempted=retry_attempted,
        )

    return PhoneProviderResult(
        provider_id=PROVIDER_ID,
        provider_name=PROVIDER_NAME,
        channel="metadata",
        status="error",
        success=False,
        normalized_number=normalized,
        data=_normalise_veriphone_payload(payload) if payload else {},
        error_code=normalize_error_code(status_code if isinstance(status_code, int) else None, error_message),
        error_message=error_message or "Veriphone.io unavailable.",
        response_time_ms=elapsed_ms,
        rate_limit=dict(live.get("rate_limit", {})),
        request_id=_request_id(payload) or None,
        retry_attempted=retry_attempted,
    )
