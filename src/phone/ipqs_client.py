"""Low-level IPQualityScore phone validation client.

This module only talks to IPQS. Fallback selection, project risk rules, and
explainability stay in sibling modules.
"""

from __future__ import annotations

from typing import Any


IPQS_PHONE_API_BASE_URL = "https://www.ipqualityscore.com/api/json/phone"


def _rate_limit_from_headers(headers: Any) -> dict[str, str]:
    return {
        "limit": str(headers.get("X-RateLimit-Limit", "") or headers.get("X-Rate-Limit-Limit", "")),
        "remaining": str(headers.get("X-RateLimit-Remaining", "") or headers.get("X-Rate-Limit-Remaining", "")),
        "reset": str(headers.get("X-RateLimit-Reset", "") or headers.get("X-Rate-Limit-Reset", "")),
    }


def _failure(
    *,
    status_code: int | None,
    error: str,
    rate_limit: dict[str, str] | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, object]:
    return {
        "ok": False,
        "provider": "ipqualityscore",
        "status_code": status_code,
        "record": record or {},
        "rate_limit": rate_limit or {},
        "error": error,
    }


def _safe_error(message: object, api_key: str) -> str:
    text = str(message)
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text


def lookup_ipqs_phone(
    phone: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    base_url: str = IPQS_PHONE_API_BASE_URL,
) -> dict[str, object]:
    """Call IPQualityScore's Phone Number Validation API."""

    api_key = str(api_key or "").strip()
    phone = str(phone or "").strip()

    if not api_key:
        return _failure(status_code=None, error="IPQualityScore API key is missing.")
    if not phone:
        return _failure(status_code=None, error="Phone number is missing.")

    try:
        import requests
    except Exception as exc:
        return _failure(
            status_code=None,
            error=f"Install `requests` to use the IPQualityScore integration: {exc}",
        )

    endpoint = f"{base_url.rstrip('/')}/{api_key}/{phone}"
    params = {"strictness": 1}

    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
    except requests.Timeout:
        return _failure(status_code=None, error="IPQualityScore lookup timed out.")
    except requests.RequestException as exc:
        return _failure(status_code=None, error=f"IPQualityScore lookup failed: {_safe_error(exc, api_key)}")

    rate_limit = _rate_limit_from_headers(response.headers)

    try:
        payload = response.json()
    except ValueError:
        return _failure(
            status_code=response.status_code,
            error=f"IPQualityScore returned malformed JSON with status {response.status_code}.",
            rate_limit=rate_limit,
        )

    if not isinstance(payload, dict):
        return _failure(
            status_code=response.status_code,
            error="IPQualityScore returned an unexpected response format.",
            rate_limit=rate_limit,
        )

    if response.status_code in {401, 403}:
        return _failure(
            status_code=response.status_code,
            error="IPQualityScore API key was rejected.",
            rate_limit=rate_limit,
            record=payload,
        )
    if response.status_code == 429:
        return _failure(
            status_code=response.status_code,
            error="IPQualityScore API rate limit reached.",
            rate_limit=rate_limit,
            record=payload,
        )
    if response.status_code in {500, 502, 503}:
        return _failure(
            status_code=response.status_code,
            error=f"IPQualityScore server error {response.status_code}.",
            rate_limit=rate_limit,
            record=payload,
        )
    if response.status_code != 200:
        message = str(payload.get("message") or payload.get("error") or response.reason or "Unknown API error")
        return _failure(
            status_code=response.status_code,
            error=f"IPQualityScore returned {response.status_code}: {message}",
            rate_limit=rate_limit,
            record=payload,
        )

    if payload.get("success") is not True:
        message = str(payload.get("message") or "IPQualityScore returned success=false.")
        return _failure(
            status_code=response.status_code,
            error=message,
            rate_limit=rate_limit,
            record=payload,
        )

    return {
        "ok": True,
        "provider": "ipqualityscore",
        "status_code": response.status_code,
        "record": payload,
        "rate_limit": rate_limit,
        "error": None,
    }


__all__ = ["lookup_ipqs_phone"]
