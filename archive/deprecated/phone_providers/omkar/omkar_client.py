"""Low-level Omkar carrier lookup client.

This module only talks to Omkar's Carrier Lookup API. Fallback selection,
project risk rules, and explainability stay in sibling modules.
"""

from __future__ import annotations

from typing import Any


OMKAR_CARRIER_API_BASE_URL = "https://carrier-lookup-api.omkar.cloud/lookup"


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
        "provider": "omkar_carrier_lookup",
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


def _response_message(payload: dict[str, Any], fallback: str = "") -> str:
    return str(
        payload.get("message")
        or payload.get("error")
        or payload.get("detail")
        or fallback
        or ""
    )


def lookup_omkar_phone(
    phone: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    base_url: str = OMKAR_CARRIER_API_BASE_URL,
) -> dict[str, object]:
    """Call Omkar's Carrier Lookup API."""

    api_key = str(api_key or "").strip()
    phone = str(phone or "").strip()

    if not api_key:
        return _failure(status_code=None, error="Omkar Carrier Lookup API key is missing.")
    if not phone:
        return _failure(status_code=None, error="Phone number is missing.")

    try:
        import requests
    except Exception as exc:
        return _failure(
            status_code=None,
            error=f"Install `requests` to use the Omkar Carrier Lookup integration: {exc}",
        )

    try:
        response = requests.get(
            base_url,
            params={"phone": phone},
            headers={"API-Key": api_key},
            timeout=timeout,
        )
    except requests.Timeout:
        return _failure(status_code=None, error="Omkar Carrier Lookup timed out.")
    except requests.ConnectionError:
        return _failure(status_code=None, error="Omkar Carrier Lookup network connection failed.")
    except requests.RequestException as exc:
        return _failure(status_code=None, error=f"Omkar Carrier Lookup failed: {_safe_error(exc, api_key)}")

    rate_limit = _rate_limit_from_headers(response.headers)

    try:
        payload = response.json()
    except ValueError:
        return _failure(
            status_code=response.status_code,
            error=f"Omkar Carrier Lookup returned malformed JSON with status {response.status_code}.",
            rate_limit=rate_limit,
        )

    if not isinstance(payload, dict):
        return _failure(
            status_code=response.status_code,
            error="Omkar Carrier Lookup returned an unexpected response format.",
            rate_limit=rate_limit,
        )

    if response.status_code in {401, 403}:
        return _failure(
            status_code=response.status_code,
            error="Omkar Carrier Lookup API key was rejected.",
            rate_limit=rate_limit,
            record=payload,
        )
    if response.status_code == 429:
        return _failure(
            status_code=response.status_code,
            error="Omkar Carrier Lookup API rate limit reached.",
            rate_limit=rate_limit,
            record=payload,
        )
    if response.status_code in {500, 502, 503}:
        return _failure(
            status_code=response.status_code,
            error=f"Omkar Carrier Lookup server error {response.status_code}.",
            rate_limit=rate_limit,
            record=payload,
        )
    if response.status_code != 200:
        message = _response_message(payload, response.reason or "Unknown API error")
        if response.status_code == 400 and "verify your phone number" in message.lower():
            message = (
                "Omkar account phone verification required. Verify a phone number on the Omkar account "
                "before free-plan carrier lookups are enabled."
            )
        return _failure(
            status_code=response.status_code,
            error=f"Omkar Carrier Lookup returned {response.status_code}: {message}",
            rate_limit=rate_limit,
            record=payload,
        )

    if payload.get("error"):
        return _failure(
            status_code=response.status_code,
            error=str(payload.get("error")),
            rate_limit=rate_limit,
            record=payload,
        )

    return {
        "ok": True,
        "provider": "omkar_carrier_lookup",
        "status_code": response.status_code,
        "record": payload,
        "rate_limit": rate_limit,
        "error": None,
    }


__all__ = ["lookup_omkar_phone"]
