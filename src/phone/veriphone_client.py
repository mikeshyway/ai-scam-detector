"""Low-level Veriphone phone metadata client."""

from __future__ import annotations

from typing import Any


VERIPHONE_API_BASE_URL = "https://api.veriphone.io/v3"
SUPPORTED_LOOKUP_MODES = {"static", "current"}


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
        "provider": "veriphone",
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
        or payload.get("type")
        or fallback
        or ""
    )


def lookup_veriphone_phone(
    phone: str,
    api_key: str,
    *,
    mode: str = "static",
    timeout: float = 10.0,
    base_url: str = VERIPHONE_API_BASE_URL,
) -> dict[str, object]:
    """Call Veriphone's phone verification API."""

    api_key = str(api_key or "").strip()
    phone = str(phone or "").strip()
    mode = str(mode or "static").strip().lower()

    if not api_key:
        return _failure(status_code=None, error="Veriphone API key is missing.")
    if not phone:
        return _failure(status_code=None, error="Phone number is missing.")
    if mode not in SUPPORTED_LOOKUP_MODES:
        return _failure(status_code=None, error=f"Unsupported Veriphone lookup mode: {mode}.")

    try:
        import requests
    except Exception as exc:
        return _failure(status_code=None, error=f"Install `requests` to use the Veriphone integration: {exc}")

    endpoint = f"{base_url.rstrip('/')}/verify"
    try:
        response = requests.get(
            endpoint,
            params={"phone": phone, "mode": mode},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.Timeout:
        return _failure(status_code=None, error="Veriphone lookup timed out.")
    except requests.ConnectionError:
        return _failure(status_code=None, error="Veriphone network connection failed.")
    except requests.RequestException as exc:
        return _failure(status_code=None, error=f"Veriphone lookup failed: {_safe_error(exc, api_key)}")

    rate_limit = _rate_limit_from_headers(response.headers)

    try:
        payload = response.json()
    except ValueError:
        return _failure(
            status_code=response.status_code,
            error=f"Veriphone returned malformed JSON with status {response.status_code}.",
            rate_limit=rate_limit,
        )

    if not isinstance(payload, dict):
        return _failure(
            status_code=response.status_code,
            error="Veriphone returned an unexpected response format.",
            rate_limit=rate_limit,
        )

    if response.status_code != 200:
        message = _response_message(payload, response.reason or "Unknown API error")
        return _failure(
            status_code=response.status_code,
            error=f"Veriphone returned {response.status_code}: {_safe_error(message, api_key)}",
            rate_limit=rate_limit,
            record=payload,
        )

    if payload.get("status") == "error":
        code = payload.get("code")
        status_code = int(code) if isinstance(code, int) else response.status_code
        return _failure(
            status_code=status_code,
            error=f"Veriphone returned {status_code}: {_safe_error(_response_message(payload), api_key)}",
            rate_limit=rate_limit,
            record=payload,
        )

    if payload.get("status") != "success":
        return _failure(
            status_code=response.status_code,
            error="Veriphone returned an unexpected status.",
            rate_limit=rate_limit,
            record=payload,
        )

    return {
        "ok": True,
        "provider": "veriphone",
        "status_code": response.status_code,
        "record": payload,
        "rate_limit": rate_limit,
        "error": None,
    }


__all__ = ["lookup_veriphone_phone"]
