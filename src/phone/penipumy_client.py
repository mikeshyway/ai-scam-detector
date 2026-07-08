"""Low-level PenipuMY API client.

This module should only know how to talk to PenipuMY. Fallback selection,
risk rules, and explainability live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


PENIPU_API_BASE_URL = "https://penipu.my/api/v1"


class PenipuClientError(RuntimeError):
    """Base error for PenipuMY API failures."""


class PenipuConfigurationError(PenipuClientError):
    """Raised when an API key is missing."""


class PenipuApiError(PenipuClientError):
    """Raised when PenipuMY returns a non-200 HTTP response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PenipuApiResponse:
    """Raw successful PenipuMY API response plus rate-limit metadata."""

    query: str
    status_code: int
    data: dict[str, Any]
    rate_limit: dict[str, str]


def phone_digits(value: str) -> str:
    """Return only digit characters from a phone number."""

    return re.sub(r"\D+", "", str(value or ""))


def normalise_phone_query(value: str) -> str:
    """Normalize a phone number while preserving a leading plus sign."""

    raw = str(value or "").strip()
    digits = phone_digits(raw)
    if raw.startswith("+") and digits:
        return f"+{digits}"
    return digits


def validate_phone_query(value: str) -> tuple[bool, str]:
    """Validate input against PenipuMY's phone lookup requirements."""

    digits = phone_digits(value)
    if not digits:
        return False, "Enter a phone number before checking reputation."
    if len(digits) < 8 or len(digits) > 15:
        return False, "Phone number must contain 8 to 15 digits."
    return True, ""


def fetch_phone_reputation(
    phone_number: str,
    api_key: str,
    *,
    base_url: str = PENIPU_API_BASE_URL,
    timeout_seconds: int = 12,
) -> PenipuApiResponse:
    """Call PenipuMY's phone endpoint and return the raw response."""

    api_key = str(api_key or "").strip()
    if not api_key:
        raise PenipuConfigurationError("PenipuMY API key is missing.")

    ok, message = validate_phone_query(phone_number)
    if not ok:
        raise PenipuClientError(message)

    try:
        import requests
    except Exception as exc:
        raise PenipuClientError("Install `requests` to use the PenipuMY API integration.") from exc

    query = normalise_phone_query(phone_number)
    endpoint = f"{base_url.rstrip('/')}/phone"

    try:
        response = requests.get(
            endpoint,
            headers={"X-API-Key": api_key},
            params={"q": query},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise PenipuClientError(f"PenipuMY lookup failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise PenipuClientError(
            f"PenipuMY returned a non-JSON response with status {response.status_code}."
        ) from exc

    if response.status_code != 200:
        detail = payload.get("error") if isinstance(payload, dict) else None
        detail = detail or response.reason or "Unknown API error"
        raise PenipuApiError(response.status_code, f"PenipuMY API returned {response.status_code}: {detail}")

    if not isinstance(payload, dict):
        raise PenipuClientError("PenipuMY returned an unexpected response format.")

    return PenipuApiResponse(
        query=query,
        status_code=response.status_code,
        data=payload,
        rate_limit={
            "limit": response.headers.get("X-RateLimit-Limit", ""),
            "remaining": response.headers.get("X-RateLimit-Remaining", ""),
        },
    )
