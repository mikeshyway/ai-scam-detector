"""Shared phone-number normalization and provider formatting helpers."""

from __future__ import annotations

import re


ALLOWED_INPUT_PATTERN = re.compile(r"^\+?[\d\s().-]+$")


def phone_digits(value: str) -> str:
    """Return only digit characters from a phone number."""

    return re.sub(r"\D+", "", str(value or ""))


def _has_valid_characters(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    if raw.count("+") > 1:
        return False
    if "+" in raw and not raw.lstrip().startswith("+"):
        return False
    return bool(ALLOWED_INPUT_PATTERN.match(raw))


def normalise_phone_query(value: str) -> str:
    """Normalize common phone inputs into E.164-style form.

    The app stores and displays one canonical form such as ``+60123456789``.
    Provider-specific helpers below convert that canonical value only when a
    provider requires a different representation.
    """

    raw = str(value or "").strip()
    digits = phone_digits(raw)
    if not digits:
        return ""

    if raw.startswith("+"):
        return f"+{digits}"
    if digits.startswith("60"):
        return f"+{digits}"
    if digits.startswith("0") and len(digits) > 1:
        return f"+60{digits[1:]}"
    if 8 <= len(digits) <= 10:
        return f"+60{digits}"
    return f"+{digits}"


def validate_phone_query(value: str) -> tuple[bool, str]:
    """Validate a phone input after accepting common visual formatting."""

    raw = str(value or "").strip()
    if not raw:
        return False, "Enter a phone number before checking reputation."
    if not _has_valid_characters(raw):
        return (
            False,
            "Use digits with optional spaces, hyphens, brackets, dots, and one leading plus sign only.",
        )

    normalized = normalise_phone_query(raw)
    digits = phone_digits(normalized)
    if not digits:
        return False, "Enter a phone number before checking reputation."
    if len(digits) < 8 or len(digits) > 15:
        return False, "Phone number must contain 8 to 15 digits after formatting is removed."
    return True, ""


def format_phone_for_penipumy(value: str) -> str:
    """Return a digit-only number for PenipuMY."""

    return phone_digits(normalise_phone_query(value))


def format_phone_for_ipqs(value: str) -> str:
    """Return canonical E.164-style input for IPQualityScore."""

    return normalise_phone_query(value)


def format_phone_for_omkar(value: str) -> str:
    """Return E.164-style input for Omkar Carrier Lookup."""

    return normalise_phone_query(value)
