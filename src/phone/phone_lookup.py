"""Phone lookup orchestration with API, local fallback, and unknown fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.phone.penipumy_client import (
    PenipuApiError,
    PenipuClientError,
    fetch_phone_reputation,
    normalise_phone_query,
    phone_digits,
    validate_phone_query,
)
from src.phone.phone_explainability import explain_phone_result
from src.phone.phone_rules import evaluate_phone_risk


LOCAL_PHONE_DATASET = Path("data") / "processed" / "phone" / "phone_dataset.csv"


def _match_keys(value: str) -> set[str]:
    digits = phone_digits(value)
    keys = {digits} if digits else set()

    if digits.startswith("60") and len(digits) > 3:
        keys.add("0" + digits[2:])
    elif digits.startswith("0") and len(digits) > 2:
        keys.add("60" + digits[1:])

    return {key for key in keys if key}


def _normalize_local_record(row: dict[str, Any], phone_number: str) -> dict[str, Any]:
    record = {
        key: ("" if pd.isna(value) else value)
        for key, value in dict(row).items()
    }
    record["phone"] = str(record.get("phone") or phone_number)
    record["police_report_count"] = int(record.get("police_report_count") or 0)
    record["verified_report_count"] = int(record.get("verified_report_count") or 0)
    record["spoofing_report_count"] = int(record.get("spoofing_report_count") or 0)
    record["spam"] = str(record.get("spam", "")).strip().lower() in {"1", "true", "yes"}
    record["fraud"] = str(record.get("fraud", "")).strip().lower() in {"1", "true", "yes"}
    record["source"] = str(record.get("source") or "local_processed")
    return record


def _load_local_dataset(root: Path) -> pd.DataFrame:
    path = root / LOCAL_PHONE_DATASET
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, dtype={"phone": str})
    except Exception:
        return pd.DataFrame()


def _lookup_local_dataset(root: Path, phone_number: str) -> dict[str, Any] | None:
    dataset = _load_local_dataset(root)
    if dataset.empty or "phone" not in dataset.columns:
        return None

    target_keys = _match_keys(phone_number)
    if not target_keys:
        return None

    normalized = dataset.copy()
    normalized["_phone_key"] = normalized["phone"].astype(str).map(phone_digits)
    normalized["_alt_key"] = normalized["_phone_key"].map(
        lambda value: "0" + value[2:] if value.startswith("60") and len(value) > 3 else (
            "60" + value[1:] if value.startswith("0") and len(value) > 2 else value
        )
    )

    match = normalized[
        normalized["_phone_key"].isin(target_keys) | normalized["_alt_key"].isin(target_keys)
    ]
    if match.empty:
        return None

    record = match.iloc[0].drop(labels=["_phone_key", "_alt_key"], errors="ignore").to_dict()
    return _normalize_local_record(record, phone_number)


def _unknown_record(phone_number: str) -> dict[str, Any]:
    return {
        "phone": normalise_phone_query(phone_number),
        "police_report_count": 0,
        "verified_report_count": 0,
        "spam": False,
        "fraud": False,
        "business_tier": "none",
        "business_name": "",
        "spoofing_report_count": 0,
        "source": "unknown_fallback",
        "found": False,
    }


def _build_result(
    *,
    source: str,
    fallback_reason: str,
    found: bool,
    record: dict[str, Any],
    rate_limit: dict[str, str] | None = None,
) -> dict[str, Any]:
    risk = evaluate_phone_risk(record)
    explanation = explain_phone_result(record, risk)
    return {
        "source": source,
        "fallback_reason": fallback_reason,
        "found": found,
        "record": record,
        "risk": risk,
        "explanation": explanation,
        "rate_limit": rate_limit or {},
    }


def lookup_phone(phone_number: str, root: Path, api_key: str = "") -> dict[str, Any]:
    """Run the 3-level phone reputation lookup chain."""

    ok, message = validate_phone_query(phone_number)
    if not ok:
        raise ValueError(message)

    normalized = normalise_phone_query(phone_number)
    fallback_reason = ""

    if api_key.strip():
        try:
            live = fetch_phone_reputation(normalized, api_key)
            if str(live.rate_limit.get("remaining", "")).strip() == "0":
                fallback_reason = "PenipuMY daily quota remaining is 0."
            else:
                record = dict(live.data)
                record["source"] = "penipumy_api"
                return _build_result(
                    source="penipumy_api",
                    fallback_reason="",
                    found=True,
                    record=record,
                    rate_limit=live.rate_limit,
                )
        except PenipuApiError as exc:
            fallback_reason = str(exc)
        except PenipuClientError as exc:
            fallback_reason = str(exc)
    else:
        fallback_reason = "PenipuMY API key unavailable."

    local_record = _lookup_local_dataset(root, normalized)
    if local_record is not None:
        return _build_result(
            source="local_fallback",
            fallback_reason=fallback_reason,
            found=True,
            record=local_record,
        )

    return _build_result(
        source="unknown_fallback",
        fallback_reason=fallback_reason or "Live lookup unavailable and no local record matched.",
        found=False,
        record=_unknown_record(normalized),
    )


__all__ = [
    "lookup_phone",
    "normalise_phone_query",
    "phone_digits",
    "validate_phone_query",
]
