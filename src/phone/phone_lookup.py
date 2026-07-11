"""Phone lookup orchestration with API, local fallback, and unknown fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.phone.penipumy_client import (
    PenipuApiError,
    PenipuClientError,
    fetch_phone_reputation,
)
from src.phone.phone_number import (
    format_phone_for_ipqs,
    format_phone_for_omkar,
    format_phone_for_penipumy,
    normalise_phone_query,
    phone_digits,
    validate_phone_query,
)
from src.phone.ipqs_client import lookup_ipqs_phone
from src.phone.omkar_client import lookup_omkar_phone
from src.phone.phone_explainability import explain_phone_result
from src.phone.phone_rules import evaluate_phone_risk


LOCAL_PHONE_DATASET = Path("data") / "processed" / "phone" / "phone_dataset.csv"
DEMO_PHONE_DATASET = Path("data") / "demo" / "phone_demo_dataset.csv"
SUPPORTED_PROVIDERS = {"penipumy", "ipqualityscore", "omkar_carrier_lookup"}


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
    record["provider"] = "local_fallback"
    record["record_type"] = str(record.get("record_type") or "").strip()
    record["is_demo"] = str(record.get("is_demo", "")).strip().lower() in {"1", "true", "yes", "y"}
    record["source_reference"] = str(record.get("source_reference") or "").strip()
    record["last_verified"] = str(record.get("last_verified") or "").strip()
    return record


def _row_is_demo(row: dict[str, Any]) -> bool:
    values = {
        "is_demo": str(row.get("is_demo", "")).strip().lower(),
        "record_type": str(row.get("record_type", "")).strip().lower(),
        "source": str(row.get("source", "")).strip().lower(),
        "source_reference": str(row.get("source_reference", "")).strip().lower(),
    }
    return (
        values["is_demo"] in {"1", "true", "yes", "y"}
        or values["record_type"] == "demo"
        or "demo" in values["source"]
        or "synthetic" in values["source"]
        or "demo" in values["source_reference"]
        or "synthetic" in values["source_reference"]
    )


def _normalized_base(phone_number: str, provider: str, source: str) -> dict[str, Any]:
    return {
        "phone": normalise_phone_query(phone_number),
        "provider": provider,
        "police_report_count": 0,
        "verified_report_count": 0,
        "spam": False,
        "fraud": False,
        "spoofing_report_count": 0,
        "valid": None,
        "active": None,
        "fraud_score": None,
        "recent_abuse": None,
        "risky": None,
        "spammer": None,
        "line_type": None,
        "carrier": None,
        "country": None,
        "region": None,
        "city": None,
        "voip": None,
        "prepaid": None,
        "business_tier": None,
        "business_name": None,
        "source": source,
    }


def _normalize_penipumy_record(record: dict[str, Any], phone_number: str) -> dict[str, Any]:
    normalized = _normalized_base(phone_number, "penipumy", "penipumy_api")
    normalized.update(record)
    normalized["phone"] = str(record.get("phone") or normalise_phone_query(phone_number))
    normalized["provider"] = "penipumy"
    normalized["source"] = "penipumy_api"
    normalized["police_report_count"] = int(record.get("police_report_count") or 0)
    normalized["verified_report_count"] = int(record.get("verified_report_count") or 0)
    normalized["spam"] = bool(record.get("spam", False))
    normalized["fraud"] = bool(record.get("fraud", False))

    business = record.get("business")
    if isinstance(business, dict):
        normalized["business_tier"] = business.get("tier")
        normalized["business_name"] = (
            business.get("display_name")
            or business.get("brand_name")
            or business.get("business_name")
        )
        normalized["spoofing_report_count"] = int(business.get("spoofing_report_count") or 0)

    return normalized


def _normalize_ipqs_record(record: dict[str, Any], phone_number: str) -> dict[str, Any]:
    normalized = _normalized_base(phone_number, "ipqualityscore", "ipqualityscore_api")
    normalized.update(
        {
            "phone": str(record.get("formatted") or record.get("local_format") or normalise_phone_query(phone_number)),
            "provider": "ipqualityscore",
            "source": "ipqualityscore_api",
            "valid": record.get("valid"),
            "active": record.get("active"),
            "fraud_score": record.get("fraud_score"),
            "recent_abuse": record.get("recent_abuse"),
            "risky": record.get("risky"),
            "spammer": record.get("spammer"),
            "line_type": record.get("line_type"),
            "carrier": record.get("carrier"),
            "country": record.get("country"),
            "region": record.get("region"),
            "city": record.get("city"),
            "voip": record.get("VOIP", record.get("voip")),
            "prepaid": record.get("prepaid"),
            "do_not_call": record.get("do_not_call"),
            "leaked": record.get("leaked"),
            "user_activity": record.get("user_activity"),
            "active_status": record.get("active_status"),
            "formatted": record.get("formatted"),
            "local_format": record.get("local_format"),
            "name": record.get("name"),
            "timezone": record.get("timezone"),
            "zip_code": record.get("zip_code"),
        }
    )
    return normalized


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_omkar_record(record: dict[str, Any], phone_number: str) -> dict[str, Any]:
    line_type = str(record.get("line_type") or "").strip()
    normalized = _normalized_base(phone_number, "omkar_carrier_lookup", "omkar_carrier_lookup_api")
    normalized.update(
        {
            "phone": str(record.get("phone_number") or normalise_phone_query(phone_number)),
            "provider": "omkar_carrier_lookup",
            "source": "omkar_carrier_lookup_api",
            "valid": _bool_or_none(record.get("is_valid_number")),
            "active": None,
            "line_type": line_type,
            "carrier": record.get("carrier"),
            "country": record.get("country_code"),
            "voip": line_type.lower() == "voip",
            "formatted": record.get("phone_number"),
            "national_format": record.get("national_format"),
            "calling_country_code": record.get("calling_country_code"),
            "mobile_country_code": record.get("mobile_country_code"),
            "mobile_network_code": record.get("mobile_network_code"),
        }
    )
    return normalized


def _load_phone_dataset(root: Path, path: Path) -> pd.DataFrame:
    path = root / path
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, dtype={"phone": str})
    except Exception:
        return pd.DataFrame()


def _load_local_dataset(root: Path, include_demo: bool = False) -> pd.DataFrame:
    datasets = [_load_phone_dataset(root, LOCAL_PHONE_DATASET)]
    if include_demo:
        datasets.append(_load_phone_dataset(root, DEMO_PHONE_DATASET))

    datasets = [dataset for dataset in datasets if not dataset.empty]
    if not datasets:
        return pd.DataFrame()

    dataset = pd.concat(datasets, ignore_index=True)
    if include_demo:
        return dataset

    rows = []
    for row in dataset.to_dict(orient="records"):
        if not _row_is_demo(row):
            rows.append(row)
    return pd.DataFrame(rows)


def _lookup_local_dataset(root: Path, phone_number: str, *, include_demo: bool = False) -> dict[str, Any] | None:
    dataset = _load_local_dataset(root, include_demo=include_demo)
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


def _unknown_record(phone_number: str, requested_provider: str) -> dict[str, Any]:
    record = _normalized_base(phone_number, requested_provider, "unknown_fallback")
    record.update(
        {
            "business_tier": "none",
            "business_name": "",
            "found": False,
        }
    )
    return record


def _build_result(
    *,
    source: str,
    fallback_reason: str,
    found: bool,
    record: dict[str, Any],
    rate_limit: dict[str, str] | None = None,
    requested_provider: str = "penipumy",
    live_provider_status: str = "",
    raw_provider_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    risk = evaluate_phone_risk(record)
    explanation = explain_phone_result(record, risk)
    return {
        "source": source,
        "provider": str(record.get("provider") or requested_provider),
        "requested_provider": requested_provider,
        "live_provider_status": live_provider_status,
        "fallback_reason": fallback_reason,
        "found": found,
        "is_demo": bool(record.get("is_demo")),
        "record": record,
        "risk": risk,
        "explanation": explanation,
        "rate_limit": rate_limit or {},
        "raw_provider_record": raw_provider_record or {},
    }


def _normalize_provider_key(provider: str) -> str:
    key = str(provider or "penipumy").strip().lower().replace(" ", "")
    aliases = {
        "penipu": "penipumy",
        "penipumy": "penipumy",
        "ipqs": "ipqualityscore",
        "ipqualityscore": "ipqualityscore",
        "carrier": "omkar_carrier_lookup",
        "carrierlookup": "omkar_carrier_lookup",
        "carrier_lookup": "omkar_carrier_lookup",
        "omkar": "omkar_carrier_lookup",
        "omkarcarrierlookup": "omkar_carrier_lookup",
        "omkar_carrier_lookup": "omkar_carrier_lookup",
    }
    return aliases.get(key, key)


def lookup_phone(
    phone_number: str,
    root: Path,
    *,
    provider: str = "penipumy",
    api_key: str = "",
    demo_mode: bool = False,
) -> dict[str, Any]:
    """Run the 3-level phone reputation lookup chain."""

    ok, message = validate_phone_query(phone_number)
    if not ok:
        raise ValueError(message)

    provider_key = _normalize_provider_key(provider)
    if provider_key not in SUPPORTED_PROVIDERS:
        raise ValueError("Unsupported phone lookup provider.")

    normalized = normalise_phone_query(phone_number)
    fallback_reason = ""
    rate_limit: dict[str, str] = {}

    if api_key.strip():
        if provider_key == "penipumy":
            try:
                live = fetch_phone_reputation(format_phone_for_penipumy(normalized), api_key)
                rate_limit = live.rate_limit
                if str(live.rate_limit.get("remaining", "")).strip() == "0":
                    fallback_reason = "PenipuMY daily quota remaining is 0."
                else:
                    raw_record = dict(live.data)
                    record = _normalize_penipumy_record(raw_record, normalized)
                    return _build_result(
                        source="penipumy_api",
                        fallback_reason="",
                        found=True,
                        record=record,
                        rate_limit=live.rate_limit,
                        requested_provider=provider_key,
                        live_provider_status="success",
                        raw_provider_record=raw_record,
                    )
            except PenipuApiError as exc:
                fallback_reason = str(exc)
            except PenipuClientError as exc:
                fallback_reason = str(exc)
        elif provider_key == "ipqualityscore":
            live_result = lookup_ipqs_phone(format_phone_for_ipqs(normalized), api_key)
            rate_limit = dict(live_result.get("rate_limit", {}))
            if bool(live_result.get("ok")):
                raw_record = dict(live_result.get("record", {}))
                record = _normalize_ipqs_record(raw_record, normalized)
                return _build_result(
                    source="ipqualityscore_api",
                    fallback_reason="",
                    found=True,
                    record=record,
                    rate_limit=rate_limit,
                    requested_provider=provider_key,
                    live_provider_status="success",
                    raw_provider_record=raw_record,
                )
            fallback_reason = str(live_result.get("error") or "IPQualityScore lookup unavailable.")
        else:
            live_result = lookup_omkar_phone(format_phone_for_omkar(normalized), api_key)
            rate_limit = dict(live_result.get("rate_limit", {}))
            if bool(live_result.get("ok")):
                raw_record = dict(live_result.get("record", {}))
                record = _normalize_omkar_record(raw_record, phone_number)
                return _build_result(
                    source="omkar_carrier_lookup_api",
                    fallback_reason="",
                    found=True,
                    record=record,
                    rate_limit=rate_limit,
                    requested_provider=provider_key,
                    live_provider_status="success",
                    raw_provider_record=raw_record,
                )
            fallback_reason = str(live_result.get("error") or "Omkar Carrier Lookup unavailable.")
    else:
        if provider_key == "penipumy":
            fallback_reason = "PenipuMY API key unavailable."
        elif provider_key == "ipqualityscore":
            fallback_reason = "IPQualityScore API key unavailable."
        else:
            fallback_reason = "Omkar Carrier Lookup API key unavailable."

    local_record = _lookup_local_dataset(root, normalized, include_demo=demo_mode)
    if local_record is not None:
        source = "demo_fallback" if bool(local_record.get("is_demo")) else "local_fallback"
        return _build_result(
            source=source,
            fallback_reason=fallback_reason,
            found=True,
            record=local_record,
            rate_limit=rate_limit,
            requested_provider=provider_key,
            live_provider_status="fallback",
        )

    return _build_result(
        source="unknown_fallback",
        fallback_reason=fallback_reason or "Live lookup unavailable and no local record matched.",
        found=False,
        record=_unknown_record(normalized, provider_key),
        rate_limit=rate_limit,
        requested_provider=provider_key,
        live_provider_status="fallback",
    )


__all__ = [
    "lookup_phone",
    "normalise_phone_query",
    "phone_digits",
    "validate_phone_query",
]
