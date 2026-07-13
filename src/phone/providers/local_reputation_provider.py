"""Local fallback reputation dataset adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.phone.phone_number import normalise_phone_query, phone_digits
from src.phone.providers.models import PhoneProviderResult


LOCAL_PHONE_DATASET = Path("data") / "processed" / "phone" / "phone_dataset.csv"
REQUIRED_COLUMNS = {
    "phone",
    "police_report_count",
    "verified_report_count",
    "spam",
    "fraud",
    "business_tier",
    "business_name",
    "spoofing_report_count",
    "source",
}


@dataclass
class LocalDatasetStatus:
    exists: bool
    schema_valid: bool
    record_count: int
    source_path: str
    source_filename: str
    last_modified: str
    error_message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataset_path(root: Path) -> Path:
    return root / LOCAL_PHONE_DATASET


def _load_dataset(root: Path) -> pd.DataFrame:
    path = _dataset_path(root)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"phone": str})


def get_local_dataset_status(root: Path) -> LocalDatasetStatus:
    path = _dataset_path(root)
    if not path.exists():
        return LocalDatasetStatus(
            exists=False,
            schema_valid=False,
            record_count=0,
            source_path=str(path),
            source_filename=path.name,
            last_modified="",
            error_message="Local phone dataset is missing.",
        )

    try:
        dataset = pd.read_csv(path, dtype={"phone": str})
    except Exception as exc:
        return LocalDatasetStatus(
            exists=True,
            schema_valid=False,
            record_count=0,
            source_path=str(path),
            source_filename=path.name,
            last_modified="",
            error_message=f"Could not read local phone dataset: {exc}",
        )

    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataset.columns))
    modified = ""
    try:
        modified = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        modified = ""

    return LocalDatasetStatus(
        exists=True,
        schema_valid=not missing_columns,
        record_count=len(dataset),
        source_path=str(path),
        source_filename=path.name,
        last_modified=modified,
        error_message=(
            f"Missing required columns: {', '.join(missing_columns)}"
            if missing_columns
            else ""
        ),
    )


def _match_keys(value: str) -> set[str]:
    digits = phone_digits(value)
    keys = {digits} if digits else set()
    if digits.startswith("60") and len(digits) > 3:
        keys.add("0" + digits[2:])
    elif digits.startswith("0") and len(digits) > 2:
        keys.add("60" + digits[1:])
    return {key for key in keys if key}


def _normalise_row(row: dict[str, Any], normalized_number: str) -> dict[str, Any]:
    clean = {
        key: ("" if pd.isna(value) else value)
        for key, value in row.items()
    }
    clean["phone"] = str(clean.get("phone") or normalized_number)
    for key in ("police_report_count", "verified_report_count", "spoofing_report_count"):
        try:
            clean[key] = int(clean.get(key) or 0)
        except (TypeError, ValueError):
            clean[key] = 0
    for key in ("spam", "fraud"):
        clean[key] = str(clean.get(key, "")).strip().lower() in {"1", "true", "yes", "y"}
    clean["provider"] = "local_fallback"
    clean["source"] = str(clean.get("source") or "local_processed")
    return clean


def lookup_local_reputation(root: Path, number: str) -> PhoneProviderResult:
    normalized = normalise_phone_query(number)
    status = get_local_dataset_status(root)
    if not status.exists or not status.schema_valid:
        return PhoneProviderResult(
            provider_id="local_fallback",
            provider_name="Local Fallback Dataset",
            channel="fallback",
            status="unavailable",
            success=False,
            normalized_number=normalized,
            error_code="dataset_unavailable",
            error_message=status.error_message or "Local phone dataset is unavailable.",
        )

    dataset = _load_dataset(root)
    target_keys = _match_keys(normalized)
    if dataset.empty or not target_keys:
        return PhoneProviderResult(
            provider_id="local_fallback",
            provider_name="Local Fallback Dataset",
            channel="fallback",
            status="no_match",
            success=True,
            normalized_number=normalized,
            data={},
        )

    working = dataset.copy()
    working["_phone_key"] = working["phone"].astype(str).map(phone_digits)
    working["_alt_key"] = working["_phone_key"].map(
        lambda value: "0" + value[2:]
        if value.startswith("60") and len(value) > 3
        else ("60" + value[1:] if value.startswith("0") and len(value) > 2 else value)
    )

    match = working[
        working["_phone_key"].isin(target_keys) | working["_alt_key"].isin(target_keys)
    ]
    if match.empty:
        return PhoneProviderResult(
            provider_id="local_fallback",
            provider_name="Local Fallback Dataset",
            channel="fallback",
            status="no_match",
            success=True,
            normalized_number=normalized,
            data={},
        )

    record = match.iloc[0].drop(labels=["_phone_key", "_alt_key"], errors="ignore").to_dict()
    return PhoneProviderResult(
        provider_id="local_fallback",
        provider_name="Local Fallback Dataset",
        channel="fallback",
        status="success",
        success=True,
        normalized_number=normalized,
        data=_normalise_row(record, normalized),
        fallback_used=True,
        fallback_reason="Local fallback dataset matched the normalized number.",
    )
