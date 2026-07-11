"""Deterministic phone reputation risk rules."""

from __future__ import annotations

from typing import Any


HIGH_RISK_LEVEL = "High Risk"
MEDIUM_RISK_LEVEL = "Medium Risk"
LOW_RISK_LEVEL = "Low Risk"
UNKNOWN_RISK_LEVEL = "Unknown"


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def to_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def business_tier(record: dict[str, Any]) -> str:
    business = record.get("business")
    if isinstance(business, dict):
        return str(business.get("tier") or "").strip().lower()
    return str(record.get("business_tier") or "").strip().lower()


def business_name(record: dict[str, Any]) -> str:
    business = record.get("business")
    if isinstance(business, dict):
        return str(
            business.get("display_name")
            or business.get("brand_name")
            or business.get("business_name")
            or ""
        ).strip()
    return str(record.get("business_name") or "").strip()


def spoofing_report_count(record: dict[str, Any]) -> int:
    business = record.get("business")
    if isinstance(business, dict):
        return to_int(business.get("spoofing_report_count"))
    return to_int(record.get("spoofing_report_count"))


def evaluate_phone_risk(record: dict[str, Any]) -> dict[str, object]:
    """Convert reputation metadata into a concise risk level."""

    if record.get("found") is False:
        return {
            "risk_score": 0,
            "risk_level": UNKNOWN_RISK_LEVEL,
            "review_required": True,
            "decision_basis": ["Number not found in live API or local fallback dataset."],
            "recommended_action": (
                "Treat as unverified. Do not share OTPs, passwords, banking details, or personal information."
            ),
        }

    police_reports = to_int(record.get("police_report_count"))
    verified_reports = to_int(record.get("verified_report_count"))
    spoofing_reports = spoofing_report_count(record)
    spam = to_bool(record.get("spam"))
    fraud = to_bool(record.get("fraud"))
    tier = business_tier(record)
    source = str(record.get("source") or "").strip().lower()
    name = business_name(record)
    provider = str(record.get("provider") or "").strip().lower()

    basis: list[str] = []

    if provider == "omkar_carrier_lookup":
        valid = record.get("valid")
        valid_false = valid is False or str(valid).strip().lower() == "false"
        line_type = str(record.get("line_type") or "").strip()
        carrier = str(record.get("carrier") or "").strip()
        country = str(record.get("country") or "").strip()
        voip = to_bool(record.get("voip"))

        if valid_false:
            basis.append("invalid carrier lookup result")
        if line_type:
            basis.append(f"line type: {line_type}")
        if carrier and carrier.upper() != "N/A":
            basis.append(f"carrier: {carrier}")
        if country:
            basis.append(f"country: {country}")
        if voip:
            basis.append("VoIP line type context")

        if valid_false:
            return {
                "risk_score": 45,
                "risk_level": MEDIUM_RISK_LEVEL,
                "review_required": True,
                "decision_basis": basis or ["carrier lookup says the number is not valid"],
                "recommended_action": "Treat the caller as unverified and confirm the number through an official source.",
            }

        score = 28 if voip else 16
        return {
            "risk_score": score,
            "risk_level": LOW_RISK_LEVEL,
            "review_required": True,
            "decision_basis": basis or ["carrier lookup returned validation metadata only"],
            "recommended_action": (
                "Carrier data supports context only. It does not confirm caller identity or prove the caller is safe."
            ),
        }

    if provider == "ipqualityscore":
        fraud_score = to_float(record.get("fraud_score"))
        recent_abuse = to_bool(record.get("recent_abuse"))
        risky = to_bool(record.get("risky"))
        spammer = to_bool(record.get("spammer"))
        valid = record.get("valid")
        active = record.get("active")
        voip = to_bool(record.get("voip"))
        prepaid = to_bool(record.get("prepaid"))
        line_type = str(record.get("line_type") or "").strip()
        carrier = str(record.get("carrier") or "").strip()

        if fraud_score:
            basis.append(f"IPQS fraud score: {fraud_score:.0f}/100")
        if recent_abuse:
            basis.append("recent abuse signal")
        if risky:
            basis.append("risky phone metadata")
        if spammer:
            basis.append("spammer signal")
        if valid is False:
            basis.append("invalid number signal")
        if active is False:
            basis.append("inactive or disconnected signal")
        if voip:
            basis.append("VoIP line type context")
        if prepaid:
            basis.append("prepaid line context")
        if line_type:
            basis.append(f"line type: {line_type}")
        if carrier and carrier.upper() != "N/A":
            basis.append(f"carrier: {carrier}")

        high_risk = (
            fraud_score >= 85
            or recent_abuse
            or spammer
            or (risky and fraud_score >= 60)
            or (valid is False and (risky or fraud_score >= 50))
        )
        medium_risk = (
            risky
            or fraud_score >= 50
            or active is False
            or valid is False
        )

        if high_risk:
            score = max(78, min(99, int(round(fraud_score)) if fraud_score else 82))
            if recent_abuse:
                score = max(score, 88)
            if spammer:
                score = max(score, 86)
            return {
                "risk_score": score,
                "risk_level": HIGH_RISK_LEVEL,
                "review_required": True,
                "decision_basis": basis or ["high-risk IPQualityScore metadata"],
                "recommended_action": "Do not trust the caller identity. Verify through an official channel first.",
            }

        if medium_risk:
            score = max(42, min(69, int(round(fraud_score)) if fraud_score else 48))
            return {
                "risk_score": score,
                "risk_level": MEDIUM_RISK_LEVEL,
                "review_required": True,
                "decision_basis": basis or ["limited IPQualityScore risk metadata"],
                "recommended_action": "Verify the caller before sharing personal, banking, or authentication details.",
            }

        return {
            "risk_score": 14 if record.get("valid") is True else 24,
            "risk_level": LOW_RISK_LEVEL,
            "review_required": True,
            "decision_basis": basis or ["IPQualityScore did not return strong abuse indicators"],
            "recommended_action": "A lower-risk validation result is not proof of safety. Verify sensitive requests independently.",
        }

    high_risk = (
        fraud
        or verified_reports >= 3
        or police_reports >= 1
        or spoofing_reports >= 2
        or tier in {"verified_spoofing", "compromised"}
    )

    medium_risk = (
        spam
        or 1 <= verified_reports <= 2
        or tier in {"unverified"}
        or source in {"suspicious", "complaint_export"}
    )

    low_risk = (
        police_reports == 0
        and verified_reports == 0
        and not spam
        and not fraud
        and tier in {"verified", "registered", "government", "bank", "telecom", "delivery", "ecommerce"}
        and bool(name)
    )

    if fraud:
        basis.append("fraud flag returned")
    if police_reports:
        basis.append(f"{police_reports} police report(s)")
    if verified_reports:
        basis.append(f"{verified_reports} verified report(s)")
    if spoofing_reports:
        basis.append(f"{spoofing_reports} spoofing report(s)")
    if spam:
        basis.append("spam caller flag")
    if tier:
        basis.append(f"business tier: {tier}")

    if high_risk:
        score = 78
        score += min(police_reports * 6, 17)
        score += min(verified_reports * 4, 14)
        score += min(spoofing_reports * 5, 12)
        if fraud:
            score = max(score, 94)
        if tier == "compromised":
            score = max(score, 96)
        if tier == "verified_spoofing":
            score = max(score, 86)
        return {
            "risk_score": min(score, 99),
            "risk_level": HIGH_RISK_LEVEL,
            "review_required": True,
            "decision_basis": basis or ["high-risk reputation evidence"],
            "recommended_action": "Do not return the call. Verify through an official channel first.",
        }

    if medium_risk:
        score = 45
        score += min(verified_reports * 7, 18)
        score += 10 if spam else 0
        score += 8 if tier == "unverified" else 0
        return {
            "risk_score": min(score, 69),
            "risk_level": MEDIUM_RISK_LEVEL,
            "review_required": True,
            "decision_basis": basis or ["limited suspicious reputation evidence"],
            "recommended_action": "Proceed carefully and verify the caller identity before sharing information.",
        }

    if low_risk:
        return {
            "risk_score": 12,
            "risk_level": LOW_RISK_LEVEL,
            "review_required": False,
            "decision_basis": basis or ["verified organisation with no report indicators"],
            "recommended_action": "Still confirm sensitive requests through the organisation's official website or app.",
        }

    return {
        "risk_score": 22,
        "risk_level": LOW_RISK_LEVEL,
        "review_required": True,
        "decision_basis": basis or ["no reports found but caller identity is not verified"],
        "recommended_action": "No known report was found, but avoid sharing sensitive information with unknown callers.",
    }
