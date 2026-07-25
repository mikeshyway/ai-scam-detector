"""Explainability layer for phone reputation lookups."""

from __future__ import annotations

from typing import Any

from src.phone.phone_rules import (
    business_name,
    business_tier,
    spoofing_report_count,
    to_bool,
    to_int,
)


def _yes_no(value: object) -> str:
    return "Yes" if to_bool(value) else "No"


def _business_status(record: dict[str, Any]) -> str:
    tier = business_tier(record)
    name = business_name(record)
    if not tier or tier == "none":
        return "No business match"
    if name:
        return f"{tier.replace('_', ' ').title()} - {name}"
    return tier.replace("_", " ").title()


def _row(
    indicator: str,
    row_type: str,
    impact: str,
    evidence: str,
    intention: str,
) -> dict[str, object]:
    return {
        "Indicator": indicator,
        "Type": row_type,
        "Impact": impact,
        "Evidence": evidence,
        "Intention": intention,
    }


def build_phone_evidence_rows(record: dict[str, Any], risk_result: dict[str, object]) -> list[dict[str, object]]:
    """Create a table-ready evidence list for the phone UI."""

    rows: list[dict[str, object]] = []
    police_reports = to_int(record.get("police_report_count"))
    verified_reports = to_int(record.get("verified_report_count"))
    spoofing_reports = spoofing_report_count(record)
    spam = to_bool(record.get("spam"))
    fraud = to_bool(record.get("fraud"))
    tier = business_tier(record)
    name = business_name(record)
    source = str(record.get("source") or "unknown").strip()
    provider = str(record.get("provider") or "").strip().lower()

    if record.get("found") is False:
        return [
            _row(
                "Not found",
                "Unknown caller reputation",
                "Low",
                "No live provider match",
                "No known evidence, but not guaranteed safe.",
            )
        ]

    if provider == "veriphone":
        valid = record.get("valid")
        valid_false = valid is False or str(valid).strip().lower() == "false"
        valid_true = valid is True or str(valid).strip().lower() == "true"
        line_type = str(record.get("line_type") or "").strip()
        carrier = str(record.get("carrier") or "").strip()
        country = str(record.get("country") or "").strip()
        formatted = str(record.get("formatted") or record.get("phone") or "").strip()
        national_format = str(record.get("national_format") or "").strip()
        calling_code = str(record.get("calling_country_code") or "").strip()
        mobile_country_code = str(record.get("mobile_country_code") or "").strip()
        mobile_network_code = str(record.get("mobile_network_code") or "").strip()
        voip = to_bool(record.get("voip"))

        if valid is not None:
            rows.append(
                _row(
                    "Valid number",
                    "Carrier validation",
                    "Context" if valid_true else "Medium",
                    "Yes" if valid_true else "No",
                    "Carrier lookup checked number validity.",
                )
            )
        if carrier and carrier.upper() != "N/A":
            rows.append(
                _row(
                    "Carrier",
                    "Telecom metadata",
                    "Context",
                    carrier,
                    "Carrier information supports verification context.",
                )
            )
        if line_type:
            rows.append(
                _row(
                    "Line type",
                    "Telecom metadata",
                    "Context",
                    line_type,
                    "Line type helps verify caller claims.",
                )
            )
        if voip:
            rows.append(
                _row(
                    "VoIP line",
                    "Line type",
                    "Context",
                    line_type or "VoIP",
                    "Internet-based number; verify identity independently.",
                )
            )
        if country:
            rows.append(
                _row(
                    "Country",
                    "Location metadata",
                    "Context",
                    country,
                    "Country code may help verify caller claims.",
                )
            )
        if formatted or national_format:
            rows.append(
                _row(
                    "Formatted number",
                    "Number formatting",
                    "Context",
                    formatted or national_format,
                    "Formatted output supports manual verification.",
                )
            )
        if calling_code:
            rows.append(
                _row(
                    "Calling code",
                    "Country dialing metadata",
                    "Context",
                    calling_code,
                    "Dialing code can reveal country context.",
                )
            )
        if mobile_country_code or mobile_network_code:
            rows.append(
                _row(
                    "Mobile network codes",
                    "Carrier network metadata",
                    "Context",
                    f"MCC {mobile_country_code or '-'} / MNC {mobile_network_code or '-'}",
                    "Network codes support carrier identification.",
                )
            )

        if not rows:
            rows.append(
                _row(
                    "No carrier metadata",
                    "Provider metadata",
                    "Low",
                    source,
                    "Carrier lookup did not return usable fields.",
                )
            )

        return rows

    if fraud:
        rows.append(
            _row(
                "Fraud flag",
                "Fraud database",
                "High",
                "fraud=true",
                "Confirmed fraud signal from reputation data.",
            )
        )

    if police_reports:
        rows.append(
            _row(
                f"{police_reports} police report(s)",
                "Police report reputation",
                "High",
                str(record.get("police_report_status") or "local record"),
                "Previously reported through official channels.",
            )
        )

    if verified_reports:
        rows.append(
            _row(
                f"{verified_reports} verified report(s)",
                "Community reputation",
                "High" if verified_reports >= 3 else "Medium",
                "Verified community submissions",
                "Reported by multiple community submissions.",
            )
        )

    if spam:
        rows.append(
            _row(
                "Spam caller flag",
                "Caller ID reputation",
                "Medium",
                "spam=true",
                "Known nuisance or unwanted caller pattern.",
            )
        )

    if spoofing_reports:
        rows.append(
            _row(
                f"{spoofing_reports} spoofing report(s)",
                "Business impersonation",
                "High" if spoofing_reports >= 2 else "Medium",
                "Spoofing reports",
                "Legitimate number may be impersonated.",
            )
        )

    if tier and tier != "none":
        if tier in {"verified", "registered", "government", "bank", "telecom", "delivery", "ecommerce"}:
            impact = "Low"
            intention = "Number is linked to an identifiable organisation."
        elif tier in {"verified_spoofing", "compromised"}:
            impact = "High"
            intention = "Business identity may be misused by scammers."
        else:
            impact = "Medium"
            intention = "Business identity needs extra verification."

        rows.append(
            _row(
                name or tier.replace("_", " ").title(),
                "Business status",
                impact,
                tier,
                intention,
            )
        )

    business = record.get("business")
    if isinstance(business, dict) and business.get("is_ambiguous"):
        rows.append(
            _row(
                "Ambiguous business match",
                "Caller identity ambiguity",
                "Medium",
                "Multiple branch or routing match",
                "Caller identity requires extra verification.",
            )
        )

    if isinstance(business, dict) and business.get("scam_alert_banner"):
        rows.append(
            _row(
                "Active scam advisory",
                "Administrative warning",
                "High",
                str(business.get("scam_alert_banner")),
                "Official advisory warns about scam misuse.",
            )
        )

    if not rows:
        rows.append(
            _row(
                "No report indicators",
                "Caller reputation",
                "Low",
                source,
                "No known reports were found for this number.",
            )
        )

    return rows


def explain_phone_result(record: dict[str, Any], risk_result: dict[str, object]) -> dict[str, object]:
    """Return a UI-ready explanation object."""

    police_reports = to_int(record.get("police_report_count"))
    verified_reports = to_int(record.get("verified_report_count"))
    spam = to_bool(record.get("spam"))
    fraud = to_bool(record.get("fraud"))
    source = str(record.get("source") or "unknown").strip()
    provider = str(record.get("provider") or "").strip()
    found = record.get("found") is not False
    risk_level = str(risk_result.get("risk_level", "Unknown"))

    if not found:
        summary = (
            "This number was not found in configured live provider evidence. "
            "This does not guarantee the number is safe."
        )
    elif provider == "veriphone":
        valid = record.get("valid")
        valid_false = valid is False or str(valid).strip().lower() == "false"
        carrier = str(record.get("carrier") or "").strip()
        line_type = str(record.get("line_type") or "").strip()
        if valid_false:
            summary = "Carrier lookup returned metadata indicating the number may be invalid."
        elif carrier or line_type:
            details = ", ".join(value for value in [carrier, line_type] if value)
            summary = f"Carrier lookup returned number metadata: {details}."
        else:
            summary = "Carrier lookup returned limited metadata. This does not prove the caller is safe."
    elif fraud:
        summary = "The reputation data contains a direct fraud flag for this phone number."
    elif police_reports or verified_reports:
        summary = (
            f"The lookup found {police_reports} police report(s) and "
            f"{verified_reports} verified community report(s)."
        )
    elif risk_level == "Low Risk":
        summary = "No scam reports were found and the caller reputation appears lower risk."
    else:
        summary = "The number has limited reputation evidence and should be manually verified."

    return {
        "summary": summary,
        "recommended_action": str(risk_result.get("recommended_action", "")),
        "indicators": build_phone_evidence_rows(record, risk_result),
        "metrics": {
            "Risk Level": risk_level,
            "Police Reports": police_reports,
            "Verified Reports": verified_reports,
            "Spam": _yes_no(spam),
            "Fraud": _yes_no(fraud),
            "Business Status": _business_status(record),
            "Source": source,
            "Provider": provider or "local_fallback",
            "Fraud Score": record.get("fraud_score", "N/A"),
            "Line Type": record.get("line_type", "N/A"),
            "Carrier": record.get("carrier", "N/A"),
            "Country": record.get("country", "N/A"),
        },
    }
