"""Phone-number reputation lookup page."""

from __future__ import annotations

from datetime import datetime
import html
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    render_analysis_ready,
    render_content_card_close,
    render_content_card_open,
    render_detection_tool_intro,
    render_info_banner,
    render_result_card,
    render_section_header,
)
from src.phone.omkar_client import lookup_omkar_phone
from src.phone.phone_lookup import lookup_phone, normalise_phone_query, validate_phone_query
from src.phone.phone_number import format_phone_for_omkar
from src.phone.providers import (
    lookup_omkar_metadata,
    lookup_penipumy_reputation,
    test_omkar_connection,
    test_penipumy_connection,
)
from src.phone.providers.models import diagnostic_rows
from src.reporting.history_db import record_history_item


def _secret_value(*keys: str) -> str:
    for key in keys:
        try:
            value = str(st.secrets.get(key, "")).strip()
            if value:
                return value
        except Exception:
            pass
    return ""


def _section_secret(section_name: str, *field_names: str) -> str:
    try:
        section = st.secrets.get(section_name, {})
    except Exception:
        return ""

    for field_name in field_names:
        try:
            value = str(section.get(field_name, "")).strip()
            if value:
                return value
        except Exception:
            pass
    return ""


def _masked_key(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "Not available"
    suffix = value[-4:] if len(value) >= 4 else value
    return f"********{suffix}"


def _key_result(key: str, source: str, variable: str) -> dict[str, object]:
    key = str(key or "").strip()
    return {
        "configured": bool(key),
        "key": key,
        "source": source if key else "Not configured",
        "variable": variable if key else "-",
        "masked_key": _masked_key(key),
    }


def _resolve_api_key(provider: str) -> dict[str, object]:
    """Resolve provider API key and safe source metadata."""

    env_names = ["OMKAR_API_KEY", "OMKAR_CARRIER_API_KEY"]
    direct_secret_names = ["OMKAR_API_KEY", "OMKAR_CARRIER_API_KEY"]
    sections = [("omkar", "api_key"), ("carrier_lookup", "api_key")]

    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return _key_result(value, f"Environment variable: {name}", name)

    for name in direct_secret_names:
        value = _secret_value(name)
        if value:
            return _key_result(value, f"Streamlit secret: {name}", name)

    for section_name, field_name in sections:
        value = _section_secret(section_name, field_name)
        if value:
            return _key_result(value, f"Streamlit secret: [{section_name}].{field_name}", field_name)

    return _key_result("", "Not configured", "-")


def _configured_penipumy_api_key() -> str:
    """Backward-compatible helper; PenipuMY is no longer visible in the UI."""

    return ""


def _configured_ipqs_api_key() -> str:
    """Backward-compatible helper; IPQS is no longer visible in the UI."""

    return ""


def _configured_omkar_api_key() -> str:
    """Read an Omkar Carrier Lookup API key from environment variables or Streamlit secrets."""

    return str(_resolve_api_key("omkar_carrier_lookup").get("key", ""))


def _source_label(source: str) -> str:
    labels = {
        "penipumy_api": "PenipuMY API",
        "ipqualityscore_api": "IPQualityScore API",
        "omkar_carrier_lookup_api": "Omkar Carrier Lookup",
        "local_fallback": "Legacy fallback record",
        "demo_fallback": "Demo record",
        "unknown_fallback": "Unknown fallback",
        "local_processed": "Local processed dataset",
    }
    return labels.get(source, source.replace("_", " ").title())


def _provider_key(provider_label: str) -> str:
    if "IPQualityScore" in provider_label:
        return "ipqualityscore"
    if "Carrier" in provider_label or "Omkar" in provider_label:
        return "omkar_carrier_lookup"
    return "penipumy"


def _provider_label(provider_key: str) -> str:
    if provider_key == "ipqualityscore":
        return "IPQualityScore"
    if provider_key == "omkar_carrier_lookup":
        return "Omkar Carrier Lookup"
    return "PenipuMY"


def _lookup_phone_compat(
    phone_number: str,
    root: Path,
    *,
    provider: str,
    api_key: str,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """Call lookup_phone while tolerating a stale old import during Streamlit reruns."""

    try:
        signature = inspect.signature(lookup_phone)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "provider" in signature.parameters:
        kwargs: dict[str, Any] = {"provider": provider, "api_key": api_key}
        if "demo_mode" in signature.parameters:
            kwargs["demo_mode"] = demo_mode
        return lookup_phone(phone_number, root, **kwargs)

    if provider != "penipumy":
        raise ValueError(
            "The phone lookup backend loaded by Streamlit is stale. "
            "Refresh or restart Streamlit once so Omkar Carrier Lookup support is loaded."
        )

    return lookup_phone(phone_number, root, api_key=api_key)


def _render_phone_step(index: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="email-step-header">
            <span>{index}</span>
            <div>
                <h3>{title}</h3>
                <p>{body}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_api_setup(provider: str, root: Path) -> dict[str, object]:
    provider_label = _provider_label(provider)
    key_meta = _resolve_api_key(provider)
    configured_key = str(key_meta.get("key", ""))
    docs_url = "https://github.com/omkarcloud/phone-lookup-api"
    env_name = "OMKAR_API_KEY"
    help_text = "The key is sent only to Omkar Carrier Lookup through the API-Key request header."

    with st.expander(f"{provider_label} API setup", expanded=not bool(configured_key)):
        if configured_key:
            st.success(f"Using a {provider_label} API key from {key_meta.get('source')}.")
            st.caption(f"Masked key: `{key_meta.get('masked_key')}`")
        else:
            render_info_banner(
                f"No {provider_label} API key was detected. Configure a live API key before checking a number.",
                kind="warning",
                code="API KEY",
            )

        render_info_banner(
            "After registering an Omkar account, you must verify your phone number before the API key can perform live lookups.",
            kind="warning",
            code="VERIFY",
        )

        st.link_button(
            "Open Omkar Carrier Lookup API Documentation",
            docs_url,
            use_container_width=True,
        )
        st.link_button(
            "Verify Omkar Account Phone Number",
            "https://www.omkar.cloud/account/verify-phone",
            use_container_width=True,
        )

        guide_path = root / "docs" / "omkar_api_setup_guide.html"
        if guide_path.exists():
            st.download_button(
                "Download Omkar API Setup Guide",
                data=guide_path.read_bytes(),
                file_name="omkar_api_setup_guide.html",
                mime="text/html",
                use_container_width=True,
            )

        manual_key = st.text_input(
            "Session API key",
            type="password",
            placeholder=f"Paste {provider_label} API key for this session",
            help=help_text,
            key=f"phone_{provider}_session_key",
        )
        st.caption(
            f"Local setup: `$env:{env_name}='your_key_here'` before launching Streamlit. "
            "Do not commit API keys to GitHub."
        )

    manual_key = manual_key.strip()
    if manual_key:
        return _key_result(manual_key, "Session input", "session")
    return key_meta


def _is_populated(value: object) -> bool:
    return value not in (None, "", [], {}, "N/A")


def _response_field_summary(payload: dict[str, Any]) -> dict[str, int]:
    total_fields = len(payload)
    populated_fields = sum(1 for value in payload.values() if _is_populated(value))
    return {
        "total_fields": total_fields,
        "populated_fields": populated_fields,
        "empty_fields": total_fields - populated_fields,
    }


def _nested_value(payload: dict[str, Any], dotted_key: str) -> object:
    value: object = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _bool_text(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "Not returned"
    return str(value)


def _diagnostic_category(provider: str, status_code: object, error: str, payload: dict[str, Any]) -> str:
    error_text = str(error or "").lower()
    if not error_text:
        return "none"
    if "account phone verification required" in error_text or "verify your phone number" in error_text:
        return "account_phone_verification_required"
    if "missing" in error_text:
        return "missing_key"
    if status_code in {401, 403} or "rejected" in error_text or "invalid key" in error_text:
        return "invalid_key"
    if status_code == 400 and any(term in error_text for term in ("phone", "number", "verify")):
        return "invalid_phone_format"
    if status_code == 400:
        return "bad_request"
    if status_code == 429 or "rate limit" in error_text:
        return "rate_limited"
    if "insufficient credit" in error_text or "insufficient credits" in error_text:
        return "insufficient_credits"
    if "timed out" in error_text:
        return "timeout"
    if "network" in error_text or "connection" in error_text:
        return "network_error"
    if status_code in {500, 502, 503} or "server error" in error_text:
        return "server_error"
    if provider == "ipqualityscore" and payload.get("valid") is False:
        return "invalid_phone"
    omkar_valid = payload.get("is_valid_number")
    if provider == "omkar_carrier_lookup" and (
        omkar_valid is False or str(omkar_valid).strip().lower() == "false"
    ):
        return "invalid_phone"
    if "malformed" in error_text or "non-json" in error_text:
        return "malformed_response"
    return "provider_error"


def _safe_rate_limit_text(rate_limit: dict[str, Any]) -> str:
    values = [
        f"{key}={value}"
        for key, value in rate_limit.items()
        if value not in (None, "")
    ]
    return ", ".join(values) if values else "Not returned"


def _provider_evidence_rows(provider: str, payload: dict[str, Any]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()

    if provider == "ipqualityscore":
        fields = [
            ("Validation", "Valid", "valid"),
            ("Validation", "Active", "active"),
            ("Risk", "Fraud score", "fraud_score"),
            ("Risk", "Risky", "risky"),
            ("Risk", "Recent abuse", "recent_abuse"),
            ("Risk", "Spammer", "spammer"),
            ("Risk", "Leaked", "leaked"),
            ("Network", "Carrier", "carrier"),
            ("Network", "Line type", "line_type"),
            ("Network", "VoIP", "VOIP"),
            ("Network", "Prepaid", "prepaid"),
            ("Location", "Country", "country"),
            ("Location", "Region", "region"),
            ("Location", "City", "city"),
            ("Identity", "Name", "name"),
            ("Metadata", "Do not call", "do_not_call"),
            ("Metadata", "Active status", "active_status"),
            ("Metadata", "User activity", "user_activity"),
            ("Metadata", "Request ID", "request_id"),
            ("SMS Pumping", "Risk score", "sms_pumping.risk_score"),
            ("SMS Pumping", "Message", "sms_pumping.message"),
            ("SMS Pumping", "Velocity", "sms_pumping.velocity"),
        ]
    elif provider == "omkar_carrier_lookup":
        fields = [
            ("Validation", "Valid", "is_valid_number"),
            ("Network", "Carrier", "carrier"),
            ("Network", "Line type", "line_type"),
            ("Network", "Phone number", "phone_number"),
            ("Network", "National format", "national_format"),
            ("Location", "Country", "country_code"),
            ("Location", "Calling country code", "calling_country_code"),
            ("Network", "Mobile country code", "mobile_country_code"),
            ("Network", "Mobile network code", "mobile_network_code"),
        ]
    else:
        fields = [
            ("Reports", "Police report count", "police_report_count"),
            ("Reports", "Verified report count", "verified_report_count"),
            ("Reputation", "Spam flag", "spam"),
            ("Reputation", "Fraud flag", "fraud"),
            ("Business", "Business name", "business.display_name"),
            ("Business", "Business tier", "business.tier"),
            ("Spoofing", "Spoofing report count", "business.spoofing_report_count"),
            ("Metadata", "Source", "police_report_status"),
        ]

    rows = []
    for category, field, key in fields:
        value = _nested_value(payload, key)
        if not _is_populated(value):
            continue
        if isinstance(value, bool):
            value = "Yes" if value else "No"
        rows.append({"Category": category, "Field": field, "Value": value})
    return pd.DataFrame(rows)


def _provider_response_statistics(provider: str, payload: dict[str, Any]) -> pd.DataFrame:
    summary = _response_field_summary(payload)
    if provider == "ipqualityscore":
        risk_fields = ["fraud_score", "recent_abuse", "risky", "spammer", "leaked", "do_not_call"]
        identity_fields = ["name", "carrier", "line_type", "active_status", "user_activity"]
        location_fields = ["country", "region", "city", "timezone", "zip_code"]
    elif provider == "omkar_carrier_lookup":
        risk_fields = ["is_valid_number", "line_type"]
        identity_fields = ["carrier", "phone_number", "national_format", "mobile_country_code", "mobile_network_code"]
        location_fields = ["country_code", "calling_country_code"]
    else:
        risk_fields = ["police_report_count", "verified_report_count", "spam", "fraud"]
        identity_fields = ["business", "phone", "police_report_status"]
        location_fields = []

    return pd.DataFrame(
        [
            {"Response Statistic": "Total top-level fields returned", "Count": summary["total_fields"]},
            {"Response Statistic": "Populated top-level fields", "Count": summary["populated_fields"]},
            {"Response Statistic": "Empty top-level fields", "Count": summary["empty_fields"]},
            {
                "Response Statistic": "Provider-specific risk fields found",
                "Count": sum(1 for key in risk_fields if _is_populated(payload.get(key))),
            },
            {
                "Response Statistic": "Identity/carrier fields found",
                "Count": sum(1 for key in identity_fields if _is_populated(payload.get(key))),
            },
            {
                "Response Statistic": "Location fields found",
                "Count": sum(1 for key in location_fields if _is_populated(payload.get(key))),
            },
        ]
    )


def _has_value(record: dict[str, Any], *keys: str) -> bool:
    return any(_is_populated(record.get(key)) for key in keys)


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _has_reputation_evidence(record: dict[str, Any]) -> bool:
    return any(
        [
            _safe_int(record.get("police_report_count")) > 0,
            _safe_int(record.get("verified_report_count")) > 0,
            _safe_int(record.get("spoofing_report_count")) > 0,
            _safe_bool(record.get("spam")),
            _safe_bool(record.get("fraud")),
        ]
    )


def _evidence_coverage_chart(record: dict[str, Any]) -> go.Figure:
    categories = [
        ("Number validity", _has_value(record, "valid")),
        ("Carrier information", _has_value(record, "carrier")),
        ("Line type", _has_value(record, "line_type")),
        ("Country information", _has_value(record, "country", "calling_country_code")),
        ("Formatting metadata", _has_value(record, "formatted", "national_format", "phone")),
        ("Owner identity", _has_value(record, "business_name", "name")),
        ("Activity status", _has_value(record, "active", "active_status")),
        ("Scam reputation", _has_reputation_evidence(record)),
    ]
    labels = [label for label, _available in categories]
    values = [100 if available else 0 for _label, available in categories]
    colors = ["#38BDF8" if value else "#334155" for value in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=["Available" if value else "Unavailable" for value in values],
            textposition="auto",
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Evidence Availability",
        height=340,
        margin=dict(l=10, r=20, t=45, b=25),
        xaxis=dict(range=[0, 100], title="Metadata coverage, not risk"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return apply_chart_theme(fig)


def _claim_country_status(record: dict[str, Any], claimed_identity: str) -> tuple[str, int]:
    claim = claimed_identity.lower()
    country = str(record.get("country") or "").strip().lower()
    country_aliases = {
        "malaysia": {"my", "malaysia"},
        "malaysian": {"my", "malaysia"},
        "singapore": {"sg", "singapore"},
        "indonesia": {"id", "indonesia"},
        "thailand": {"th", "thailand"},
    }
    expected = next((aliases for token, aliases in country_aliases.items() if token in claim), None)
    if not expected:
        return "Unknown", 0
    if not country:
        return "Unknown", 0
    return ("Match", 100) if country in expected else ("Review", 50)


def _line_type_status(record: dict[str, Any], claimed_identity: str) -> tuple[str, int]:
    claim = claimed_identity.lower()
    line_type = str(record.get("line_type") or "").strip().lower()
    if not line_type:
        return "Unknown", 0
    formal_claim = any(term in claim for term in ("bank", "government", "department", "office", "landline", "official"))
    if formal_claim and any(term in line_type for term in ("voip", "mobile", "prepaid")):
        return "Review", 50
    return "Match", 100


def _caller_claim_consistency_chart(record: dict[str, Any], claimed_identity: str) -> go.Figure | None:
    if not claimed_identity.strip():
        return None

    country_label, country_value = _claim_country_status(record, claimed_identity)
    line_label, line_value = _line_type_status(record, claimed_identity)
    checks = [
        ("Country matches claim", country_label, country_value),
        ("Carrier metadata available", "Match" if _has_value(record, "carrier") else "Unknown", 100 if _has_value(record, "carrier") else 0),
        ("Line type expected", line_label, line_value),
        ("Number format valid", "Match" if record.get("valid") is True else ("Review" if record.get("valid") is False else "Unknown"), 100 if record.get("valid") is True else (50 if record.get("valid") is False else 0)),
        ("Business identity confirmed", "Match" if _has_value(record, "business_name") else "Unknown", 100 if _has_value(record, "business_name") else 0),
        ("Reputation evidence available", "Match" if _has_reputation_evidence(record) else "Unknown", 100 if _has_reputation_evidence(record) else 0),
    ]
    labels = [item[0] for item in checks]
    states = [item[1] for item in checks]
    values = [item[2] for item in checks]
    color_map = {"Match": "#22C55E", "Review": "#F59E0B", "Unknown": "#334155"}

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[color_map.get(state, "#334155") for state in states],
            text=states,
            textposition="auto",
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Caller Claim Consistency",
        height=300,
        margin=dict(l=10, r=20, t=45, b=25),
        xaxis=dict(range=[0, 100], title="Rule-based consistency checks, not an AI probability"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return apply_chart_theme(fig)


def _response_completeness_chart(record: dict[str, Any]) -> go.Figure:
    omkar_supported_fields = [
        "valid",
        "carrier",
        "line_type",
        "country",
        "formatted",
        "national_format",
        "calling_country_code",
        "mobile_country_code",
        "mobile_network_code",
    ]
    not_supplied_fields = [
        "owner_identity",
        "activity_status",
        "scam_reputation",
        "police_report_count",
        "verified_report_count",
        "fraud_score",
    ]
    populated = sum(1 for key in omkar_supported_fields if _is_populated(record.get(key)))
    empty = len(omkar_supported_fields) - populated
    unsupported = len(not_supplied_fields)

    fig = go.Figure(
        go.Bar(
            x=["Populated fields", "Empty fields", "Not supplied by provider"],
            y=[populated, empty, unsupported],
            marker_color=["#38BDF8", "#F59E0B", "#64748B"],
            text=[populated, empty, unsupported],
            textposition="auto",
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Provider Response Completeness",
        height=300,
        margin=dict(l=10, r=20, t=45, b=35),
        yaxis_title="Field count",
        showlegend=False,
    )
    return apply_chart_theme(fig)


def _session_lookup_history_chart(history: list[dict[str, object]]) -> go.Figure | None:
    phone_rows = [item for item in history if str(item.get("type", "")).lower() == "phone"]
    if len(phone_rows) <= 1:
        return None

    live = sum(1 for item in phone_rows if "omkar" in str(item.get("source", "")).lower() or "carrier lookup" in str(item.get("source", "")).lower())
    non_live = sum(1 for item in phone_rows if "fallback" in str(item.get("source", "")).lower() or "unknown" in str(item.get("source", "")).lower())
    unknown = sum(1 for item in phone_rows if "unknown" in str(item.get("source", "")).lower() or str(item.get("prediction", "")).lower() == "unknown")
    provider_failures = max(0, non_live + unknown)

    fig = go.Figure(
        go.Bar(
            x=["Live API successes", "Non-live results", "Unknown results", "Provider failures"],
            y=[live, non_live, unknown, provider_failures],
            marker_color=["#38BDF8", "#F59E0B", "#64748B", "#EF4444"],
            text=[live, non_live, unknown, provider_failures],
            textposition="auto",
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Session Lookup History",
        height=280,
        margin=dict(l=10, r=20, t=45, b=35),
        yaxis_title="Lookup count",
        showlegend=False,
    )
    return apply_chart_theme(fig)


def _run_provider_connection_test(provider: str, phone_number: str, key_meta: dict[str, object]) -> dict[str, Any]:
    api_key = str(key_meta.get("key", "") or "")
    configured = bool(api_key)
    normalized = normalise_phone_query(phone_number)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "provider": provider,
        "configured": configured,
        "configuration_source": str(key_meta.get("source", "Not configured")),
        "key_variable": str(key_meta.get("variable", "-")),
        "connected": False,
        "authenticated": False,
        "http_status": None,
        "provider_success": None,
        "response_time_ms": 0.0,
        "request_id": "",
        "rate_limit": {},
        "total_fields": 0,
        "populated_fields": 0,
        "empty_fields": 0,
        "fallback_used": False,
        "error": None,
        "error_category": "not_tested",
        "payload": {},
    }

    ok, validation_message = validate_phone_query(phone_number)
    if not ok:
        result["error"] = validation_message
        result["error_category"] = "invalid_phone"
        return result
    if not configured:
        result["error"] = f"{_provider_label(provider)} API key is not configured."
        result["error_category"] = "missing_key"
        return result

    live = lookup_omkar_phone(format_phone_for_omkar(normalized), api_key, timeout=10.0)
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = dict(live.get("record", {})) if isinstance(live.get("record"), dict) else {}
    field_summary = _response_field_summary(payload)
    result.update(
        {
            "connected": live.get("status_code") is not None,
            "authenticated": bool(live.get("ok")),
            "http_status": live.get("status_code"),
            "provider_success": payload.get("is_valid_number"),
            "response_time_ms": elapsed_ms,
            "request_id": str(payload.get("request_id") or ""),
            "rate_limit": dict(live.get("rate_limit", {})),
            "total_fields": field_summary["total_fields"],
            "populated_fields": field_summary["populated_fields"],
            "empty_fields": field_summary["empty_fields"],
            "error": live.get("error"),
            "payload": payload,
        }
    )
    result["error_category"] = _diagnostic_category(
        provider,
        result["http_status"],
        str(result["error"] or ""),
        payload,
    )
    return result


def _diagnostic_rows(diagnostic: dict[str, Any]) -> pd.DataFrame:
    success = "Yes"
    failure = "No"
    status_code = diagnostic.get("http_status")
    configured = bool(diagnostic.get("configured"))
    connected = bool(diagnostic.get("connected"))
    error_category = str(diagnostic.get("error_category") or "")
    if status_code in {401, 403} or error_category == "invalid_key":
        authentication_result = failure
        authentication_detail = "Provider rejected the API key"
    elif configured and connected:
        authentication_result = "Not rejected"
        authentication_detail = "No 401/403 authentication error returned"
    elif configured:
        authentication_result = "Not evaluated"
        authentication_detail = "No provider response"
    else:
        authentication_result = failure
        authentication_detail = "API key missing"

    if status_code is None:
        request_result = "Not evaluated"
    elif int(status_code) == 200:
        request_result = success
    else:
        request_result = failure

    return pd.DataFrame(
        [
            {"Check": "Live provider", "Result": _provider_label(str(diagnostic.get("provider"))), "Detail": "Omkar Carrier Lookup"},
            {"Check": "API key detected", "Result": success if diagnostic.get("configured") else failure, "Detail": str(diagnostic.get("configuration_source", ""))},
            {"Check": "Key variable", "Result": str(diagnostic.get("key_variable", "-")), "Detail": "Value hidden"},
            {"Check": "Provider reachable", "Result": "Success" if diagnostic.get("connected") else failure, "Detail": f"Status {diagnostic.get('http_status') or 'N/A'}"},
            {"Check": "Authentication", "Result": authentication_result, "Detail": authentication_detail},
            {"Check": "Request accepted", "Result": request_result, "Detail": str(diagnostic.get("error_category") or "")},
            {"Check": "HTTP status", "Result": str(diagnostic.get("http_status") or "N/A"), "Detail": "Live request" if diagnostic.get("response_time_ms") else "Not tested"},
            {"Check": "Provider success", "Result": _bool_text(diagnostic.get("provider_success")), "Detail": "Provider response field"},
            {"Check": "Response time", "Result": f"{float(diagnostic.get('response_time_ms') or 0):.0f} ms", "Detail": "Live request"},
            {"Check": "Fields returned", "Result": str(diagnostic.get("total_fields", 0)), "Detail": "Top-level response fields"},
            {"Check": "Fields populated", "Result": str(diagnostic.get("populated_fields", 0)), "Detail": "Non-empty top-level fields"},
            {"Check": "Rate limit", "Result": _safe_rate_limit_text(dict(diagnostic.get("rate_limit", {}))), "Detail": "Provider headers"},
            {"Check": "Fallback used", "Result": success if diagnostic.get("fallback_used") else failure, "Detail": "Connection check only"},
            {"Check": "Request ID", "Result": "Present" if diagnostic.get("request_id") else "Not returned", "Detail": str(diagnostic.get("request_id") or "-")},
            {"Check": "Error", "Result": str(diagnostic.get("error_category") or "none"), "Detail": str(diagnostic.get("error") or "-")},
        ]
    )


def _render_provider_connection_check(provider: str, key_meta: dict[str, object]) -> None:
    render_section_header(
        "Omkar connection check",
        "Test whether Omkar Carrier Lookup is configured, reachable, authenticated, and returning usable fields.",
        "API diagnostics",
    )
    render_content_card_open("violet")
    st.caption(
        "This test makes one live request only when you click the button. It never displays API keys, request headers, or provider request URLs."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Item": "Live provider", "Value": _provider_label(provider)},
                {"Item": "Configuration source", "Value": key_meta.get("source", "Not configured")},
                {"Item": "API key status", "Value": "Detected" if key_meta.get("configured") else "Not configured"},
                {"Item": "Masked key", "Value": key_meta.get("masked_key", "Not available")},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    default_number = "016-240 4384"
    test_number = st.text_input(
        "Connection test phone number",
        value=st.session_state.get("phone_connection_test_number", default_number),
        key="phone_connection_test_number",
        help="Use a number you are comfortable sending to Omkar Carrier Lookup for testing.",
    )

    if st.button("Test Provider Connection", use_container_width=True, key="phone_test_provider_connection"):
        with st.spinner("Testing provider connection..."):
            diagnostic = _run_provider_connection_test(provider, test_number, key_meta)
        st.session_state["phone_provider_diagnostic"] = diagnostic

    diagnostic = st.session_state.get("phone_provider_diagnostic")
    if isinstance(diagnostic, dict) and diagnostic.get("provider") == provider:
        if diagnostic.get("authenticated") and diagnostic.get("populated_fields", 0):
            render_analysis_ready("Provider connection successful")
        elif diagnostic.get("error_category") == "account_phone_verification_required":
            render_info_banner(
                "Carrier Lookup is reachable, but the Omkar account must verify a phone number before "
                "free-plan lookups are enabled. Open https://www.omkar.cloud/account/verify-phone, "
                "complete verification, then test again.",
                kind="warning",
                code="VERIFY ACCOUNT",
            )
        elif diagnostic.get("configured"):
            render_info_banner(
                f"Provider test completed with status: {diagnostic.get('error_category')}. "
                "Fix the live provider response before running lookup.",
                kind="warning",
                code="DIAGNOSTIC",
            )
        else:
            render_info_banner(
                "No API key was configured. Add a live provider API key before running lookup.",
                kind="warning",
                code="MISSING KEY",
            )

        st.dataframe(_diagnostic_rows(diagnostic), hide_index=True, use_container_width=True)

        payload = dict(diagnostic.get("payload", {}))
        evidence = _provider_evidence_rows(provider, payload)
        if not evidence.empty:
            st.markdown("**Provider response summary**")
            st.dataframe(evidence, hide_index=True, use_container_width=True)

            st.markdown("**Current response statistics**")
            st.dataframe(
                _provider_response_statistics(provider, payload),
                hide_index=True,
                use_container_width=True,
            )
        elif diagnostic.get("error"):
            st.caption("Provider response fields are unavailable because the live test did not return usable data.")
    else:
        st.caption("Run the provider connection test to verify the selected API key and response fields.")

    render_content_card_close()


def _business_rows(record: dict[str, Any]) -> pd.DataFrame:
    business = record.get("business") if isinstance(record.get("business"), dict) else None
    if business:
        fields = [
            ("Display Name", business.get("display_name")),
            ("Tier", business.get("tier")),
            ("Brand", business.get("brand_name")),
            ("Branch", business.get("branch_name")),
            ("Address", business.get("address")),
            ("Website", business.get("website")),
            ("Maps Place", business.get("place_url")),
            ("Rating", business.get("rating")),
            ("Review Count", business.get("review_count")),
            ("Opening Hours", business.get("opening_hours_status")),
            ("Scam Advisory", business.get("scam_alert_banner")),
            ("Spoofing Reports", business.get("spoofing_report_count")),
        ]
    else:
        fields = [
            ("Business Name", record.get("business_name")),
            ("Business Tier", record.get("business_tier")),
            ("Source", record.get("source")),
            ("Record Type", record.get("record_type")),
            ("Demo Record", "Yes" if record.get("is_demo") else ""),
            ("Source Reference", record.get("source_reference")),
            ("Last Verified", record.get("last_verified")),
        ]

    return pd.DataFrame(
        [{"Field": field, "Value": value} for field, value in fields if value not in (None, "")]
    )


def _record_phone_result(
    history: list[dict[str, object]],
    phone_number: str,
    result: dict[str, Any],
) -> None:
    risk = dict(result.get("risk", {}))
    explanation = dict(result.get("explanation", {}))
    indicators = explanation.get("indicators", [])
    flags = ", ".join(
        str(row.get("Indicator", ""))
        for row in indicators
        if isinstance(row, dict) and row.get("Indicator")
    )

    record_history_item(
        history,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Phone",
            "prediction": risk.get("risk_level", "Unknown"),
            "confidence": float(risk.get("risk_score", 0)),
            "model": f"{_provider_label(str(result.get('requested_provider', 'omkar_carrier_lookup')))} reputation rules",
            "source": _source_label(str(result.get("source", ""))),
            "preview": phone_number,
            "flags": flags,
            "explanation": explanation.get("summary", ""),
            "raw_input": json.dumps(result, ensure_ascii=True, default=str),
            "is_demo": bool(result.get("is_demo") or dict(result.get("record", {})).get("is_demo")),
        },
    )


def _render_result(root: Path, lookup_result: dict[str, Any], claimed_identity: str, history: list[dict[str, object]]) -> None:
    record = dict(lookup_result.get("record", {}))
    risk = dict(lookup_result.get("risk", {}))
    explanation = dict(lookup_result.get("explanation", {}))
    metrics = dict(explanation.get("metrics", {}))
    source = str(lookup_result.get("source", "unknown_fallback"))
    requested_provider = str(lookup_result.get("requested_provider", "omkar_carrier_lookup"))
    fallback_reason = str(lookup_result.get("fallback_reason") or "")
    phone = str(record.get("phone") or "")
    is_demo = bool(lookup_result.get("is_demo") or record.get("is_demo"))
    score = float(risk.get("risk_score", 0))
    risk_level = str(risk.get("risk_level", "Unknown"))
    is_omkar_result = source == "omkar_carrier_lookup_api"
    fallback_used = source != "omkar_carrier_lookup_api"
    reputation_available = _has_reputation_evidence(record)
    metadata_available = _has_value(record, "carrier", "line_type", "valid", "country", "formatted", "national_format")

    render_analysis_ready("Phone lookup complete - results ready below")

    if is_omkar_result and risk_level == "Unknown":
        render_info_banner(
            "Omkar returned carrier metadata. This confirms lookup context only; it does not provide scam reports or prove the caller is safe.",
            kind="info",
            code="METADATA",
        )
    elif risk_level == "Unknown":
        render_info_banner(
            "This number was not found in the live API or local dataset. This does not guarantee the number is safe.",
            kind="warning",
            code="UNKNOWN",
        )
    else:
        if is_demo:
            render_info_banner(
                "Demonstration result - this is fictional sample data for presentation only, not a real-world reputation record.",
                kind="warning",
                code="DEMO",
            )
        render_result_card(
            "Caller reputation result",
            score,
            str(explanation.get("summary", "No explanation returned.")),
        )

    _render_phone_step("03", "Review Evidence", "Inspect source, provider status, metadata, reputation evidence, and recommended action.")

    render_content_card_open("violet")
    cols = st.columns(3)
    cols[0].metric("Risk Level", risk_level)
    cols[1].metric("Source Used", _source_label(source))
    if is_omkar_result:
        cols[2].metric("Line Type", str(record.get("line_type") or "N/A"))
    else:
        cols[2].metric("Reports Found", _safe_int(record.get("police_report_count")) + _safe_int(record.get("verified_report_count")))

    cols = st.columns(3)
    if is_omkar_result:
        cols[0].metric("Carrier", str(record.get("carrier") or "N/A"))
    else:
        cols[0].metric("Business Status", metrics.get("Business Status", "No match"))
    cols[1].metric("Fallback Status", "No fallback" if not fallback_used else "Fallback used")
    cols[2].metric("Review Required", "Yes" if risk.get("review_required", True) else "No")

    if fallback_reason:
        render_info_banner(
            f"Live API unavailable or not configured. Reason: {fallback_reason}. "
            f"Using {_source_label(source).lower()} instead.",
            kind="warning",
            code="FALLBACK",
        )

    if is_demo:
        st.caption(
            "Demo provenance: "
            f"{record.get('source_reference') or 'synthetic demo record'}"
            f" | Last verified: {record.get('last_verified') or 'not applicable'}"
        )

    if claimed_identity.strip():
        st.caption(f"Claimed caller identity: {claimed_identity.strip()}")

    st.markdown("**Lookup provenance**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Item": "Live provider", "Value": "Omkar Carrier Lookup"},
                {"Item": "Fallback used", "Value": "Yes" if fallback_used else "No"},
                {
                    "Item": "Provider returned",
                    "Value": "Carrier or validation metadata" if metadata_available else "No usable carrier metadata",
                },
                {"Item": "Scam reputation available", "Value": "Yes" if reputation_available else "No"},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(_evidence_coverage_chart(record), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(_response_completeness_chart(record), use_container_width=True)

    claim_chart = _caller_claim_consistency_chart(record, claimed_identity)
    if claim_chart is not None:
        st.plotly_chart(claim_chart, use_container_width=True)
        st.caption("Caller Claim Consistency uses transparent rules only. It is not an AI probability and does not modify the final lookup result.")

    chart_history = [
        {"type": "Phone", "source": _source_label(source), "prediction": risk_level},
        *history,
    ]
    history_chart = _session_lookup_history_chart(chart_history)
    if history_chart is not None:
        st.plotly_chart(history_chart, use_container_width=True)

    st.markdown("**Explanation**")
    st.write(explanation.get("summary", "No explanation returned."))
    st.markdown("**Recommended Action**")
    st.write(explanation.get("recommended_action", risk.get("recommended_action", "")))

    indicators = explanation.get("indicators", [])
    if indicators:
        st.dataframe(pd.DataFrame(indicators), hide_index=True, use_container_width=True)
    else:
        st.info("No caller reputation indicators were returned.")

    rate_limit = dict(lookup_result.get("rate_limit", {}))
    if rate_limit.get("limit") or rate_limit.get("remaining"):
        st.caption(
            f"{_provider_label(requested_provider)} rate limit: {rate_limit.get('remaining') or '?'} request(s) remaining "
            f"out of {rate_limit.get('limit') or '?'} today."
        )

    render_content_card_close()

    business_df = _business_rows(record)
    if not business_df.empty:
        render_section_header(
            "Caller identity details",
            "Business identity fields associated with the queried number.",
            "Business status",
        )
        render_content_card_open("green")
        st.dataframe(business_df, hide_index=True, use_container_width=True)
        render_content_card_close()

    with st.expander("Normalized lookup object", expanded=False):
        st.code(json.dumps(lookup_result, indent=2, ensure_ascii=False, default=str), language="json")

    _record_phone_result(history, phone, lookup_result)


PHONE_ACCENT = "#F97316"


def _inject_phone_input_css() -> None:
    st.markdown(
        """
        <style>
        .phone-workflow-shell {
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            padding: 1.05rem;
            background:
                radial-gradient(circle at 78% 0%, rgba(249,115,22,.10), transparent 22rem),
                rgba(15, 23, 42, 0.40);
        }

        .phone-provider-heading {
            display: flex;
            align-items: center;
            gap: .72rem;
            margin-bottom: .6rem;
        }

        .phone-provider-heading iconify-icon {
            color: #F97316;
            font-size: 2rem;
        }

        .phone-provider-heading strong {
            display: block;
            color: #F8FAFC;
            font-size: .98rem;
        }

        .phone-provider-heading span {
            display: block;
            color: #94A3B8;
            font-size: .78rem;
            margin-top: .08rem;
        }

        .phone-status-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: .25rem .62rem;
            font-size: .76rem;
            font-weight: 800;
            color: #FDBA74;
            border: 1px solid rgba(249,115,22,.30);
            background: rgba(249,115,22,.10);
            margin: .2rem 0 .7rem;
        }

        .phone-normalized-preview {
            border: 1px solid rgba(34,197,94,.24);
            border-radius: 14px;
            padding: .95rem 1rem;
            background: rgba(34,197,94,.08);
            min-height: 6rem;
        }

        .phone-normalized-preview span {
            display: block;
            color: #94A3B8;
            font-size: .78rem;
            margin-bottom: .35rem;
        }

        .phone-normalized-preview strong {
            color: #4ADE80;
            font-size: 1.2rem;
        }

        .st-key-phone_investigate_button button {
            min-height: 3rem !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #F97316, #EA580C) !important;
            border: 1px solid rgba(251,146,60,.42) !important;
            color: #FFF7ED !important;
            font-weight: 850 !important;
        }

        .st-key-phone_investigate_button button:disabled {
            opacity: .48 !important;
            cursor: not-allowed !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_phone_input_state() -> None:
    defaults: dict[str, Any] = {
        "phone_omkar_enabled": True,
        "phone_penipumy_enabled": True,
        "phone_omkar_api_key": "",
        "phone_penipumy_api_key": "",
        "phone_omkar_test_number": "016-240 4384",
        "phone_penipumy_test_number": "016-240 4384",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _phone_widget_key(session_key: str, suffix: str) -> str:
    return f"_{session_key}_{suffix}"


def _hydrate_phone_widget_from_session(widget_key: str, session_key: str) -> None:
    session_value = str(st.session_state.get(session_key, "") or "")
    if session_value and not st.session_state.get(widget_key):
        st.session_state[widget_key] = session_value


def _sync_phone_provider_enabled(
    widget_key: str,
    enabled_key: str,
    session_key: str,
    api_widget_key: str,
    diagnostic_key: str,
) -> None:
    enabled = bool(st.session_state.get(widget_key, True))
    st.session_state[enabled_key] = enabled
    if not enabled:
        st.session_state[session_key] = ""
        st.session_state[api_widget_key] = ""
        st.session_state.pop(diagnostic_key, None)


def _sync_phone_session_api_key(widget_key: str, session_key: str, diagnostic_key: str) -> None:
    value = str(st.session_state.get(widget_key, "") or "").strip()
    previous = str(st.session_state.get(session_key, "") or "").strip()
    st.session_state[session_key] = value
    if value != previous:
        st.session_state.pop(diagnostic_key, None)


def _phone_provider_key_config(provider_id: str) -> dict[str, object]:
    if provider_id == "penipumy":
        return {
            "env_names": ["PENIPUMY_API_KEY", "PENIPU_API_KEY"],
            "secret_names": ["PENIPUMY_API_KEY", "PENIPU_API_KEY"],
            "sections": [("penipumy", "api_key"), ("penipu", "api_key")],
        }
    return {
        "env_names": ["OMKAR_API_KEY", "OMKAR_CARRIER_API_KEY"],
        "secret_names": ["OMKAR_API_KEY", "OMKAR_CARRIER_API_KEY"],
        "sections": [("omkar", "api_key"), ("carrier_lookup", "api_key")],
    }


def _resolve_phone_provider_key(provider_id: str, session_value: str = "") -> dict[str, object]:
    config = _phone_provider_key_config(provider_id)
    session_value = str(session_value or "").strip()
    if session_value:
        return _key_result(session_value, "Session input", "session")

    for name in list(config["env_names"]):
        value = os.environ.get(str(name), "").strip()
        if value:
            return _key_result(value, f"Environment variable: {name}", str(name))

    for name in list(config["secret_names"]):
        value = _secret_value(str(name))
        if value:
            return _key_result(value, f"Streamlit secret: {name}", str(name))

    for section_name, field_name in list(config["sections"]):
        value = _section_secret(str(section_name), str(field_name))
        if value:
            return _key_result(value, f"Streamlit secret: [{section_name}].{field_name}", str(field_name))

    return _key_result("", "Not configured", "-")


def _status_chip(label: str) -> None:
    st.markdown(
        f'<div class="phone-status-chip">{html.escape(label)}</div>',
        unsafe_allow_html=True,
    )


def _provider_status_label(
    enabled: bool,
    key_meta: dict[str, object],
    diagnostic: dict[str, Any] | None,
) -> str:
    if not enabled:
        return "Disabled"
    if diagnostic:
        error_code = str(diagnostic.get("error_code") or "none")
        if error_code == "none" and diagnostic.get("provider_success") is not False:
            return "Tested successfully"
        if error_code == "missing_key":
            return "Not configured"
        if error_code == "authentication_failed":
            return "Authentication rejected"
        if error_code == "rate_limited":
            return "Rate limited"
        return "Connection failed"
    return "Ready" if key_meta.get("configured") else "Not configured"


def _render_diagnostics_expander(title: str, diagnostic_key: str) -> None:
    diagnostic = st.session_state.get(diagnostic_key)
    if not isinstance(diagnostic, dict):
        st.caption("Run the connection test to view provider diagnostics.")
        return

    with st.expander(title, expanded=False):
        st.dataframe(
            pd.DataFrame(diagnostic_rows(diagnostic)),
            hide_index=True,
            use_container_width=True,
        )
        fields = diagnostic.get("raw_field_names") or []
        if fields:
            st.caption(f"Fields returned: {', '.join(str(item) for item in fields)}")


def _render_live_provider_card(
    *,
    provider_id: str,
    title: str,
    purpose: str,
    icon: str,
    enabled_key: str,
    session_key: str,
    test_number_key: str,
    diagnostic_key: str,
) -> dict[str, object]:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="phone-provider-heading">
                <iconify-icon icon="{icon}"></iconify-icon>
                <div>
                    <strong>{html.escape(title)}</strong>
                    <span>{html.escape(purpose)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enabled_widget_key = _phone_widget_key(enabled_key, "toggle")
        if enabled_widget_key not in st.session_state:
            st.session_state[enabled_widget_key] = bool(st.session_state.get(enabled_key, True))

        api_widget_key = _phone_widget_key(session_key, "input")

        enabled = st.toggle(
            "Enable provider",
            key=enabled_widget_key,
            on_change=_sync_phone_provider_enabled,
            args=(enabled_widget_key, enabled_key, session_key, api_widget_key, diagnostic_key),
        )
        st.session_state[enabled_key] = bool(enabled)

        if not enabled:
            st.session_state[session_key] = ""
            st.session_state[api_widget_key] = ""
            st.session_state.pop(diagnostic_key, None)
        else:
            _hydrate_phone_widget_from_session(api_widget_key, session_key)

        api_widget_value = st.text_input(
            f"{title} API key",
            type="password",
            key=api_widget_key,
            disabled=not enabled,
            placeholder=f"Paste {title} API key for this session",
            on_change=_sync_phone_session_api_key,
            args=(api_widget_key, session_key, diagnostic_key),
        )
        session_value = str(api_widget_value or "").strip()
        if enabled:
            st.session_state[session_key] = session_value

        key_meta = _resolve_phone_provider_key(provider_id, session_value)
        st.caption(f"Key source: {key_meta.get('source', 'Not configured')}")

        test_number = st.text_input(
            "Test number (optional)",
            value=str(st.session_state.get(test_number_key, "016-240 4384")),
            key=test_number_key,
            disabled=not enabled,
        )

        if st.button(
            f"Test {title} Connection",
            key=f"{provider_id}_connection_test",
            use_container_width=True,
            disabled=not enabled,
        ):
            with st.spinner(f"Testing {title}..."):
                if provider_id == "penipumy":
                    diagnostic = test_penipumy_connection(
                        test_number,
                        str(key_meta.get("key", "")),
                        key_source=str(key_meta.get("source", "Not configured")),
                        key_variable=str(key_meta.get("variable", "-")),
                    )
                else:
                    diagnostic = test_omkar_connection(
                        test_number,
                        str(key_meta.get("key", "")),
                        key_source=str(key_meta.get("source", "Not configured")),
                        key_variable=str(key_meta.get("variable", "-")),
                    )
                st.session_state[diagnostic_key] = diagnostic.as_dict()

        diagnostic = st.session_state.get(diagnostic_key)
        _status_chip(_provider_status_label(enabled, key_meta, diagnostic if isinstance(diagnostic, dict) else None))
        _render_diagnostics_expander(f"View {title} Diagnostics", diagnostic_key)
        return {"enabled": enabled, "key_meta": key_meta}


def _render_status_row(
    omkar_enabled: bool,
    omkar_key: dict[str, object],
    penipu_enabled: bool,
    penipu_key: dict[str, object],
) -> None:
    rows = [
        {
            "Provider": "Omkar",
            "Status": "Ready" if omkar_enabled and omkar_key.get("configured") else "Not configured" if omkar_enabled else "Disabled",
        },
        {
            "Provider": "PenipuMY",
            "Status": "Ready" if penipu_enabled and penipu_key.get("configured") else "Not configured" if penipu_enabled else "Disabled",
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _provider_result_block(result: dict[str, Any], *, default_provider: str, enabled: bool) -> dict[str, Any]:
    return {
        "provider": default_provider,
        "enabled": enabled,
        "status": str(result.get("status", "unavailable")),
        "data": result.get("data", {}),
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
    }


def _build_phone_investigation(
    *,
    root: Path,
    raw_number: str,
    normalized_number: str,
    claimed_identity: str,
    omkar_enabled: bool,
    omkar_key: str,
    penipumy_enabled: bool,
    penipumy_key: str,
) -> dict[str, Any]:
    metadata = {
        "provider": "omkar",
        "enabled": omkar_enabled,
        "status": "not_configured" if omkar_enabled else "unavailable",
        "data": {},
        "error_code": "missing_key" if omkar_enabled and not omkar_key else None,
        "error_message": "Omkar Carrier Lookup API key is not configured." if omkar_enabled and not omkar_key else None,
    }
    reputation = {
        "provider": "penipumy",
        "enabled": penipumy_enabled,
        "status": "not_configured" if penipumy_enabled else "unavailable",
        "data": {},
        "error_code": "missing_key" if penipumy_enabled and not penipumy_key else None,
        "error_message": "PenipuMY API key is not configured." if penipumy_enabled and not penipumy_key else None,
    }
    requested = 0
    completed = 0
    failed = 0

    if omkar_enabled:
        requested += 1
        if omkar_key:
            omkar_result = lookup_omkar_metadata(normalized_number, omkar_key).as_dict()
            metadata = _provider_result_block(omkar_result, default_provider="omkar", enabled=True)
            completed += 1 if omkar_result.get("success") else 0
            failed += 0 if omkar_result.get("success") else 1

    if penipumy_enabled:
        requested += 1
        if penipumy_key:
            penipu_result = lookup_penipumy_reputation(normalized_number, penipumy_key).as_dict()
            reputation = _provider_result_block(penipu_result, default_provider="penipumy", enabled=True)
            if penipu_result.get("success"):
                completed += 1
            else:
                failed += 1

    return {
        "input": {
            "raw_number": raw_number,
            "normalized_number": normalized_number,
            "claimed_identity": claimed_identity,
        },
        "metadata": metadata,
        "reputation": reputation,
        "provider_coverage": {
            "requested": requested,
            "completed": completed,
            "failed": failed,
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


CONCERN_WEIGHTS = {
    "police_report": 8,
    "police_report_cap": 32,
    "verified_report": 10,
    "verified_report_cap": 30,
    "fraud_flag": 25,
    "spam_flag": 15,
    "spoofing_report": 5,
    "spoofing_report_cap": 20,
    "scam_advisory": 10,
    "multiple_categories": 10,
    "recent_reports": 10,
    "claimed_identity_unverified": 10,
    "invalid_number": 15,
}


OMKAR_PROFILE_FIELDS = {
    "Number": [
        ("Number status", "valid"),
        ("Phone number", "phone_number"),
        ("National format", "national_format"),
    ],
    "Network": [
        ("Carrier", "carrier"),
        ("Line type", "line_type"),
        ("Mobile country code", "mobile_country_code"),
        ("Mobile network code", "mobile_network_code"),
    ],
    "Region": [
        ("Country code", "country_code"),
        ("Calling country code", "calling_country_code"),
    ],
}


PENIPUMY_REPUTATION_FIELDS = [
    ("Police reports", "police_report_count"),
    ("Verified reports", "verified_report_count"),
    ("Fraud flag", "fraud"),
    ("Spam flag", "spam"),
    ("Police report status", "police_report_status"),
    ("Spoofing reports", "spoofing_report_count"),
]


BUSINESS_RECORD_FIELDS = [
    ("Display name", "display_name"),
    ("Business tier", "tier"),
    ("Brand", "brand_name"),
    ("Branch", "branch_name"),
    ("Address", "address"),
    ("Website", "website"),
    ("Maps listing", "place_url"),
    ("Rating", "rating"),
    ("Reviews", "review_count"),
    ("Opening status", "opening_hours_status"),
    ("Scam advisory", "scam_alert_banner"),
    ("Spoofing reports", "spoofing_report_count"),
]


OUTPUT_FIELD_OWNERS = {
    "raw_number": "input",
    "normalized_number": "input",
    "claimed_identity": "input",
    "is_valid_number": "omkar",
    "phone_number": "omkar",
    "national_format": "omkar",
    "country_code": "omkar",
    "calling_country_code": "omkar",
    "carrier": "omkar",
    "line_type": "omkar",
    "mobile_country_code": "omkar",
    "mobile_network_code": "omkar",
    "police_report_count": "penipumy",
    "verified_report_count": "penipumy",
    "spam": "penipumy",
    "fraud": "penipumy",
    "police_report_status": "penipumy",
    "business": "penipumy",
    "spoofing_report_count": "penipumy",
}


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if isinstance(value, str) and value.strip():
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        return [value.strip()]
    return []


def _extract_categories(data: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for key in ("categories", "report_categories", "scam_categories", "tags", "category"):
        value = data.get(key)
        for item in _as_list(value):
            text = str(item).strip()
            if text and text.lower() not in {existing.lower() for existing in categories}:
                categories.append(text)
    return categories


def _extract_report_count(data: dict[str, Any]) -> int:
    count = 0
    for key in (
        "report_count",
        "reports_count",
        "total_reports",
        "verified_report_count",
        "police_report_count",
        "complaint_count",
    ):
        count += _safe_int(data.get(key))

    reports = data.get("reports")
    if isinstance(reports, list):
        count = max(count, len(reports))

    return count


def _bool_display(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value in (None, ""):
        return ""
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return "Yes"
    if lowered in {"0", "false", "no", "n"}:
        return "No"
    return str(value)


def _profile_valid_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "valid"}:
        return True
    if lowered in {"0", "false", "no", "n", "invalid"}:
        return False
    return None


def _business_from_reputation_data(data: dict[str, Any]) -> dict[str, Any]:
    business = data.get("business")
    if isinstance(business, dict):
        return dict(business)

    direct = {
        "display_name": data.get("business_name") or data.get("display_name"),
        "tier": data.get("business_tier") or data.get("tier"),
        "spoofing_report_count": data.get("spoofing_report_count"),
    }
    return {key: value for key, value in direct.items() if _is_populated(value)}


def build_caller_output_view(investigation: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(investigation.get("metadata", {}))
    reputation = dict(investigation.get("reputation", {}))
    input_data = dict(investigation.get("input", {}))

    omkar_data = dict(metadata.get("data", {}))
    penipumy_data = dict(reputation.get("data", {}))
    reputation_source = "penipumy"
    reputation_status = str(reputation.get("status", "unavailable"))
    reputation_data = penipumy_data

    business = _business_from_reputation_data(reputation_data)
    spoofing_count = _safe_int(
        reputation_data.get("spoofing_report_count")
        or business.get("spoofing_report_count")
    )

    caller_profile = {
        "source": "omkar",
        "status": metadata.get("status", "unavailable"),
        "valid": _profile_valid_value(omkar_data.get("is_valid_number", omkar_data.get("valid"))),
        "phone_number": omkar_data.get("phone_number"),
        "national_format": omkar_data.get("national_format"),
        "country_code": omkar_data.get("country_code") or omkar_data.get("country"),
        "calling_country_code": omkar_data.get("calling_country_code"),
        "carrier": omkar_data.get("carrier"),
        "line_type": omkar_data.get("line_type"),
        "mobile_country_code": omkar_data.get("mobile_country_code"),
        "mobile_network_code": omkar_data.get("mobile_network_code"),
    }

    reputation_view = {
        "source": reputation_source,
        "status": reputation_status,
        "police_report_count": _safe_int(reputation_data.get("police_report_count")),
        "verified_report_count": _safe_int(reputation_data.get("verified_report_count")),
        "spam": reputation_data.get("spam"),
        "fraud": reputation_data.get("fraud"),
        "police_report_status": reputation_data.get("police_report_status"),
        "spoofing_report_count": spoofing_count,
        "categories": _extract_categories(reputation_data),
        "scam_alert_banner": business.get("scam_alert_banner"),
    }

    reported_identity = {
        "source": reputation_source,
        "available": bool(business),
        "display_name": business.get("display_name") or business.get("business_name"),
        "tier": business.get("tier"),
        "brand_name": business.get("brand_name"),
        "branch_name": business.get("branch_name"),
        "address": business.get("address"),
        "website": business.get("website"),
        "place_url": business.get("place_url"),
        "rating": business.get("rating"),
        "review_count": business.get("review_count"),
        "opening_hours_status": business.get("opening_hours_status"),
        "scam_alert_banner": business.get("scam_alert_banner"),
        "spoofing_report_count": spoofing_count,
    }

    return {
        "input": input_data,
        "caller_profile": caller_profile,
        "reputation": reputation_view,
        "reported_identity": reported_identity,
        "coverage": {
            "omkar": metadata.get("status", "unavailable"),
            "penipumy": reputation.get("status", "unavailable"),
        },
    }


def _combined_record_for_visuals(output_view: dict[str, Any]) -> dict[str, Any]:
    profile = dict(output_view.get("caller_profile", {}))
    reputation = dict(output_view.get("reputation", {}))
    identity = dict(output_view.get("reported_identity", {}))
    return {
        "valid": profile.get("valid"),
        "phone": profile.get("phone_number"),
        "formatted": profile.get("phone_number"),
        "national_format": profile.get("national_format"),
        "country": profile.get("country_code"),
        "calling_country_code": profile.get("calling_country_code"),
        "carrier": profile.get("carrier"),
        "line_type": profile.get("line_type"),
        "mobile_country_code": profile.get("mobile_country_code"),
        "mobile_network_code": profile.get("mobile_network_code"),
        "business_name": identity.get("display_name"),
        "police_report_count": reputation.get("police_report_count"),
        "verified_report_count": reputation.get("verified_report_count"),
        "spoofing_report_count": reputation.get("spoofing_report_count"),
        "spam": reputation.get("spam"),
        "fraud": reputation.get("fraud"),
    }


def _metadata_validity(metadata: dict[str, Any]) -> tuple[str, bool | None]:
    data = dict(metadata.get("data", {}))
    value = data.get("is_valid_number", data.get("valid"))
    parsed = _safe_bool(value) if value not in (None, "") else None
    if parsed is True:
        return "Valid number", True
    if parsed is False:
        return "Invalid number", False
    return "Validity unknown", None


def _provider_completed(investigation: dict[str, Any]) -> str:
    coverage = dict(investigation.get("provider_coverage", {}))
    requested = _safe_int(coverage.get("requested"))
    completed = _safe_int(coverage.get("completed"))
    if requested <= 0:
        return "0/0 providers completed"
    return f"{completed}/{requested} providers completed"


def _live_provider_metric(investigation: dict[str, Any]) -> str:
    coverage = dict(investigation.get("provider_coverage", {}))
    requested = _safe_int(coverage.get("requested"))
    completed = _safe_int(coverage.get("completed"))
    if requested <= 0:
        return "0/0 live providers"
    return f"{completed}/{requested} live providers"


def _phone_priority(score: int | None) -> tuple[str, str]:
    if score is None:
        return (
            "Reputation unknown",
            "There is not enough reputation evidence to calculate a concern level.",
        )
    if score >= 80:
        return (
            "Critical review",
            "End the call and verify immediately.",
        )
    if score >= 60:
        return (
            "High concern",
            "Pause the interaction and verify independently.",
        )
    if score >= 30:
        return (
            "Needs verification",
            "Verify before continuing.",
        )
    return (
        "Lower concern",
        "Continue with normal caution.",
    )


def _phone_recommendation(priority: str) -> str:
    if priority == "Critical review":
        return (
            "Community reputation evidence strongly raises concern. Do not transfer money, install software, "
            "click links, disclose sensitive information, or provide remote access."
        )
    if priority == "High concern":
        return (
            "Do not provide OTPs, passwords, banking details, or payment. End or pause the call and verify "
            "the organization using an official contact number."
        )
    if priority == "Needs verification":
        return (
            "Ask for the caller's name and department, end the call, then contact the organization using "
            "a number from its official website."
        )
    if priority == "Lower concern":
        return (
            "No strong reputation evidence was found. Unexpected requests should still be verified through "
            "an official channel."
        )
    return (
        "There was not enough reputation evidence to calculate a concern level. Verify unexpected requests "
        "independently before sharing personal, banking, OTP, or account information."
    )


def _build_phone_assessment(investigation: dict[str, Any]) -> dict[str, Any]:
    output_view = build_caller_output_view(investigation)
    metadata = {"data": dict(output_view.get("caller_profile", {}))}
    reputation = dict(investigation.get("reputation", {}))
    input_data = dict(output_view.get("input", {}))

    profile = dict(output_view.get("caller_profile", {}))
    reputation_view = dict(output_view.get("reputation", {}))
    claimed_identity = str(input_data.get("claimed_identity", "")).strip()

    contributions: list[dict[str, object]] = []
    neutral: list[dict[str, object]] = []

    police_count = _safe_int(reputation_view.get("police_report_count"))
    verified_count = _safe_int(reputation_view.get("verified_report_count"))
    spoofing_count = _safe_int(reputation_view.get("spoofing_report_count"))
    categories = [str(item) for item in _as_list(reputation_view.get("categories"))]
    fraud_flag = _safe_bool(reputation_view.get("fraud"))
    spam_flag = _safe_bool(reputation_view.get("spam"))

    if police_count > 0:
        contributions.append(
            {
                "indicator": f"{police_count} police report(s) found",
                "source": "PenipuMY",
                "points": min(
                    police_count * CONCERN_WEIGHTS["police_report"],
                    CONCERN_WEIGHTS["police_report_cap"],
                ),
                "effect": "raises_concern",
            }
        )

    if verified_count > 0:
        contributions.append(
            {
                "indicator": f"{verified_count} verified report(s) found",
                "source": "PenipuMY",
                "points": min(
                    verified_count * CONCERN_WEIGHTS["verified_report"],
                    CONCERN_WEIGHTS["verified_report_cap"],
                ),
                "effect": "raises_concern",
            }
        )

    if fraud_flag:
        contributions.append(
            {
                "indicator": "Fraud flag returned",
                "source": "PenipuMY",
                "points": CONCERN_WEIGHTS["fraud_flag"],
                "effect": "raises_concern",
            }
        )

    if spam_flag:
        contributions.append(
            {
                "indicator": "Spam flag returned",
                "source": "PenipuMY",
                "points": CONCERN_WEIGHTS["spam_flag"],
                "effect": "raises_concern",
            }
        )

    if spoofing_count > 0:
        contributions.append(
            {
                "indicator": f"{spoofing_count} spoofing report(s) found",
                "source": "PenipuMY",
                "points": min(
                    spoofing_count * CONCERN_WEIGHTS["spoofing_report"],
                    CONCERN_WEIGHTS["spoofing_report_cap"],
                ),
                "effect": "raises_concern",
            }
        )

    if _is_populated(reputation_view.get("scam_alert_banner")):
        contributions.append(
            {
                "indicator": "Scam advisory returned",
                "source": "PenipuMY",
                "points": CONCERN_WEIGHTS["scam_advisory"],
                "effect": "raises_concern",
            }
        )

    combined_report_count = police_count + verified_count + spoofing_count
    if len(categories) >= 2:
        contributions.append(
            {
                "indicator": "Multiple report categories",
                "source": "Reputation evidence",
                "points": CONCERN_WEIGHTS["multiple_categories"],
                "effect": "raises_concern",
            }
        )

    if any(
        _is_populated(dict(reputation.get("data", {})).get(key))
        for key in ("latest_report", "latest_report_date", "last_report_date")
    ):
        contributions.append(
            {
                "indicator": "Recent report detail returned",
                "source": "PenipuMY",
                "points": CONCERN_WEIGHTS["recent_reports"],
                "effect": "raises_concern",
            }
        )

    if claimed_identity:
        contributions.append(
            {
                "indicator": "Claimed identity requires checking",
                "source": "Investigation input",
                "points": CONCERN_WEIGHTS["claimed_identity_unverified"],
                "effect": "raises_concern",
            }
        )

    validity_label, is_valid = _metadata_validity(metadata)
    if is_valid is False:
        contributions.append(
            {
                "indicator": "Provider marked number invalid",
                "source": "Omkar",
                "points": CONCERN_WEIGHTS["invalid_number"],
                "effect": "raises_concern",
            }
        )
    elif is_valid is True:
        neutral.append(
            {
                "indicator": validity_label,
                "source": "Omkar",
                "points": 0,
                "effect": "neutral",
            }
        )

    if _is_populated(profile.get("carrier")):
        neutral.append(
            {
                "indicator": f"Carrier identified: {profile.get('carrier')}",
                "source": "Omkar",
                "points": 0,
                "effect": "neutral",
            }
        )

    if _is_populated(profile.get("line_type")):
        neutral.append(
            {
                "indicator": f"Line type identified: {profile.get('line_type')}",
                "source": "Omkar",
                "points": 0,
                "effect": "neutral",
            }
        )

    if _is_populated(profile.get("country_code")):
        neutral.append(
            {
                "indicator": f"Country identified: {profile.get('country_code')}",
                "source": "Omkar",
                "points": 0,
                "effect": "neutral",
            }
        )

    reputation_status = str(reputation_view.get("status") or "")
    reputation_was_checked = reputation_status in {"success", "no_match"}
    reputation_evidence_available = bool(contributions) or combined_report_count > 0
    score = min(100, sum(int(item.get("points", 0)) for item in contributions))
    score_value: int | None = score if reputation_evidence_available or reputation_was_checked or contributions else None
    priority, interpretation = _phone_priority(score_value)

    if score_value is not None and not contributions:
        priority, interpretation = _phone_priority(0)

    return {
        "score_type": "evidence_concern",
        "score": score_value,
        "priority": priority,
        "interpretation": interpretation,
        "contributions": contributions,
        "neutral_information": neutral,
        "recommended_action": _phone_recommendation(priority),
        "report_count": combined_report_count,
        "categories": categories,
    }


def _record_phone_investigation(
    history: list[dict[str, object]],
    investigation: dict[str, Any],
) -> None:
    output_view = build_caller_output_view(investigation)
    assessment = dict(investigation.get("assessment") or _build_phone_assessment(investigation))
    investigation["assessment"] = assessment

    input_data = dict(output_view.get("input", {}))
    profile = dict(output_view.get("caller_profile", {}))
    reputation = dict(output_view.get("reputation", {}))
    number = str(
        input_data.get("normalized_number")
        or input_data.get("raw_number")
        or profile.get("phone_number")
        or "Unknown phone number"
    )
    claimed_identity = str(input_data.get("claimed_identity") or "").strip()
    preview = number if not claimed_identity else f"{number} | Claimed identity: {claimed_identity}"
    flags = [
        str(item.get("indicator"))
        for item in assessment.get("contributions", [])
        if isinstance(item, dict) and item.get("indicator")
    ]
    confidence = assessment.get("score")
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    coverage = dict(output_view.get("coverage", {}))
    source_name = (
        f"Omkar: {coverage.get('omkar', 'unavailable')}; "
        f"PenipuMY: {coverage.get('penipumy', 'unavailable')}"
    )
    total_reports = (
        _safe_int(reputation.get("police_report_count"))
        + _safe_int(reputation.get("verified_report_count"))
        + _safe_int(reputation.get("spoofing_report_count"))
    )
    if total_reports:
        flags.append(f"{total_reports} total report indicator(s)")

    record_history_item(
        history,
        {
            "time": str(investigation.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "type": "Phone",
            "prediction": assessment.get("priority", "Reputation unknown"),
            "confidence": confidence_value,
            "model": "Phone provider reputation rules",
            "source_name": source_name,
            "preview": preview,
            "flags": flags,
            "explanation": assessment.get("interpretation", ""),
            "raw_input": json.dumps(investigation, ensure_ascii=True, default=str),
        },
    )


def _concern_meter(score: int | None, priority: str) -> go.Figure:
    value = 0 if score is None else score
    title = "Evidence unavailable" if score is None else f"{score} / 100"
    fig = go.Figure(
        go.Indicator(
            mode="gauge" if score is None else "gauge+number",
            value=value,
            number={"suffix": "" if score is None else " / 100", "font": {"size": 30}},
            title={"text": f"Evidence Concern<br><span style='font-size:0.8em'>{html.escape(priority)}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#F97316"},
                "bgcolor": "rgba(15,23,42,0.25)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.25)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.18)"},
                    {"range": [30, 60], "color": "rgba(250,204,21,0.20)"},
                    {"range": [60, 80], "color": "rgba(249,115,22,0.24)"},
                    {"range": [80, 100], "color": "rgba(239,68,68,0.26)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=12, r=12, t=42, b=10),
        annotations=[
            {
                "text": title,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.05,
                "showarrow": False,
                "font": {"size": 12, "color": "#94A3B8"},
            }
        ],
    )
    return apply_chart_theme(fig)


def _contribution_chart(contributions: list[dict[str, object]]) -> go.Figure | None:
    positive = [item for item in contributions if int(item.get("points", 0)) > 0]
    if not positive:
        return None

    labels = [str(item.get("indicator", "")) for item in positive]
    values = [int(item.get("points", 0)) for item in positive]
    sources = [str(item.get("source", "")) for item in positive]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#F97316",
            text=[f"+{value}" for value in values],
            textposition="auto",
            customdata=sources,
            hovertemplate="%{y}<br>Source: %{customdata}<br>Contribution: +%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(260, 52 * len(labels)),
        margin=dict(l=10, r=20, t=20, b=30),
        xaxis_title="Concern contribution",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return apply_chart_theme(fig)


def _display_value(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value in (None, "", [], {}):
        return ""
    return str(value)


def _coverage_summary(output_view: dict[str, Any]) -> str:
    coverage = dict(output_view.get("coverage", {}))
    omkar_complete = coverage.get("omkar") == "success"
    reputation_complete = coverage.get("penipumy") in {"success", "no_match"}

    if omkar_complete and reputation_complete:
        return "Metadata + reputation complete"
    if omkar_complete:
        return "Metadata complete"
    if reputation_complete:
        return "Reputation lookup complete"
    return "No provider evidence complete"


def _line_profile_summary(profile: dict[str, Any]) -> tuple[str, str]:
    if not any(_is_populated(profile.get(key)) for key in ("line_type", "carrier", "country_code", "phone_number")):
        return "Line profile unavailable", "Omkar metadata unavailable"
    title = str(profile.get("line_type") or "Line type unknown").title()
    detail_parts = [
        str(profile.get("carrier") or "").strip(),
        str(profile.get("country_code") or "").strip(),
    ]
    detail = " - ".join(part for part in detail_parts if part) or "Carrier metadata returned"
    return title, detail


def _reputation_summary(reputation: dict[str, Any]) -> tuple[str, str]:
    total = (
        _safe_int(reputation.get("police_report_count"))
        + _safe_int(reputation.get("verified_report_count"))
        + _safe_int(reputation.get("spoofing_report_count"))
    )
    if total <= 0 and not (_safe_bool(reputation.get("fraud")) or _safe_bool(reputation.get("spam"))):
        status = str(reputation.get("status") or "unavailable")
        if status == "no_match":
            return "No matching record", "This does not confirm the caller is safe"
        return "Reputation unavailable", "No live report evidence returned"

    flags = []
    if _safe_bool(reputation.get("fraud")):
        flags.append("Fraud flag")
    if _safe_bool(reputation.get("spam")):
        flags.append("Spam flag")
    flag_text = " and ".join(flags) + " found" if flags else "Report indicators found"
    return f"{total} report indicators", flag_text


def _profile_rows(profile: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group, fields in OMKAR_PROFILE_FIELDS.items():
        for label, key in fields:
            value = profile.get(key)
            if key == "valid" and value is not None:
                value = "Valid" if value is True else "Invalid"
            value_text = _display_value(value)
            if value_text:
                rows.append({"Group": group, "Field": label, "Value": value_text})
    return rows


def _reputation_rows(reputation: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, key in PENIPUMY_REPUTATION_FIELDS:
        value_text = _display_value(reputation.get(key))
        if value_text:
            rows.append({"Field": label, "Value": value_text})
    return rows


def _identity_rows(identity: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, key in BUSINESS_RECORD_FIELDS:
        value_text = _display_value(identity.get(key))
        if value_text:
            rows.append({"Field": label, "Value": value_text})
    return rows


def _render_phone_line_profile(output_view: dict[str, Any]) -> None:
    profile = dict(output_view.get("caller_profile", {}))
    rows = _profile_rows(profile)

    _render_phone_step(
        "05",
        "Phone Line Profile",
        "Omkar Carrier Lookup number and network metadata only.",
    )
    with st.container(border=True):
        st.caption("Source: Omkar Carrier Lookup")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Phone line profile metadata is unavailable.")
        st.caption(
            "Carrier metadata confirms characteristics of the number and network. "
            "It does not verify who is currently calling or whether the caller is safe."
        )


def _render_scam_reputation(output_view: dict[str, Any]) -> None:
    reputation = dict(output_view.get("reputation", {}))
    rows = _reputation_rows(reputation)
    total_reports = (
        _safe_int(reputation.get("police_report_count"))
        + _safe_int(reputation.get("verified_report_count"))
        + _safe_int(reputation.get("spoofing_report_count"))
    )

    _render_phone_step(
        "06",
        "Scam Reputation & Reports",
        "PenipuMY reputation fields, report counts, and flags only.",
    )
    with st.container(border=True):
        st.caption("Source: PenipuMY")
        _status_chip(str(reputation.get("status") or "Data unavailable").replace("_", " ").title())
        if total_reports:
            st.metric("Total report indicators", total_reports)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No matching reputation record found. This does not confirm that the caller is safe.")


def _render_reported_identity(output_view: dict[str, Any]) -> None:
    identity = dict(output_view.get("reported_identity", {}))
    if not identity.get("available"):
        return

    rows = _identity_rows(identity)
    if not rows:
        return

    _render_phone_step(
        "07",
        "Reported Identity or Business Record",
        "Provider-reported business association, shown only when returned.",
    )
    with st.container(border=True):
        st.caption("Source: PenipuMY")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "This record indicates an association reported by the provider. "
            "It does not verify that the current caller represents the organization."
        )


def _combined_evidence_rows(output_view: dict[str, Any], assessment: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    profile = dict(output_view.get("caller_profile", {}))

    if profile.get("valid") is True:
        rows.append(
            {
                "Finding": "Number is structurally valid",
                "Source": "Omkar",
                "Role": "Number metadata",
                "Effect": "Neutral",
            }
        )
    elif profile.get("valid") is False:
        rows.append(
            {
                "Finding": "Provider marked number invalid",
                "Source": "Omkar",
                "Role": "Number metadata",
                "Effect": "Raises concern",
            }
        )

    if _is_populated(profile.get("carrier")):
        rows.append(
            {
                "Finding": f"Carrier identified as {profile.get('carrier')}",
                "Source": "Omkar",
                "Role": "Network metadata",
                "Effect": "Neutral",
            }
        )

    if _is_populated(profile.get("line_type")):
        rows.append(
            {
                "Finding": f"Line type is {profile.get('line_type')}",
                "Source": "Omkar",
                "Role": "Network metadata",
                "Effect": "Neutral",
            }
        )

    for contribution in assessment.get("contributions", []):
        rows.append(
            {
                "Finding": str(contribution.get("indicator", "")),
                "Source": str(contribution.get("source", "")),
                "Role": "Reputation evidence"
                if str(contribution.get("source", "")).lower() != "investigation input"
                else "Caller context",
                "Effect": "Raises concern",
            }
        )

    deduped = []
    seen = set()
    for row in rows:
        key = (row["Finding"], row["Source"], row["Role"], row["Effect"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _render_combined_evidence(output_view: dict[str, Any], assessment: dict[str, Any]) -> None:
    rows = _combined_evidence_rows(output_view, assessment)
    _render_phone_step(
        "08",
        "Combined Evidence",
        "One deduplicated table separating neutral metadata from concern-producing reputation evidence.",
    )
    with st.container(border=True):
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No combined evidence rows are available yet.")


def _render_evidence_availability(output_view: dict[str, Any]) -> None:
    record = _combined_record_for_visuals(output_view)
    _render_phone_step(
        "09",
        "Evidence Availability",
        "Which metadata and reputation categories are available for this caller investigation.",
    )
    with st.container(border=True):
        st.plotly_chart(_evidence_coverage_chart(record), use_container_width=True)
        with st.expander("View Provider Diagnostics", expanded=False):
            st.plotly_chart(_response_completeness_chart(record), use_container_width=True)


def _render_claim_consistency(output_view: dict[str, Any]) -> None:
    input_data = dict(output_view.get("input", {}))
    claimed_identity = str(input_data.get("claimed_identity") or "").strip()
    if not claimed_identity:
        return

    record = _combined_record_for_visuals(output_view)
    claim_chart = _caller_claim_consistency_chart(record, claimed_identity)
    if claim_chart is None:
        return

    _render_phone_step(
        "10",
        "Caller Claim Consistency",
        "Transparent rule-based checks against the claimed identity, not an AI probability.",
    )
    with st.container(border=True):
        st.plotly_chart(claim_chart, use_container_width=True)
        st.caption("Caller Claim Consistency uses transparent rules only and does not modify the concern score.")


def _render_recommended_response(assessment: dict[str, Any]) -> None:
    _render_phone_step(
        "11",
        "Recommended Response",
        "Direct safety action based on the available evidence.",
    )
    with st.container(border=True):
        st.markdown(f"### {html.escape(str(assessment.get('priority') or 'Reputation unknown'))}")
        st.write(assessment.get("recommended_action", "Verify the caller through an official channel."))


def _render_caller_investigation_summary(investigation: dict[str, Any]) -> None:
    output_view = build_caller_output_view(investigation)
    assessment = dict(investigation.get("assessment") or _build_phone_assessment(investigation))
    investigation["assessment"] = assessment
    investigation["caller_profile"] = output_view.get("caller_profile", {})
    investigation["reported_identity"] = output_view.get("reported_identity", {})
    investigation["coverage"] = output_view.get("coverage", {})
    st.session_state["phone_investigation_result"] = investigation

    score = assessment.get("score")
    score_value = int(score) if isinstance(score, (int, float)) else None
    priority = str(assessment.get("priority") or "Reputation unknown")
    contributions = list(assessment.get("contributions", []))
    neutral = list(assessment.get("neutral_information", []))
    input_data = dict(output_view.get("input", {}))
    profile = dict(output_view.get("caller_profile", {}))
    reputation_view = dict(output_view.get("reputation", {}))
    identity = dict(output_view.get("reported_identity", {}))

    _render_phone_step(
        "04",
        "Caller Investigation Summary",
        "Review a transparent evidence-concern assessment and recommended next action.",
    )

    summary_col, action_col = st.columns([0.42, 0.58], gap="medium", vertical_alignment="top")
    with summary_col:
        with st.container(border=True):
            st.plotly_chart(_concern_meter(score_value, priority), use_container_width=True)
            st.caption(
                "This is an evidence concern score, not a trained-model scam probability. "
                "Carrier validity is treated as neutral information."
            )

    with action_col:
        with st.container(border=True):
            st.markdown(f"### {html.escape(priority)}")
            st.write(assessment.get("interpretation", "Review the caller evidence before continuing."))
            top_reasons = [str(item.get("indicator")) for item in contributions[:4]]
            if top_reasons:
                st.markdown("**What requires attention**")
                for reason in top_reasons:
                    st.markdown(f"- {html.escape(reason)}")
            else:
                st.info("No strong reputation evidence was collected. Unknown does not mean safe.")
            st.markdown("**Recommended action**")
            st.write(assessment.get("recommended_action", "Verify the caller through an official channel."))

    line_value, line_detail = _line_profile_summary(profile)
    reputation_value, reputation_detail = _reputation_summary(reputation_view)
    quick_cols = st.columns(3)
    quick_cols[0].metric("Phone Line", line_value)
    quick_cols[0].caption(line_detail)
    quick_cols[1].metric("Reputation", reputation_value)
    quick_cols[1].caption(reputation_detail)
    quick_cols[2].metric("Evidence Coverage", _live_provider_metric(investigation))
    quick_cols[2].caption(_coverage_summary(output_view))

    profile_bits = [
        str(input_data.get("normalized_number") or "").strip(),
        str(profile.get("country_code") or "").strip(),
        str(profile.get("line_type") or "").strip().title(),
        str(profile.get("carrier") or "").strip(),
    ]
    if _is_populated(profile.get("national_format")):
        profile_bits.append(f"National format: {profile.get('national_format')}")
    profile_line = " - ".join(bit for bit in profile_bits if bit)
    if profile_line:
        st.caption(profile_line)

    if identity.get("available") and _is_populated(identity.get("display_name")):
        association = f"Reported association: {identity.get('display_name')}"
        spoofing_count = _safe_int(identity.get("spoofing_report_count"))
        if spoofing_count:
            association += f" - Spoofing reports: {spoofing_count}"
        st.caption(association)

    chart = _contribution_chart(contributions)
    if chart is not None:
        st.markdown("**What raised the concern level?**")
        st.plotly_chart(chart, use_container_width=True)

    if neutral:
        st.markdown("**Neutral information**")
        st.dataframe(pd.DataFrame(neutral), hide_index=True, use_container_width=True)

    _render_phone_line_profile(output_view)
    _render_scam_reputation(output_view)
    _render_reported_identity(output_view)
    _render_combined_evidence(output_view, assessment)
    _render_evidence_availability(output_view)
    _render_claim_consistency(output_view)
    _render_recommended_response(assessment)


def render_phone_risk_page(root: Path, history: list[dict[str, object]]) -> None:
    _init_phone_input_state()
    _inject_phone_input_css()

    render_detection_tool_intro(
        title="Phone Number",
        description=(
            "Investigate caller numbers with independent carrier metadata and community reputation checks "
            "from configured live API providers."
        ),
        icon="solar:phone-calling-rounded-bold-duotone",
        accent="orange",
    )

    st.markdown('<div class="phone-workflow-shell">', unsafe_allow_html=True)

    _render_phone_step(
        "01",
        "Enter Phone Number",
        "Provide the caller number and any identity claimed during the call.",
    )

    number_col, identity_col, preview_col = st.columns([0.36, 0.38, 0.26], gap="small")
    with number_col:
        number = st.text_input(
            "Phone number",
            placeholder="012-345 6789 or +60 12-345 6789",
            help="Accepted examples: 012-345 6789, +60 12-345 6789, or (03) 1234 5678.",
            key="phone_investigation_number",
        )
        st.caption("Examples: 012-345 6789, +60 12-345 6789, (03) 1234 5678")

    with identity_col:
        claimed_identity = st.text_input(
            "Claimed caller identity (optional)",
            placeholder="e.g., bank officer, courier, university support",
            key="phone_claimed_identity",
        )
        st.caption("Helps interpret the context of the call.")

    ok, validation_message = validate_phone_query(number)
    normalized = normalise_phone_query(number)
    with preview_col:
        if ok and normalized:
            st.markdown(
                f"""
                <div class="phone-normalized-preview">
                    <span>Normalized number</span>
                    <strong>{html.escape(normalized)}</strong>
                    <span style="margin-top:.55rem">Malaysia - E.164-style</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif number.strip():
            st.warning(validation_message)
        else:
            st.info("Enter a number to preview normalization.")

    st.divider()

    _render_phone_step(
        "02",
        "Provider Configuration & Status",
        "Configure providers, test connections, and view their current status.",
    )

    provider_cols = st.columns(2, gap="small")
    with provider_cols[0]:
        omkar_info = _render_live_provider_card(
            provider_id="omkar_carrier_lookup",
            title="Omkar Carrier Lookup",
            purpose="Carrier and number metadata",
            icon="solar:radio-bold-duotone",
            enabled_key="phone_omkar_enabled",
            session_key="phone_omkar_api_key",
            test_number_key="phone_omkar_test_number",
            diagnostic_key="phone_omkar_diagnostic",
        )

    with provider_cols[1]:
        penipu_info = _render_live_provider_card(
            provider_id="penipumy",
            title="PenipuMY",
            purpose="Malaysian scam reports and community reputation",
            icon="solar:users-group-rounded-bold-duotone",
            enabled_key="phone_penipumy_enabled",
            session_key="phone_penipumy_api_key",
            test_number_key="phone_penipumy_test_number",
            diagnostic_key="phone_penipumy_diagnostic",
        )

    _render_status_row(
        bool(omkar_info["enabled"]),
        dict(omkar_info["key_meta"]),
        bool(penipu_info["enabled"]),
        dict(penipu_info["key_meta"]),
    )

    st.divider()

    _render_phone_step(
        "03",
        "Start Investigation",
        "Run enabled provider checks and prepare combined caller evidence.",
    )

    has_usable_provider = any(
        [
            bool(omkar_info["enabled"]) and bool(omkar_info["key_meta"].get("configured")),
            bool(penipu_info["enabled"]) and bool(penipu_info["key_meta"].get("configured")),
        ]
    )
    disabled_reason = ""
    if not ok:
        disabled_reason = validation_message or "Enter a valid phone number first."
    elif not has_usable_provider:
        disabled_reason = "Enable at least one live provider with a configured API key."

    with st.container(key="phone_investigate_button"):
        investigate = st.button(
            "Investigate Phone Number",
            use_container_width=True,
            disabled=bool(disabled_reason),
        )

    if disabled_reason:
        st.caption(disabled_reason)

    if investigate:
        with st.spinner("Collecting phone evidence..."):
            investigation = _build_phone_investigation(
                root=root,
                raw_number=number,
                normalized_number=normalized,
                claimed_identity=claimed_identity,
                omkar_enabled=bool(omkar_info["enabled"]),
                omkar_key=str(omkar_info["key_meta"].get("key", "")),
                penipumy_enabled=bool(penipu_info["enabled"]),
                penipumy_key=str(penipu_info["key_meta"].get("key", "")),
            )
            investigation["assessment"] = _build_phone_assessment(investigation)
            st.session_state["phone_investigation_result"] = investigation
            _record_phone_investigation(history, investigation)
        render_analysis_ready("Phone investigation evidence collected")
        st.caption("The unified investigation object is saved in session state for the next output phase.")

    investigation_result = st.session_state.get("phone_investigation_result")
    if isinstance(investigation_result, dict):
        st.divider()
        _render_caller_investigation_summary(investigation_result)

    st.markdown("</div>", unsafe_allow_html=True)
    return
