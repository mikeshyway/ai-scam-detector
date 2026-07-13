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
    get_local_dataset_status,
    lookup_local_reputation,
    lookup_omkar_metadata,
    lookup_penipumy_reputation,
    test_omkar_connection,
    test_penipumy_connection,
)
from src.phone.providers.models import diagnostic_rows


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
        "local_fallback": "Local fallback dataset",
        "demo_fallback": "Demo fallback dataset",
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
                f"No {provider_label} API key was detected. The lookup will automatically use the local fallback "
                "dataset when you check a number.",
                kind="warning",
                code="FALLBACK",
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
        title="Lookup Evidence Coverage",
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
    local = sum(1 for item in phone_rows if "fallback dataset" in str(item.get("source", "")).lower() or "local" in str(item.get("source", "")).lower())
    unknown = sum(1 for item in phone_rows if "unknown" in str(item.get("source", "")).lower() or str(item.get("prediction", "")).lower() == "unknown")
    provider_failures = max(0, local + unknown)

    fig = go.Figure(
        go.Bar(
            x=["Live API successes", "Local fallback uses", "Unknown results", "Provider failures"],
            y=[live, local, unknown, provider_failures],
            marker_color=["#38BDF8", "#F59E0B", "#64748B", "#EF4444"],
            text=[live, local, unknown, provider_failures],
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
                "Normal lookup can still continue through the local fallback dataset.",
                kind="warning",
                code="DIAGNOSTIC",
            )
        else:
            render_info_banner(
                "No API key was configured. Normal lookup can still use the local fallback dataset.",
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

    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Phone",
            "prediction": risk.get("risk_level", "Unknown"),
            "confidence": float(risk.get("risk_score", 0)),
            "model": f"{_provider_label(str(result.get('requested_provider', 'omkar_carrier_lookup')))} + fallback reputation rules",
            "source": _source_label(str(result.get("source", ""))),
            "preview": phone_number,
            "flags": flags,
            "explanation": explanation.get("summary", ""),
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

    _render_phone_step("03", "Review Evidence", "Inspect source, fallback status, metadata, reputation evidence, and recommended action.")

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
            "Business or fallback identity fields associated with the queried number.",
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
        "phone_local_fallback_enabled": True,
        "phone_auto_fallback_enabled": True,
        "phone_omkar_api_key": "",
        "phone_penipumy_api_key": "",
        "phone_omkar_test_number": "016-240 4384",
        "phone_penipumy_test_number": "016-240 4384",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
        enabled = st.toggle("Enable provider", key=enabled_key)
        session_value = st.text_input(
            "API key",
            type="password",
            key=session_key,
            disabled=not enabled,
            placeholder=f"Paste {title} API key for this session",
        )
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


def _render_local_fallback_card(root: Path) -> dict[str, Any]:
    with st.container(border=True):
        st.markdown(
            """
            <div class="phone-provider-heading">
                <iconify-icon icon="solar:database-bold-duotone"></iconify-icon>
                <div>
                    <strong>Local Fallback Dataset</strong>
                    <span>Offline educational reputation evidence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enabled = st.toggle(
            "Enable fallback dataset",
            key="phone_local_fallback_enabled",
        )
        auto_fallback = st.toggle(
            "Automatic fallback",
            key="phone_auto_fallback_enabled",
            disabled=not enabled,
        )
        status = get_local_dataset_status(root)
        if st.button("Validate Dataset", key="phone_validate_local_dataset", use_container_width=True, disabled=not enabled):
            status = get_local_dataset_status(root)
            st.session_state["phone_local_dataset_status"] = status.as_dict()

        dataset_usable = enabled and status.schema_valid and status.record_count > 0
        label = (
            "Ready"
            if dataset_usable
            else "Empty dataset"
            if enabled and status.schema_valid
            else "Missing"
            if not status.exists
            else "Invalid schema"
        )
        _status_chip(label)
        st.caption(f"{status.record_count:,} records loaded" if status.schema_valid else status.error_message)
        st.caption(f"Source: {status.source_filename}")
        return {"enabled": enabled, "auto_fallback": auto_fallback, "status": status}


def _render_status_row(
    omkar_enabled: bool,
    omkar_key: dict[str, object],
    penipu_enabled: bool,
    penipu_key: dict[str, object],
    local_info: dict[str, Any],
) -> None:
    local_status = local_info["status"]
    rows = [
        {
            "Provider": "Omkar",
            "Status": "Ready" if omkar_enabled and omkar_key.get("configured") else "Not configured" if omkar_enabled else "Disabled",
        },
        {
            "Provider": "PenipuMY",
            "Status": "Ready" if penipu_enabled and penipu_key.get("configured") else "Not configured" if penipu_enabled else "Disabled",
        },
        {
            "Provider": "Local dataset",
            "Status": (
                "Ready"
                if local_info.get("enabled") and local_status.schema_valid and local_status.record_count > 0
                else "Missing / Invalid"
            ),
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
    fallback_enabled: bool,
    auto_fallback: bool,
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
    fallback = {
        "provider": "local",
        "enabled": fallback_enabled,
        "used": False,
        "reason": None,
        "data": {},
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

    should_try_fallback = False
    fallback_reason = None
    if penipumy_enabled:
        requested += 1
        if penipumy_key:
            penipu_result = lookup_penipumy_reputation(normalized_number, penipumy_key).as_dict()
            reputation = _provider_result_block(penipu_result, default_provider="penipumy", enabled=True)
            if penipu_result.get("success"):
                completed += 1
            else:
                failed += 1
                should_try_fallback = bool(auto_fallback)
                fallback_reason = reputation.get("error_message") or reputation.get("error_code")
        else:
            should_try_fallback = bool(auto_fallback)
            fallback_reason = "PenipuMY API key is not configured."
    elif fallback_enabled:
        should_try_fallback = True
        fallback_reason = "PenipuMY disabled; local fallback used as reputation channel."

    if fallback_enabled and should_try_fallback:
        local_result = lookup_local_reputation(root, normalized_number).as_dict()
        fallback.update(
            {
                "used": True,
                "reason": fallback_reason,
                "data": local_result.get("data", {}),
                "status": local_result.get("status"),
                "error_code": local_result.get("error_code"),
                "error_message": local_result.get("error_message"),
            }
        )

    return {
        "input": {
            "raw_number": raw_number,
            "normalized_number": normalized_number,
            "claimed_identity": claimed_identity,
        },
        "metadata": metadata,
        "reputation": reputation,
        "fallback": fallback,
        "provider_coverage": {
            "requested": requested,
            "completed": completed,
            "failed": failed,
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def render_phone_risk_page(root: Path, history: list[dict[str, object]]) -> None:
    _init_phone_input_state()
    _inject_phone_input_css()

    render_detection_tool_intro(
        title="Phone Number",
        description=(
            "Investigate caller numbers with independent carrier metadata, community reputation checks, "
            "and local fallback evidence when live reputation data is unavailable."
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

    provider_cols = st.columns(3, gap="small")
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

    with provider_cols[2]:
        local_info = _render_local_fallback_card(root)

    _render_status_row(
        bool(omkar_info["enabled"]),
        dict(omkar_info["key_meta"]),
        bool(penipu_info["enabled"]),
        dict(penipu_info["key_meta"]),
        local_info,
    )

    st.divider()

    _render_phone_step(
        "03",
        "Start Investigation",
        "Run enabled provider checks and prepare combined caller evidence.",
    )

    local_status = local_info["status"]
    has_usable_provider = any(
        [
            bool(omkar_info["enabled"]) and bool(omkar_info["key_meta"].get("configured")),
            bool(penipu_info["enabled"]) and bool(penipu_info["key_meta"].get("configured")),
            bool(local_info["enabled"]) and bool(local_status.schema_valid) and int(local_status.record_count) > 0,
        ]
    )
    disabled_reason = ""
    if not ok:
        disabled_reason = validation_message or "Enter a valid phone number first."
    elif not has_usable_provider:
        disabled_reason = "Enable at least one configured live provider or a valid local fallback dataset."

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
            st.session_state["phone_investigation_result"] = _build_phone_investigation(
                root=root,
                raw_number=number,
                normalized_number=normalized,
                claimed_identity=claimed_identity,
                omkar_enabled=bool(omkar_info["enabled"]),
                omkar_key=str(omkar_info["key_meta"].get("key", "")),
                penipumy_enabled=bool(penipu_info["enabled"]),
                penipumy_key=str(penipu_info["key_meta"].get("key", "")),
                fallback_enabled=(
                    bool(local_info["enabled"])
                    and bool(local_status.schema_valid)
                    and int(local_status.record_count) > 0
                ),
                auto_fallback=bool(local_info["auto_fallback"]),
            )
        render_analysis_ready("Phone investigation evidence collected")
        st.caption("The unified investigation object is saved in session state for the next output phase.")

    st.markdown("</div>", unsafe_allow_html=True)
    return

    provider = "omkar_carrier_lookup"

    render_detection_tool_intro(
        title="Phone Number",
        description=(
            "Check phone-number validity, carrier metadata, and local reputation evidence using Omkar Carrier "
            "Lookup first, then a local fallback dataset when live metadata is unavailable."
        ),
        icon="solar:phone-calling-rounded-bold-duotone",
        accent="orange",
    )

    render_section_header(
        "Phone number metadata lookup",
        "Omkar provides carrier and validity metadata. Local fallback records provide reputation evidence when available.",
        "Carrier lookup",
    )

    st.caption(
        "Live provider: Omkar Carrier Lookup. Results use carrier metadata first, then local fallback evidence when available."
    )

    render_info_banner(
        "Omkar metadata does not contain police reports, community scam reports, or scam probability. "
        "A valid number is not automatically a safe caller.",
        kind="info",
        code="SCOPE",
    )

    demo_mode = st.checkbox(
        "Demo Mode - allow fictional phone records for presentation",
        value=False,
        key="phone_demo_mode",
        help=(
            "Normal lookups do not search synthetic demo records. Enable this only when you intentionally "
            "want capstone presentation samples."
        ),
    )
    if demo_mode:
        render_info_banner(
            "Demo Mode is enabled. Fictional local records may be returned and will be labelled as demonstration data.",
            kind="warning",
            code="DEMO MODE",
        )

    key_meta = _render_api_setup(provider, root)
    _render_provider_connection_check(provider, key_meta)
    api_key = str(key_meta.get("key", "") or "")

    _render_phone_step(
        "01",
        "Enter Phone Number",
        "Use local or international format. Examples: 012-345 6789, +60 12-345 6789, or (03) 1234 5678.",
    )
    render_content_card_open("violet")
    number = st.text_input(
        "Phone number",
        placeholder="012-345 6789 or +60 12-345 6789",
        help=(
            "Spaces, hyphens, brackets, and one leading plus sign are accepted. "
            "The app normalizes the value before provider lookup."
        ),
    )
    claimed_identity = st.text_input(
        "Claimed caller identity (optional)",
        placeholder="Example: bank officer, courier, university support, unknown caller",
    )

    ok, validation_message = validate_phone_query(number)
    normalized = normalise_phone_query(number)
    if number.strip() and not ok:
        st.warning(validation_message)
    elif normalized:
        st.caption(f"Normalized lookup query: `{normalized}`")
    render_content_card_close()

    _render_phone_step("02", "Check Reputation", "Try Omkar Carrier Lookup first, then local fallback, then unknown fallback.")
    check_button = st.button("Check Phone Reputation", type="primary", use_container_width=True)

    render_info_banner(
        "This module does not intercept calls or verify telecom ownership. It checks reputation evidence and teaches safe response.",
        kind="info",
        code="SCOPE",
    )

    if not check_button:
        return

    if not ok:
        st.warning(validation_message)
        return

    with st.spinner("Checking caller reputation..."):
        try:
            lookup_result = _lookup_phone_compat(
                normalized,
                root,
                provider=provider,
                api_key=api_key,
                demo_mode=demo_mode,
            )
        except ValueError as exc:
            st.warning(str(exc))
            return

    _render_result(root, lookup_result, claimed_identity, history)
