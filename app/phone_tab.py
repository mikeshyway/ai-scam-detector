"""Phone-number reputation lookup page."""

from __future__ import annotations

from datetime import datetime
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.ui_components import (
    render_analysis_ready,
    render_content_card_close,
    render_content_card_open,
    render_detection_tool_intro,
    render_info_banner,
    render_result_card,
    render_section_header,
)
from src.phone.ipqs_client import lookup_ipqs_phone
from src.phone.omkar_client import lookup_omkar_phone
from src.phone.penipumy_client import PenipuApiError, PenipuClientError, fetch_phone_reputation
from src.phone.phone_lookup import lookup_phone, normalise_phone_query, validate_phone_query
from src.phone.phone_number import format_phone_for_ipqs, format_phone_for_omkar, format_phone_for_penipumy


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

    if provider == "ipqualityscore":
        env_names = ["IPQS_API_KEY"]
        direct_secret_names = ["IPQS_API_KEY"]
        sections = [("ipqs", "api_key")]
    elif provider == "omkar_carrier_lookup":
        env_names = ["OMKAR_API_KEY", "OMKAR_CARRIER_API_KEY"]
        direct_secret_names = ["OMKAR_API_KEY", "OMKAR_CARRIER_API_KEY"]
        sections = [("omkar", "api_key"), ("carrier_lookup", "api_key")]
    else:
        env_names = ["PENIPUMY_API_KEY", "PENIPU_API_KEY"]
        direct_secret_names = ["PENIPUMY_API_KEY", "PENIPU_API_KEY"]
        sections = [("penipumy", "api_key"), ("penipu", "api_key")]

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
    """Read a PenipuMY API key from environment variables or Streamlit secrets."""

    return str(_resolve_api_key("penipumy").get("key", ""))


def _configured_ipqs_api_key() -> str:
    """Read an IPQualityScore API key from environment variables or Streamlit secrets."""

    return str(_resolve_api_key("ipqualityscore").get("key", ""))


def _configured_omkar_api_key() -> str:
    """Read an Omkar Carrier Lookup API key from environment variables or Streamlit secrets."""

    return str(_resolve_api_key("omkar_carrier_lookup").get("key", ""))


def _source_label(source: str) -> str:
    labels = {
        "penipumy_api": "PenipuMY API",
        "ipqualityscore_api": "IPQualityScore API",
        "omkar_carrier_lookup_api": "Carrier Lookup API",
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
        return "Carrier Lookup"
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
            "The phone lookup backend loaded by Streamlit is still the older PenipuMY-only version. "
            "Refresh or restart Streamlit once so IPQualityScore provider support is loaded."
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

    docs_url_by_provider = {
        "ipqualityscore": "https://www.ipqualityscore.com/documentation/phone-number-validation-api/overview",
        "omkar_carrier_lookup": "https://github.com/omkarcloud/phone-lookup-api",
        "penipumy": "https://penipu.my/api/v1/docs",
    }
    env_name_by_provider = {
        "ipqualityscore": "IPQS_API_KEY",
        "omkar_carrier_lookup": "OMKAR_API_KEY",
        "penipumy": "PENIPUMY_API_KEY",
    }
    help_text_by_provider = {
        "ipqualityscore": "The key is sent only to IPQualityScore in the request URL.",
        "omkar_carrier_lookup": "The key is sent only to Omkar Carrier Lookup through the API-Key request header.",
        "penipumy": "The key is sent only to PenipuMY through the X-API-Key request header.",
    }
    docs_url = docs_url_by_provider.get(provider, docs_url_by_provider["penipumy"])
    env_name = env_name_by_provider.get(provider, "PENIPUMY_API_KEY")
    help_text = help_text_by_provider.get(provider, help_text_by_provider["penipumy"])

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

        st.link_button(
            f"Open {provider_label} API Documentation",
            docs_url,
            use_container_width=True,
        )

        guide_path = root / "docs" / "phone_api_setup_guide.html"
        if guide_path.exists():
            st.download_button(
                "Download API Setup Guide",
                data=guide_path.read_bytes(),
                file_name="phone_api_setup_guide.html",
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

    if provider == "ipqualityscore":
        live = lookup_ipqs_phone(format_phone_for_ipqs(normalized), api_key, timeout=10.0)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = dict(live.get("record", {})) if isinstance(live.get("record"), dict) else {}
        field_summary = _response_field_summary(payload)
        result.update(
            {
                "connected": live.get("status_code") is not None,
                "authenticated": bool(live.get("ok")),
                "http_status": live.get("status_code"),
                "provider_success": payload.get("success"),
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

    if provider == "omkar_carrier_lookup":
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

    try:
        live = fetch_phone_reputation(format_phone_for_penipumy(normalized), api_key)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = dict(live.data)
        field_summary = _response_field_summary(payload)
        result.update(
            {
                "connected": True,
                "authenticated": True,
                "http_status": live.status_code,
                "provider_success": True,
                "response_time_ms": elapsed_ms,
                "request_id": str(payload.get("request_id") or ""),
                "rate_limit": live.rate_limit,
                "total_fields": field_summary["total_fields"],
                "populated_fields": field_summary["populated_fields"],
                "empty_fields": field_summary["empty_fields"],
                "error": None,
                "error_category": "none",
                "payload": payload,
            }
        )
    except PenipuApiError as exc:
        result.update(
            {
                "connected": True,
                "http_status": exc.status_code,
                "response_time_ms": (time.perf_counter() - started) * 1000,
                "error": str(exc),
                "error_category": _diagnostic_category(provider, exc.status_code, str(exc), {}),
            }
        )
    except PenipuClientError as exc:
        result.update(
            {
                "response_time_ms": (time.perf_counter() - started) * 1000,
                "error": str(exc),
                "error_category": _diagnostic_category(provider, None, str(exc), {}),
            }
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
            {"Check": "Selected provider", "Result": _provider_label(str(diagnostic.get("provider"))), "Detail": "User selected"},
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
        "Provider connection check",
        "Test whether the selected provider is configured, reachable, authenticated, and returning usable fields.",
        "API diagnostics",
    )
    render_content_card_open("violet")
    st.caption(
        "This test makes one live request only when you click the button. It never displays API keys, request headers, or provider request URLs."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Item": "Selected provider", "Value": _provider_label(provider)},
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
        help="Use a number you are comfortable sending to the selected provider for testing.",
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
            "model": f"{_provider_label(str(result.get('requested_provider', 'penipumy')))} + fallback reputation rules",
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
    requested_provider = str(lookup_result.get("requested_provider", "penipumy"))
    fallback_reason = str(lookup_result.get("fallback_reason") or "")
    phone = str(record.get("phone") or "")
    is_demo = bool(lookup_result.get("is_demo") or record.get("is_demo"))
    score = float(risk.get("risk_score", 0))
    risk_level = str(risk.get("risk_level", "Unknown"))
    is_ipqs_result = source == "ipqualityscore_api"
    is_omkar_result = source == "omkar_carrier_lookup_api"

    render_analysis_ready("Phone reputation check complete - results ready below")

    if risk_level == "Unknown":
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

    _render_phone_step("03", "Review Evidence", "Inspect source, fallback status, reports, and recommended action.")

    render_content_card_open("violet")
    cols = st.columns(3)
    cols[0].metric("Risk Level", risk_level)
    cols[1].metric("Source Used", _source_label(source))
    if is_ipqs_result:
        fraud_score = record.get("fraud_score")
        cols[2].metric("Fraud Score", f"{float(fraud_score or 0):.0f}/100")
    elif is_omkar_result:
        cols[2].metric("Line Type", str(record.get("line_type") or "N/A"))
    else:
        cols[2].metric("Reports Found", int(record.get("police_report_count", 0) or 0) + int(record.get("verified_report_count", 0) or 0))

    cols = st.columns(3)
    if is_omkar_result:
        cols[0].metric("Carrier", str(record.get("carrier") or "N/A"))
    else:
        cols[0].metric("Business Status", metrics.get("Business Status", "No match"))
    live_source = f"{requested_provider}_api"
    if requested_provider == "ipqualityscore":
        live_source = "ipqualityscore_api"
    elif requested_provider == "omkar_carrier_lookup":
        live_source = "omkar_carrier_lookup_api"
    cols[1].metric("Fallback Status", "Live API" if source == live_source else "Fallback used")
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


def render_phone_risk_page(root: Path, history: list[dict[str, object]]) -> None:
    render_detection_tool_intro(
        title="Phone Number",
        description=(
            "Choose PenipuMY, IPQualityScore, or Carrier Lookup for live caller checks, then fall back to "
            "a local processed dataset and transparent rules when the selected provider is unavailable."
        ),
        icon="solar:phone-calling-rounded-bold-duotone",
        accent="orange",
    )

    render_section_header(
        "Phone number reputation lookup",
        "Phone numbers are reputation lookups, not ML text classifiers. The system explains database evidence and fallback status.",
        "Caller reputation",
    )

    provider_label = st.radio(
        "Lookup provider",
        ["PenipuMY", "IPQualityScore", "Carrier Lookup"],
        horizontal=True,
        key="phone_lookup_provider",
    )
    provider = _provider_key(provider_label)

    if provider == "penipumy":
        st.caption("PenipuMY checks Malaysian community scam reports and business reputation.")
    elif provider == "ipqualityscore":
        st.caption("IPQualityScore checks phone validity, carrier metadata, line type, and fraud-risk signals.")
    else:
        st.caption("Carrier Lookup checks number validity, carrier, line type, country code, and formatting metadata.")

    render_info_banner(
        "This page supports API keys for all providers, but each provider needs its own key: "
        "PenipuMY uses PENIPUMY_API_KEY, IPQualityScore uses IPQS_API_KEY, and Carrier Lookup uses OMKAR_API_KEY. "
        "The selected lookup only calls the selected provider.",
        kind="info",
        code="API KEYS",
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

    _render_phone_step("02", "Check Reputation", f"Try {_provider_label(provider)} first, then local fallback, then unknown fallback.")
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
