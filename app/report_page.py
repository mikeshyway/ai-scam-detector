"""AI report generator page."""

from __future__ import annotations

import json
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    render_analysis_ready,
    render_info_banner,
    render_metric_row,
    render_section_header,
)
from src.reporting.history_db import (
    DEFAULT_SESSION_ID,
    delete_all_history,
    delete_selected,
    history_fingerprint,
    init_db,
    log_export,
    query_history,
    sync_session_history,
)
from src.reporting.report_builder import (
    DEFAULT_RECOMMENDATION,
    DEFAULT_SECTIONS,
    REPORT_PROFILES,
    REPORT_SCHEMA_VERSION,
    STUDENT_PROFILE,
    build_preview,
    build_report,
)
from src.reporting.evidence_snapshot import (
    ACTION_STATUSES,
    IMMEDIATE_ACTION,
    NO_IMMEDIATE_ACTION,
    REVIEW_REQUIRED,
)
from src.utils.time_utils import now_for_app


def _parse_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace(" ", "T")[:19]).date()
    except ValueError:
        return None


def _unique(rows: list[dict[str, object]], key: str) -> list[str]:
    values = {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}
    return sorted(values)


def _confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summary_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    immediate_count = sum(1 for row in rows if row.get("action_status") == IMMEDIATE_ACTION)
    review_count = sum(1 for row in rows if row.get("action_status") == REVIEW_REQUIRED)
    no_immediate_count = sum(1 for row in rows if row.get("action_status") == NO_IMMEDIATE_ACTION)
    available_scores = [
        _confidence(row.get("concern_score"))
        for row in rows
        if bool(row.get("score_available")) and row.get("concern_score") is not None
    ]
    average = sum(available_scores) / len(available_scores) if available_scores else 0
    types = len(_unique(rows, "scan_type"))
    return [
        {"label": "Evidence Items", "value": len(rows), "color": "#0891B2"},
        {"label": "Immediate Action", "value": immediate_count, "color": "#DC2626"},
        {"label": "Review Required", "value": review_count, "color": "#D97706"},
        {"label": "No Immediate Action", "value": no_immediate_count, "color": "#059669"},
        {"label": "Average Concern", "value": f"{average:.0f}%" if available_scores else "N/A", "color": "#7C3AED"},
        {"label": "Scan Types", "value": types, "color": "#2563EB"},
    ]


def _history_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Select": False,
                "ID": int(row.get("id", 0)),
                "Time": str(row.get("scanned_at", "")).replace("T", " ")[:19],
                "Type": row.get("scan_type", "-"),
                "Native Verdict": row.get("native_prediction") or row.get("prediction", "-"),
                "Action Status": row.get("action_status", REVIEW_REQUIRED),
                "Concern Score": (
                    f"{_confidence(row.get('concern_score')):.1f}%"
                    if bool(row.get("score_available")) and row.get("concern_score") is not None
                    else "N/A"
                ),
                "Native Confidence": f"{_confidence(row.get('confidence')):.1f}%",
                "Model": row.get("model_name", "-"),
                "Source": row.get("source_name", "-"),
                "Preview": row.get("preview", "-"),
            }
            for row in rows
        ]
    )


def _selected_ids_from_editor(frame: pd.DataFrame) -> list[int]:
    if frame.empty or "Select" not in frame.columns:
        return []
    selected = frame[frame["Select"] == True]  # noqa: E712 - Streamlit returns bool-like values.
    return [int(value) for value in selected["ID"].tolist()]


def _rows_by_id(rows: list[dict[str, object]], selected_ids: list[int]) -> list[dict[str, object]]:
    selected = {int(scan_id) for scan_id in selected_ids}
    return [row for row in rows if int(row.get("id", 0)) in selected]


def _remove_deleted_from_session(history: list[dict[str, object]], deleted_rows: list[dict[str, object]]) -> None:
    deleted_fingerprints = {str(row.get("source_fingerprint")) for row in deleted_rows}
    history[:] = [
        item
        for item in history
        if str(item.get("source_fingerprint") or history_fingerprint(item)) not in deleted_fingerprints
    ]


def _no_result_message(
    *,
    all_rows: list[dict[str, object]],
    date_from: date,
    date_to: date,
    selected_types: list[str],
    selected_action_statuses: list[str],
    selected_native_predictions: list[str],
) -> str:
    if date_from > date_to:
        return "No result: the start date is after the end date. Choose a valid date range."
    if not all_rows:
        return "No result: no scan evidence has been saved yet. Run a scan first, then return to this page."
    if not selected_types:
        return "No result: no scan type is selected."
    if not selected_action_statuses:
        return "No result: no action status is selected."
    if not selected_native_predictions:
        return "No result: no native verdict is selected."

    date_rows = query_history(
        session_id=DEFAULT_SESSION_ID,
        date_from=str(date_from),
        date_to=str(date_to),
    )
    if not date_rows:
        return f"No result: no saved scans were found from {date_from} to {date_to}."

    type_rows = query_history(
        session_id=DEFAULT_SESSION_ID,
        date_from=str(date_from),
        date_to=str(date_to),
        scan_types=selected_types,
    )
    if not type_rows:
        return "No result: the selected scan type has no saved scans in this date range."

    return "No result: the selected action status or native verdict has no saved scans for the current filters."


def _render_risk_chart(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    chart_rows = pd.DataFrame(
        [
            {
                "Evidence": f"{row.get('scan_type', 'Unknown')} #{index}",
                "Concern": (
                    _confidence(row.get("concern_score"))
                    if bool(row.get("score_available")) and row.get("concern_score") is not None
                    else 0.0
                ),
                "Score Availability": (
                    str(row.get("score_label") or "Concern score")
                    if bool(row.get("score_available")) and row.get("concern_score") is not None
                    else "N/A - no comparable concern score"
                ),
                "Action Status": row.get("action_status", REVIEW_REQUIRED),
                "Native Verdict": row.get("native_prediction") or row.get("prediction", "Unknown"),
                "Native Confidence": _confidence(row.get("confidence")),
            }
            for index, row in enumerate(rows, 1)
        ]
    )
    fig = px.bar(
        chart_rows,
        x="Concern",
        y="Evidence",
        color="Action Status",
        orientation="h",
        title="Selected evidence action overview",
        labels={"Concern": "Source-native concern score (%)"},
        hover_data={
            "Native Verdict": True,
            "Native Confidence": ":.1f",
            "Score Availability": True,
        },
        color_discrete_map={
            IMMEDIATE_ACTION: "#DC2626",
            REVIEW_REQUIRED: "#D97706",
            NO_IMMEDIATE_ACTION: "#059669",
        },
    )
    fig.update_layout(
        height=max(280, 54 * len(rows) + 100),
        margin=dict(l=10, r=10, t=40, b=35),
        xaxis=dict(range=[0, 100]),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(chart_rows["Evidence"].tolist()))),
    )
    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)


def _report_rows_json(rows: list[dict[str, object]]) -> str:
    return json.dumps(rows, sort_keys=True, default=str, ensure_ascii=True)


def _sections_json(sections: dict[str, bool]) -> str:
    return json.dumps(sections, sort_keys=True, ensure_ascii=True)


def _context_json(context: dict[str, object]) -> str:
    return json.dumps(context, sort_keys=True, ensure_ascii=True)


@st.cache_data(show_spinner=False, ttl=900, max_entries=16)
def _build_cached_report(
    report_format: str,
    rows_json: str,
    report_note: str,
    sections_json: str,
    incident_context_json: str,
    report_schema_version: str,
) -> tuple[bytes, str, str]:
    del report_schema_version
    rows = json.loads(rows_json)
    sections = json.loads(sections_json)
    incident_context = json.loads(incident_context_json)
    return build_report(report_format, rows, report_note, sections, incident_context)


def _clear_all_evidence(history: list[dict[str, object]]) -> None:
    deleted_count = delete_all_history(DEFAULT_SESSION_ID)
    history.clear()
    st.session_state["report_notice"] = f"Cleared {deleted_count} evidence record(s)."
    st.session_state["show_clear_all_dialog"] = False
    st.rerun()


def _render_clear_all_confirmation(history: list[dict[str, object]]) -> None:
    dialog_factory = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
    if dialog_factory:
        @dialog_factory("Clear all evidence?")
        def confirm_clear_all() -> None:
            st.warning(
                "This will permanently remove every saved scan evidence record from the local report history."
            )
            cancel_col, delete_col = st.columns(2)
            with cancel_col:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["show_clear_all_dialog"] = False
                    st.rerun()
            with delete_col:
                if st.button("Delete all evidence", type="primary", use_container_width=True):
                    _clear_all_evidence(history)

        confirm_clear_all()
        return

    with st.container(border=True):
        st.warning("Confirm clear all: this will permanently remove every saved scan evidence record.")
        cancel_col, delete_col = st.columns(2)
        with cancel_col:
            if st.button("Cancel clear all", use_container_width=True):
                st.session_state["show_clear_all_dialog"] = False
                st.rerun()
        with delete_col:
            if st.button("Confirm delete all evidence", type="primary", use_container_width=True):
                _clear_all_evidence(history)


def render_report_page(root: Path, history: list[dict[str, object]]) -> None:
    del root
    init_db()
    if "report_case_identifier" not in st.session_state:
        st.session_state["report_case_identifier"] = now_for_app().strftime("AI-FDS-%Y%m%d-%H%M")
    synced = sync_session_history(history, session_id=DEFAULT_SESSION_ID)

    render_section_header(
        "AI analysis report generator",
        "Select scan evidence, configure report sections, preview the output, and export a professional summary.",
        "Evidence reporting",
    )
    render_info_banner(
        "This page now acts as the central evidence destination. Current scan pages can keep writing to session history; "
        "this report page syncs those entries into a local SQLite store automatically.",
        kind="success",
        code="SYNC",
    )
    if synced:
        render_analysis_ready(f"{synced} new scan record(s) synced into report history")
    notice = st.session_state.pop("report_notice", None)
    if notice:
        render_analysis_ready(str(notice))

    all_rows = query_history(session_id=DEFAULT_SESSION_ID)
    render_metric_row(_summary_metrics(all_rows))

    with st.container(border=True):
        st.subheader("Evidence filters")
        dates = [_parse_date(row.get("scanned_at")) for row in all_rows]
        dates = [item for item in dates if item]
        current_date = now_for_app().date()
        default_from = min(dates) if dates else current_date - timedelta(days=30)
        default_to = max(dates) if dates else current_date

        col_a, col_b = st.columns(2)
        with col_a:
            date_from = st.date_input("From", value=default_from)
        with col_b:
            date_to = st.date_input("To", value=default_to)

        type_options = _unique(all_rows, "scan_type")
        action_options = [
            status for status in ACTION_STATUSES if status in _unique(all_rows, "action_status")
        ]
        native_prediction_options = _unique(all_rows, "native_prediction")
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            selected_types = st.multiselect("Scan types", type_options, default=type_options)
        with col_e:
            selected_action_statuses = st.multiselect(
                "Action status",
                action_options,
                default=action_options,
                help="Unified triage label. The source's original verdict remains available separately.",
            )
        with col_f:
            selected_native_predictions = st.multiselect(
                "Native verdict",
                native_prediction_options,
                default=native_prediction_options,
                help="Original result produced by the Email, Transcript/Audio, or Phone investigation.",
            )

    if (
        date_from <= date_to
        and selected_types
        and selected_action_statuses
        and selected_native_predictions
    ):
        filtered_rows = query_history(
            session_id=DEFAULT_SESSION_ID,
            date_from=str(date_from),
            date_to=str(date_to),
            scan_types=selected_types,
            action_statuses=selected_action_statuses,
            native_predictions=selected_native_predictions,
        )
    else:
        filtered_rows = []

    render_section_header(
        "Saved scan evidence",
        "Choose specific rows for the report, or leave everything unselected to include the filtered evidence set.",
        "Report input",
    )

    if not filtered_rows:
        render_info_banner(
            _no_result_message(
                all_rows=all_rows,
                date_from=date_from,
                date_to=date_to,
                selected_types=selected_types,
                selected_action_statuses=selected_action_statuses,
                selected_native_predictions=selected_native_predictions,
            ),
            kind="warning",
            code="NO RESULT",
        )
        return

    with st.container(border=True):
        edited_frame = st.data_editor(
            _history_frame(filtered_rows),
            hide_index=True,
            use_container_width=True,
            disabled=[
                "ID",
                "Time",
                "Type",
                "Native Verdict",
                "Action Status",
                "Concern Score",
                "Native Confidence",
                "Model",
                "Source",
                "Preview",
            ],
            column_config={
                "Select": st.column_config.CheckboxColumn("Use", help="Include this evidence item in the next report."),
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Preview": st.column_config.TextColumn("Preview", width="large"),
            },
            key="report_history_editor",
        )
        selected_ids = _selected_ids_from_editor(edited_frame)
        selected_rows = _rows_by_id(filtered_rows, selected_ids)
        report_rows = selected_rows or filtered_rows

        action_a, action_b, action_c = st.columns([0.42, 0.29, 0.29])
        with action_a:
            st.caption(f"{len(report_rows)} record(s) will be included.")
        with action_b:
            if st.button("Delete selected evidence", disabled=not selected_ids, use_container_width=True):
                deleted_rows = _rows_by_id(filtered_rows, selected_ids)
                deleted_count = delete_selected(selected_ids)
                _remove_deleted_from_session(history, deleted_rows)
                st.success(f"Deleted {deleted_count} selected evidence record(s).")
                st.rerun()
        with action_c:
            if st.button("Clear all evidence", use_container_width=True):
                st.session_state["show_clear_all_dialog"] = True

    if st.session_state.get("show_clear_all_dialog"):
        _render_clear_all_confirmation(history)

    with st.expander("Inspect selected evidence details", expanded=False):
        for index, row in enumerate(report_rows, 1):
            st.markdown(
                f"**{index}. {row.get('scan_type', 'Unknown')} - "
                f"{row.get('native_prediction') or row.get('prediction', 'Unknown')}**"
            )
            st.caption(
                f"{str(row.get('scanned_at', '')).replace('T', ' ')[:19]} | "
                f"{row.get('action_status', REVIEW_REQUIRED)} | "
                f"{row.get('score_label', 'Concern score')}: "
                f"{f'{_confidence(row.get('concern_score')):.1f}%' if row.get('score_available') else 'N/A'} | "
                f"{row.get('model_name', '-')}"
            )
            if row.get("preview"):
                st.write(str(row["preview"]))
            if row.get("explanation"):
                st.info(str(row["explanation"]))
            st.divider()

    render_section_header(
        "Report configuration",
        "Pick the file type and sections your examiner or reviewer should see.",
        "Export setup",
    )
    with st.container(border=True):
        config_a, config_b = st.columns([0.34, 0.66])
        with config_a:
            report_profile = st.radio(
                "Report profile",
                options=REPORT_PROFILES,
                index=0,
                help=(
                    "Student Brief prioritizes decisions and learning. "
                    "Technical Report preserves the full forensic-style export."
                ),
            )
            report_format = st.radio(
                "Report format",
                options=["PDF", "DOCX", "TXT"],
                index=0,
                horizontal=True,
            )
            student_mode = report_profile == STUDENT_PROFILE
            section_labels = (
                {
                    "summary": "Investigation summary",
                    "evidence": "Evidence review",
                    "explanations": "Key findings",
                    "risk": "Attention chart",
                    "recommendations": "Next actions",
                    "appendix": "Technical appendix (all saved visuals)",
                }
                if student_mode
                else {
                    "summary": "Executive summary",
                    "evidence": "Evidence table",
                    "explanations": "AI explanations and flags",
                    "risk": "Risk interpretation",
                    "recommendations": "Recommendations",
                    "appendix": "Appendix and scope",
                }
            )
            section_values: dict[str, bool] = {}
            st.write("Sections")
            for section_name, section_label in section_labels.items():
                section_values[section_name] = st.checkbox(
                    section_label,
                    value=DEFAULT_SECTIONS[section_name],
                    key=f"report_section_{section_name}_{report_profile}",
                )
        with config_b:
            _render_risk_chart(report_rows)
            with st.expander("Incident context for precise remediation", expanded=False):
                st.caption(
                    "Select only events that actually occurred. These details add trigger-specific response steps; "
                    "unchecked items are not assumed."
                )
                context_a, context_b = st.columns(2)
                with context_a:
                    link_clicked = st.checkbox("Suspicious link opened")
                    credentials_shared = st.checkbox("Password or credentials shared")
                    otp_shared = st.checkbox("OTP or verification code shared")
                with context_b:
                    software_installed = st.checkbox("Software or remote access installed")
                    funds_transferred = st.checkbox("Money or cryptocurrency transferred")
                    interaction_ongoing = st.checkbox("Contact is still ongoing")
            incident_context = {
                "link_clicked": link_clicked,
                "credentials_shared": credentials_shared,
                "otp_shared": otp_shared,
                "software_installed": software_installed,
                "funds_transferred": funds_transferred,
                "interaction_ongoing": interaction_ongoing,
            }
            report_note = st.text_area(
                "Reviewer note and recommendations",
                value=DEFAULT_RECOMMENDATION,
                height=130,
            )
        with st.expander("Case and documentation details", expanded=False):
            st.caption(
                "These fields provide report identity, purpose, accountability, and evidence disposition."
            )
            metadata_a, metadata_b = st.columns(2)
            with metadata_a:
                case_identifier = st.text_input(
                    "Case identifier",
                    key="report_case_identifier",
                )
                requester = st.text_input(
                    "Requester or course",
                    placeholder="e.g., CEH capstone review",
                )
                report_author = st.text_input(
                    "Report author",
                    placeholder="Student or investigator name",
                )
            with metadata_b:
                purpose_scope = st.text_area(
                    "Purpose and scope",
                    value="Educational review of selected scam-detection evidence and recommended response.",
                    height=92,
                )
                evidence_disposition = st.text_area(
                    "Evidence disposition",
                    value=(
                        "Original inputs remain in local investigation history; "
                        "this report is a derivative educational export."
                    ),
                    height=92,
                )
        incident_context.update(
            {
                "report_profile": report_profile,
                "case_identifier": case_identifier,
                "requester": requester,
                "report_author": report_author,
                "purpose_scope": purpose_scope,
                "evidence_disposition": evidence_disposition,
            }
        )

    preview = build_preview(report_rows, report_note, section_values, incident_context)
    rows_json = _report_rows_json(report_rows)
    sections_json = _sections_json(section_values)
    incident_context_json = _context_json(incident_context)
    with st.container(border=True):
        st.subheader("Report preview")
        st.text_area("Preview text", value=preview, height=250, disabled=True, label_visibility="collapsed")

        report_token = hashlib.sha256(
            (
                f"{REPORT_SCHEMA_VERSION}:{report_format}:{rows_json}:{sections_json}:"
                f"{incident_context_json}:{report_note}"
            ).encode("utf-8")
        ).hexdigest()
        generated_report = st.session_state.get("generated_report")
        generated_is_current = (
            isinstance(generated_report, dict)
            and generated_report.get("token") == report_token
        )

        if st.button(
            f"Generate {report_profile} {report_format} ({len(report_rows)} record(s))",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(f"Building {report_format} report and rendering saved artifacts..."):
                    payload, filename, mime_type = _build_cached_report(
                        str(report_format),
                        rows_json,
                        report_note,
                        sections_json,
                        incident_context_json,
                        REPORT_SCHEMA_VERSION,
                    )
            except Exception as exc:
                st.session_state.pop("generated_report", None)
                st.error(f"Report generation failed: {exc}")
            else:
                generated_report = {
                    "token": report_token,
                    "payload": payload,
                    "filename": filename,
                    "mime_type": mime_type,
                }
                st.session_state["generated_report"] = generated_report
                generated_is_current = True
                render_analysis_ready(f"{report_format} report ready for download")

        if generated_is_current and isinstance(generated_report, dict):
            payload = bytes(generated_report.get("payload", b""))
            filename = str(generated_report.get("filename", ""))
            mime_type = str(
                generated_report.get("mime_type", "application/octet-stream")
            )
            downloaded = st.download_button(
                f"Download {report_format} report",
                data=payload,
                file_name=filename,
                mime=mime_type,
                use_container_width=True,
            )
            download_token = (
                f"{filename}:{len(payload)}:{rows_json}:{sections_json}:"
                f"{incident_context_json}:{report_note}"
            )
            if downloaded and st.session_state.get("last_report_download_token") != download_token:
                scan_ids = [int(row.get("id", 0)) for row in report_rows]
                report_sha256 = hashlib.sha256(payload).hexdigest()
                log_export(
                    report_format=str(report_format),
                    scan_ids=scan_ids,
                    filename=filename,
                    session_id=DEFAULT_SESSION_ID,
                    report_sha256=report_sha256,
                    manifest={
                        "records": len(report_rows),
                        "scan_ids": scan_ids,
                        "sections": section_values,
                        "incident_context": incident_context,
                        "report_sha256": report_sha256,
                    },
                )
                st.session_state["last_report_download_token"] = download_token
                render_analysis_ready(f"{report_format} report download recorded")
        else:
            st.caption(
                "The text preview updates immediately. Generate the file when the configuration is ready; "
                "PDF and DOCX chart rendering runs only on that command."
            )
