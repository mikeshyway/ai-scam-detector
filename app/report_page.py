"""AI report generator page."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
    REPORT_SCHEMA_VERSION,
    STUDENT_PROFILE,
    build_preview,
    build_report,
)
from src.reporting.evidence_snapshot import (
    IMMEDIATE_ACTION,
    NO_IMMEDIATE_ACTION,
    REVIEW_REQUIRED,
)
from src.utils.time_utils import now_for_app


REPORT_VERDICT_OPTIONS = [
    NO_IMMEDIATE_ACTION,
    IMMEDIATE_ACTION,
    REVIEW_REQUIRED,
]


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


def _reportable_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("scan_type", "")).strip().casefold() != "simulation"
    ]


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
                "Native Verdict": row.get("action_status", REVIEW_REQUIRED),
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
    selected_verdicts: list[str],
) -> str:
    if date_from > date_to:
        return "No result: the start date is after the end date. Choose a valid date range."
    if not all_rows:
        return "No result: no scan evidence has been saved yet. Run a scan first, then return to this page."
    if not selected_types:
        return "No result: no scan type is selected."
    if not selected_verdicts:
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

    return "No result: the selected native verdict has no saved scans for the current filters."


def _render_risk_chart(rows: list[dict[str, object]]) -> None:
    if not rows:
        fig = go.Figure()
        fig.update_layout(
            title="Selected evidence action overview",
            height=280,
            margin=dict(l=10, r=10, t=40, b=35),
            showlegend=False,
            xaxis=dict(
                range=[0, 100],
                title="Source-native concern score (%)",
            ),
            yaxis=dict(
                title="Evidence",
                showticklabels=False,
            ),
        )
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
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
                "Source Result": row.get("native_prediction") or row.get("prediction", "Unknown"),
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
            "Source Result": True,
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


def _auto_download_markup(
    button_label: str,
    nonce: int,
) -> str:
    return f"""
    <script>
    (() => {{
      const buttonLabel = {json.dumps(button_label)};
      const triggerKey = {json.dumps(f"aifds-report-download-{nonce}")};
      let attempts = 0;
      const clickDownload = () => {{
        const buttons = Array.from(
          window.parent.document.querySelectorAll("button")
        );
        const target = buttons.find(
          (button) => button.textContent.trim() === buttonLabel
        );
        if (target) {{
          target.dataset.aifdsAutoDownload = triggerKey;
          target.click();
          return;
        }}
        attempts += 1;
        if (attempts < 20) {{
          window.setTimeout(clickDownload, 100);
        }}
      }};
      clickDownload();
    }})();
    </script>
    """


def _trigger_browser_download(
    payload: bytes,
    filename: str,
    mime_type: str,
    nonce: int,
) -> None:
    button_label = f"Automatic report download {nonce}"
    container_key = f"aifds_auto_download_{nonce}"
    with st.container(key=container_key):
        st.download_button(
            button_label,
            data=payload,
            file_name=filename,
            mime=mime_type,
            key=f"{container_key}_button",
            on_click="ignore",
        )
    st.markdown(
        f"<style>.st-key-{container_key}{{display:none!important;}}</style>",
        unsafe_allow_html=True,
    )
    components.html(
        _auto_download_markup(button_label, nonce),
        height=0,
        scrolling=False,
    )


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
    st.session_state.pop("generated_report", None)
    st.session_state.pop("last_report_download_token", None)
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

    all_rows = _reportable_rows(query_history(session_id=DEFAULT_SESSION_ID))
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
        col_d, col_f = st.columns(2)
        with col_d:
            selected_types = st.multiselect("Scan types", type_options, default=type_options)
        with col_f:
            selected_verdicts = st.multiselect(
                "Native verdict",
                REPORT_VERDICT_OPTIONS,
                default=REPORT_VERDICT_OPTIONS,
                help="Student-facing attention level used consistently throughout the report.",
            )

    if (
        date_from <= date_to
        and selected_types
        and selected_verdicts
    ):
        filtered_rows = query_history(
            session_id=DEFAULT_SESSION_ID,
            date_from=str(date_from),
            date_to=str(date_to),
            scan_types=selected_types,
            action_statuses=selected_verdicts,
        )
    else:
        filtered_rows = []

    render_section_header(
        "Saved scan evidence",
        "Select one or more evidence records to include in the report.",
        "Report input",
    )

    if not filtered_rows:
        render_info_banner(
            _no_result_message(
                all_rows=all_rows,
                date_from=date_from,
                date_to=date_to,
                selected_types=selected_types,
                selected_verdicts=selected_verdicts,
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
        report_rows = selected_rows

        action_a, action_b, action_c = st.columns([0.42, 0.29, 0.29])
        with action_a:
            if report_rows:
                st.caption(f"{len(report_rows)} selected record(s) will be included.")
            else:
                st.caption("No evidence selected.")
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

    render_section_header(
        "Report configuration",
        "Pick the file type and sections your examiner or reviewer should see.",
        "Export setup",
    )
    with st.container(border=True):
        config_a, config_b = st.columns([0.34, 0.66])
        with config_a:
            report_profile = STUDENT_PROFILE
            st.caption("Report profile")
            st.markdown("**Student Brief**")
            report_format = st.radio(
                "Report format",
                options=["PDF", "DOCX", "TXT"],
                index=0,
                horizontal=True,
            )
            section_labels = {
                "summary": "Investigation summary",
                "evidence": "Evidence review",
                "explanations": "Key findings",
                "risk": "Attention chart",
                "recommendations": "Next actions",
                "appendix": "Technical appendix (all saved visuals)",
            }
            section_values: dict[str, bool] = {}
            st.write("Sections")
            for section_name, section_label in section_labels.items():
                section_values[section_name] = st.checkbox(
                    section_label,
                    value=DEFAULT_SECTIONS[section_name],
                    key=f"report_section_{section_name}_student",
                )
        with config_b:
            _render_risk_chart(report_rows)
        with st.expander("Case File Remarks", expanded=False):
            metadata_a, metadata_b = st.columns(2)
            with metadata_a:
                case_identifier = st.text_input(
                    "Case identifier",
                    key="report_case_identifier",
                )
            with metadata_b:
                purpose_scope = st.text_area(
                    "Purpose and scope",
                    value="Educational review of selected scam-detection evidence and recommended response.",
                    height=92,
                )
            report_note = st.text_area(
                "Reviewer note and recommendations",
                value=DEFAULT_RECOMMENDATION,
                height=130,
            )
        incident_context = {
            "report_profile": report_profile,
            "case_identifier": case_identifier,
            "purpose_scope": purpose_scope,
        }

    preview = (
        build_preview(report_rows, report_note, section_values, incident_context)
        if report_rows
        else ""
    )
    rows_json = _report_rows_json(report_rows)
    sections_json = _sections_json(section_values)
    incident_context_json = _context_json(incident_context)
    with st.container(border=True):
        st.subheader("Report preview")
        st.text_area("Preview text", value=preview, height=250, disabled=True, label_visibility="collapsed")

        generate_clicked = st.button(
            f"Generate Report {report_format}",
            type="primary",
            use_container_width=True,
            disabled=not report_rows,
        )
        if generate_clicked:
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
                st.error(f"Report generation failed: {exc}")
            else:
                download_nonce = int(
                    st.session_state.get("report_download_nonce", 0)
                ) + 1
                st.session_state["report_download_nonce"] = download_nonce
                _trigger_browser_download(
                    payload,
                    filename,
                    mime_type,
                    download_nonce,
                )
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
                render_analysis_ready(
                    f"{report_format} report generated; download started"
                )
        else:
            if report_rows:
                st.caption(
                    f"Generate the selected evidence as a downloadable {report_format} report."
                )
            else:
                st.caption("Select at least one saved evidence record to generate a report.")
