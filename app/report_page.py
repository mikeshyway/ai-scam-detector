"""AI report generator page."""

from __future__ import annotations

import json
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
    build_preview,
    build_report,
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
    high_terms = ("suspicious", "scam", "phishing", "high risk", "ai-generated", "chunk")
    lower_terms = ("legitimate", "lower risk", "real human", "safe")
    high_count = sum(1 for row in rows if any(term in str(row.get("prediction", "")).lower() for term in high_terms))
    lower_count = sum(1 for row in rows if any(term in str(row.get("prediction", "")).lower() for term in lower_terms))
    average = sum(_confidence(row.get("confidence")) for row in rows) / len(rows) if rows else 0
    types = len(_unique(rows, "scan_type"))
    return [
        {"label": "Evidence Items", "value": len(rows), "color": "#0891B2"},
        {"label": "High Risk Signals", "value": high_count, "color": "#DC2626"},
        {"label": "Lower Risk Signals", "value": lower_count, "color": "#059669"},
        {"label": "Average Confidence", "value": f"{average:.0f}%", "color": "#D97706"},
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
                "Prediction": row.get("prediction", "-"),
                "Confidence": f"{_confidence(row.get('confidence')):.1f}%",
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
        if history_fingerprint(item) not in deleted_fingerprints
    ]


def _no_result_message(
    *,
    all_rows: list[dict[str, object]],
    date_from: date,
    date_to: date,
    selected_types: list[str],
    selected_predictions: list[str],
) -> str:
    if date_from > date_to:
        return "No result: the start date is after the end date. Choose a valid date range."
    if not all_rows:
        return "No result: no scan evidence has been saved yet. Run a scan first, then return to this page."
    if not selected_types:
        return "No result: no scan type is selected."
    if not selected_predictions:
        return "No result: no prediction type is selected."

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

    return "No result: the selected prediction type has no saved scans for the current date range and scan type."


def _render_risk_chart(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    chart_rows = pd.DataFrame(
        {
            "scan_type": [row.get("scan_type", "Unknown") for row in rows],
            "confidence": [_confidence(row.get("confidence")) for row in rows],
            "prediction": [row.get("prediction", "Unknown") for row in rows],
        }
    )
    fig = px.bar(
        chart_rows,
        x="scan_type",
        y="confidence",
        color="prediction",
        title="Selected evidence confidence overview",
        labels={"scan_type": "Scan type", "confidence": "Confidence (%)", "prediction": "Prediction"},
        color_discrete_sequence=["#0891B2", "#DC2626", "#D97706", "#059669", "#2563EB"],
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=35), yaxis=dict(range=[0, 100]))
    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)


def _report_rows_json(rows: list[dict[str, object]]) -> str:
    return json.dumps(rows, sort_keys=True, default=str, ensure_ascii=True)


def _sections_json(sections: dict[str, bool]) -> str:
    return json.dumps(sections, sort_keys=True, ensure_ascii=True)


@st.cache_data(show_spinner=False)
def _build_live_report(
    report_format: str,
    rows_json: str,
    report_note: str,
    sections_json: str,
) -> tuple[bytes, str, str]:
    rows = json.loads(rows_json)
    sections = json.loads(sections_json)
    return build_report(report_format, rows, report_note, sections)


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
        prediction_options = _unique(all_rows, "prediction")
        col_d, col_e = st.columns(2)
        with col_d:
            selected_types = st.multiselect("Scan types", type_options, default=type_options)
        with col_e:
            selected_predictions = st.multiselect("Predictions", prediction_options, default=prediction_options)

    if date_from <= date_to and selected_types and selected_predictions:
        filtered_rows = query_history(
            session_id=DEFAULT_SESSION_ID,
            date_from=str(date_from),
            date_to=str(date_to),
            scan_types=selected_types,
            predictions=selected_predictions,
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
                selected_predictions=selected_predictions,
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
            disabled=["ID", "Time", "Type", "Prediction", "Confidence", "Model", "Source", "Preview"],
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
            st.markdown(f"**{index}. {row.get('scan_type', 'Unknown')} - {row.get('prediction', 'Unknown')}**")
            st.caption(
                f"{str(row.get('scanned_at', '')).replace('T', ' ')[:19]} | "
                f"{_confidence(row.get('confidence')):.1f}% | {row.get('model_name', '-')}"
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
            report_format = st.radio(
                "Report format",
                options=["PDF", "DOCX", "TXT"],
                index=0,
                horizontal=True,
            )
            section_values: dict[str, bool] = {}
            st.write("Sections")
            section_values["summary"] = st.checkbox("Executive summary", value=DEFAULT_SECTIONS["summary"])
            section_values["evidence"] = st.checkbox("Evidence table", value=DEFAULT_SECTIONS["evidence"])
            section_values["explanations"] = st.checkbox("AI explanations and flags", value=DEFAULT_SECTIONS["explanations"])
            section_values["risk"] = st.checkbox("Risk interpretation", value=DEFAULT_SECTIONS["risk"])
            section_values["recommendations"] = st.checkbox("Recommendations", value=DEFAULT_SECTIONS["recommendations"])
            section_values["appendix"] = st.checkbox("Appendix and scope", value=DEFAULT_SECTIONS["appendix"])
        with config_b:
            _render_risk_chart(report_rows)
            report_note = st.text_area(
                "Reviewer note and recommendations",
                value=DEFAULT_RECOMMENDATION,
                height=130,
            )

    preview = build_preview(report_rows, report_note, section_values)
    rows_json = _report_rows_json(report_rows)
    sections_json = _sections_json(section_values)
    with st.container(border=True):
        st.subheader("Report preview")
        try:
            payload, filename, mime_type = _build_live_report(
                str(report_format),
                rows_json,
                report_note,
                sections_json,
            )
        except Exception as exc:
            payload = b""
            filename = ""
            mime_type = "application/octet-stream"
            st.error(f"Live report update failed: {exc}")
        else:
            st.caption(
                f"Live {report_format} export ready: {filename}. "
                "Changing format, sections, selected evidence, or notes updates this file automatically."
            )

        st.text_area("Preview text", value=preview, height=250, disabled=True, label_visibility="collapsed")

        if payload:
            downloaded = st.download_button(
                f"Download {report_format} report now ({len(report_rows)} record(s))",
                data=payload,
                file_name=filename,
                mime=mime_type,
                type="primary",
                use_container_width=True,
            )
            download_token = f"{filename}:{len(payload)}:{rows_json}:{sections_json}:{report_note}"
            if downloaded and st.session_state.get("last_report_download_token") != download_token:
                scan_ids = [int(row.get("id", 0)) for row in report_rows]
                log_export(
                    report_format=str(report_format),
                    scan_ids=scan_ids,
                    filename=filename,
                    session_id=DEFAULT_SESSION_ID,
                )
                st.session_state["last_report_download_token"] = download_token
                render_analysis_ready(f"{report_format} report download recorded")
