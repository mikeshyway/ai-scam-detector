import inspect
from unittest.mock import patch

from app.report_page import (
    REPORT_VERDICT_OPTIONS,
    _auto_download_markup,
    _history_frame,
    _render_risk_chart,
    _reportable_rows,
    _rows_by_id,
    render_report_page,
)
from src.reporting.evidence_snapshot import (
    IMMEDIATE_ACTION,
    NO_IMMEDIATE_ACTION,
    REVIEW_REQUIRED,
)


def _row(
    scan_id: int,
    scan_type: str,
    action_status: str,
    native_prediction: str = "Suspicious",
) -> dict[str, object]:
    return {
        "id": scan_id,
        "scan_type": scan_type,
        "action_status": action_status,
        "native_prediction": native_prediction,
        "scanned_at": "2026-07-27T00:00:00",
        "score_available": False,
    }


def test_report_generator_uses_only_three_student_verdicts() -> None:
    assert REPORT_VERDICT_OPTIONS == [
        NO_IMMEDIATE_ACTION,
        IMMEDIATE_ACTION,
        REVIEW_REQUIRED,
    ]


def test_report_generator_excludes_simulations_and_requires_selection() -> None:
    rows = [
        _row(1, "Email", IMMEDIATE_ACTION),
        _row(2, "Simulation", REVIEW_REQUIRED),
    ]

    assert [row["id"] for row in _reportable_rows(rows)] == [1]
    assert _rows_by_id(rows, []) == []


def test_saved_evidence_table_uses_student_verdict_labels() -> None:
    frame = _history_frame(
        [_row(1, "Email", NO_IMMEDIATE_ACTION, native_prediction="Legitimate")]
    )

    assert frame.loc[0, "Native Verdict"] == NO_IMMEDIATE_ACTION
    assert "Action Status" not in frame.columns


def test_report_generator_ui_omits_removed_controls() -> None:
    source = inspect.getsource(render_report_page)

    assert "Inspect selected evidence details" not in source
    assert "Technical Report" not in source
    assert "Incident context for precise remediation" not in source
    assert "Requester or course" not in source
    assert "Report author" not in source
    assert "Evidence disposition" not in source
    assert "Case File Remarks" in source
    assert "Generate Report {report_format}" in source
    assert "st.download_button" not in source


def test_empty_selection_keeps_an_empty_action_overview_chart() -> None:
    with (
        patch("app.report_page.apply_chart_theme", side_effect=lambda fig: fig),
        patch("app.report_page.st.plotly_chart") as plotly_chart,
    ):
        _render_risk_chart([])

    figure = plotly_chart.call_args.args[0]
    assert len(figure.data) == 0
    assert figure.layout.title.text == "Selected evidence action overview"
    assert list(figure.layout.xaxis.range) == [0, 100]
    assert figure.layout.yaxis.title.text == "Evidence"


def test_auto_download_markup_clicks_a_named_browser_download() -> None:
    markup = _auto_download_markup(
        "Automatic report download 3",
        3,
    )

    assert 'const buttonLabel = "Automatic report download 3"' in markup
    assert 'const triggerKey = "aifds-report-download-3"' in markup
    assert 'window.parent.document.querySelectorAll("button")' in markup
    assert "target.click()" in markup
