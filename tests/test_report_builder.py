import json
import subprocess
import struct
import zlib
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from pypdf import PdfReader

from src.reporting.evidence_snapshot import (
    IMMEDIATE_ACTION,
    build_evidence_bundle,
    chart_artifact,
    provenance_record,
    remediation_plan,
    table_artifact,
    xai_record,
)
from src.reporting.report_builder import (
    DEFAULT_SECTIONS,
    STUDENT_PROFILE,
    _render_artifact_pngs,
    build_report,
)


def _report_row() -> dict[str, object]:
    remediation = remediation_plan(
        evidence_type="Email",
        action_status=IMMEDIATE_ACTION,
        findings=["OTP request"],
    )
    bundle = build_evidence_bundle(
        evidence_type="Email",
        source_input={"source_name": "case.eml"},
        dashboard_summary={
            "final_verdict": "Suspicious",
            "average_suspicious_risk": 88.5,
            "action_status": IMMEDIATE_ACTION,
        },
        artifacts=[
            chart_artifact(
                "Performance Metrics",
                None,
                description="Evaluation tab 1 of 3",
                data=[{"Model": "NB", "F1": 93.2}],
            ),
            chart_artifact("Confusion Matrix Heatmap", None, description="Evaluation tab 2 of 3"),
            chart_artifact("ROC-AUC Curve", None, description="Evaluation tab 3 of 3"),
            table_artifact("Findings", [{"Indicator": "OTP request", "Impact": "High"}]),
        ],
        findings=["OTP request"],
        xai=xai_record(
            method="Local TF-IDF contribution",
            factors=[{"factor": "OTP", "effect": "raises concern", "strength": 0.8}],
            limitations=["Statistical support, not proof of intent."],
        ),
        remediation=remediation,
    )
    return {
        "id": 1,
        "scan_type": "Email",
        "source_name": "case.eml",
        "native_prediction": "Suspicious",
        "prediction": "Suspicious",
        "action_status": IMMEDIATE_ACTION,
        "concern_score": 88.5,
        "score_label": "Average suspicious risk",
        "score_available": 1,
        "confidence": 94.0,
        "model_name": "Email consensus",
        "preview": "Urgent: send your OTP",
        "flags": json.dumps(["OTP request"]),
        "explanation": "Three models agreed.",
        "scanned_at": "2026-07-26T12:00:00",
        "evidence_bundle": json.dumps(bundle),
        "provenance": json.dumps(provenance_record("Urgent: send your OTP", source_name="case.eml")),
    }


def _png(red: int, green: int, blue: int) -> bytes:
    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    raw = b"\x00" + bytes((red, green, blue))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_txt_contains_ordered_artifacts_xai_and_remediation() -> None:
    payload, _filename, _mime = build_report(
        "TXT",
        [_report_row()],
        "Reviewer note",
        DEFAULT_SECTIONS,
        {"credentials_shared": True},
    )
    text = payload.decode("utf-8")

    assert text.index("Performance Metrics") < text.index("Confusion Matrix Heatmap")
    assert text.index("Confusion Matrix Heatmap") < text.index("ROC-AUC Curve")
    assert "Action status: Immediate Action Required" in text
    assert "Explainable analysis" in text
    assert "Change the password from a trusted device" in text
    assert "SHA-256" in text


def test_pdf_and_docx_build_with_snapshot_content() -> None:
    pdf, _pdf_name, pdf_mime = build_report(
        "PDF",
        [_report_row()],
        "Reviewer note",
        DEFAULT_SECTIONS,
    )
    docx, _docx_name, docx_mime = build_report(
        "DOCX",
        [_report_row()],
        "Reviewer note",
        DEFAULT_SECTIONS,
    )

    assert pdf.startswith(b"%PDF")
    assert pdf_mime == "application/pdf"
    assert docx.startswith(b"PK")
    assert "word/document.xml" in __import__("zipfile").ZipFile(__import__("io").BytesIO(docx)).namelist()
    assert "wordprocessingml" in docx_mime


def test_docx_embeds_all_three_saved_metric_tab_charts_in_order() -> None:
    row = _report_row()
    bundle = json.loads(str(row["evidence_bundle"]))
    chart_artifacts = [
        artifact
        for artifact in bundle["artifacts"]
        if artifact.get("kind") == "chart"
    ]
    for index, artifact in enumerate(chart_artifacts, 1):
        artifact["figure_json"] = json.dumps({"chart": index})
        artifact["figure_sha256"] = f"metric-tab-{index}"
    row["evidence_bundle"] = json.dumps(bundle)

    images = {
        "metric-tab-1": _png(220, 38, 38),
        "metric-tab-2": _png(37, 99, 235),
        "metric-tab-3": _png(5, 150, 105),
    }
    with patch(
        "src.reporting.report_builder._render_artifact_pngs",
        return_value=images,
    ):
        docx, _name, _mime = build_report(
            "DOCX",
            [row],
            "Reviewer note",
            DEFAULT_SECTIONS,
        )

    with ZipFile(BytesIO(docx)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        media = [
            name
            for name in archive.namelist()
            if name.startswith("word/media/")
        ]

    assert document_xml.index("Performance Metrics") < document_xml.index(
        "Confusion Matrix Heatmap"
    )
    assert document_xml.index("Confusion Matrix Heatmap") < document_xml.index(
        "ROC-AUC Curve"
    )
    assert len(media) >= 3


def test_chart_batch_timeout_returns_fallback_without_hanging(monkeypatch) -> None:
    workspace_temp = Path("tmp") / "chart-timeout-test"
    workspace_temp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIFDS_CHART_RENDER_TIMEOUT", "10")
    row = _report_row()
    bundle = json.loads(str(row["evidence_bundle"]))
    bundle["artifacts"][0]["figure_json"] = "{}"
    bundle["artifacts"][0]["figure_sha256"] = "timeout-chart"
    row["evidence_bundle"] = json.dumps(bundle)

    class FixedTemporaryDirectory:
        def __enter__(self):
            return str(workspace_temp)

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

    with (
        patch(
            "src.reporting.report_builder.tempfile.TemporaryDirectory",
            return_value=FixedTemporaryDirectory(),
        ),
        patch(
            "src.reporting.report_builder.subprocess.run",
            side_effect=subprocess.TimeoutExpired("chart-renderer", 10),
        ) as run,
    ):
        assert _render_artifact_pngs([row]) == {}

    assert run.call_args.kwargs["timeout"] == 10


def test_student_txt_prioritizes_actions_and_deduplicates_artifacts() -> None:
    duplicate = _report_row()
    duplicate["id"] = 2
    payload, _filename, _mime = build_report(
        "TXT",
        [_report_row(), duplicate],
        "Reviewer note",
        DEFAULT_SECTIONS,
        {"report_profile": STUDENT_PROFILE},
    )
    text = payload.decode("utf-8")

    assert text.index("What should I do next?") < text.index("Evidence Review")
    assert "Result: Suspicious" in text
    assert "Attention: Immediate Action Required" in text
    assert "Analysis method: Email consensus" in text
    assert "Dashboard Summary" not in text
    assert "Evidence Provenance" not in text
    assert "Saved Dashboard Artifacts" not in text
    assert text.index("Combined Investigation Findings") < text.index(
        "Visual Evidence Summary"
    )
    assert text.index("Visual Evidence Summary") < text.index("Reviewer Note")
    assert text.index("Reviewer Note") < text.index("Technical Appendix")
    evidence_specific = text.index(
        "Evidence-Specific Supporting Visuals and Data"
    )
    shared_metrics = text.index("Shared Model Reference Metrics")
    appendix_evidence = text.index(
        "Evidence 1 - Email",
        evidence_specific,
        shared_metrics,
    )
    assert appendix_evidence < text.index(
        "Supporting Visuals and Data",
        appendix_evidence,
        shared_metrics,
    )
    assert text.index("[Table] Findings", appendix_evidence) < shared_metrics
    assert "Used by: Evidence 1 - Email, Evidence 2 - Email" in text
    assert text.count("Performance Metrics") == 1
    assert text.count("Confusion Matrix Heatmap") == 1
    assert text.count("ROC-AUC Curve") == 1


def test_student_pdf_and_docx_use_compact_evidence_structure() -> None:
    context = {
        "report_profile": STUDENT_PROFILE,
        "case_identifier": "AIFDS-TEST-001",
    }
    pdf, _pdf_name, _pdf_mime = build_report(
        "PDF",
        [_report_row()],
        "Reviewer note",
        DEFAULT_SECTIONS,
        context,
    )
    docx, _docx_name, _docx_mime = build_report(
        "DOCX",
        [_report_row()],
        "Reviewer note",
        DEFAULT_SECTIONS,
        context,
    )

    pdf_text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(pdf)).pages
    )
    with ZipFile(BytesIO(docx)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    for content in (pdf_text, document_xml):
        assert "AI-FDS Student Investigation Brief" in content
        assert "What should I do next?" in content
        assert "Evidence Review" in content
        assert "Combined Investigation Findings" in content
        assert "Visual Evidence Summary" in content
        assert "Evidence-Specific Supporting Visuals and Data" in content
        assert "Supporting Visuals and Data" in content
        assert "Shared Model Reference Metrics" in content
        assert "Dashboard Summary" not in content
        assert "Evidence Provenance" not in content
        assert "Saved Dashboard Artifacts" not in content
