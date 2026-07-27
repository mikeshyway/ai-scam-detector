from src.reporting.evidence_snapshot import (
    IMMEDIATE_ACTION,
    NO_IMMEDIATE_ACTION,
    REVIEW_REQUIRED,
    build_evidence_bundle,
    chart_artifact,
    derive_action_status,
    provenance_record,
    remediation_plan,
    table_artifact,
)


def test_action_status_keeps_unknown_and_process_labels_in_review() -> None:
    assert (
        derive_action_status(
            native_prediction="Reputation unknown",
            evidence_type="Phone",
            score_available=False,
        )
        == REVIEW_REQUIRED
    )
    assert (
        derive_action_status(
            native_prediction="Uploaded recording chunk analysis",
            evidence_type="Audio",
            concern_score=99,
            score_available=True,
        )
        == REVIEW_REQUIRED
    )


def test_action_status_uses_evidence_aware_triage() -> None:
    assert (
        derive_action_status(
            native_prediction="Suspicious",
            evidence_type="Email",
            concern_score=84,
            score_available=True,
            model_agreement="3/3",
            evidence_complete=True,
        )
        == IMMEDIATE_ACTION
    )
    assert (
        derive_action_status(
            native_prediction="Suspicious",
            evidence_type="Email",
            concern_score=55,
            score_available=True,
            model_agreement="2/3",
            evidence_complete=True,
        )
        == REVIEW_REQUIRED
    )
    assert (
        derive_action_status(
            native_prediction="Legitimate",
            evidence_type="Transcript",
            concern_score=8,
            score_available=True,
            evidence_complete=True,
        )
        == NO_IMMEDIATE_ACTION
    )


def test_bundle_preserves_artifact_order_and_provenance_hash() -> None:
    artifacts = [
        chart_artifact("Performance Metrics", None, description="Tab 1"),
        chart_artifact("Confusion Matrix Heatmap", None, description="Tab 2"),
        chart_artifact("ROC-AUC Curve", None, description="Tab 3"),
        table_artifact("Evidence table", [{"Finding": "Urgency"}]),
    ]
    bundle = build_evidence_bundle(
        evidence_type="Email",
        source_input={"source_name": "sample.eml"},
        dashboard_summary={"final_verdict": "Suspicious"},
        artifacts=artifacts,
    )
    assert [item["title"] for item in bundle["artifacts"]] == [
        "Performance Metrics",
        "Confusion Matrix Heatmap",
        "ROC-AUC Curve",
        "Evidence table",
    ]

    first = provenance_record("same input", source_name="sample.eml")
    second = provenance_record("same input", source_name="renamed.eml")
    assert first["sha256"] == second["sha256"]
    assert len(first["sha256"]) == 64

    binary = provenance_record(b"original audio bytes", source_name="call.wav")
    assert binary["hash_scope"] == "Raw binary input bytes"
    assert binary["sha256"] != first["sha256"]


def test_remediation_uses_only_declared_exposure_context() -> None:
    baseline = remediation_plan(
        evidence_type="Email",
        action_status=REVIEW_REQUIRED,
        findings=["urgent account verification"],
    )
    exposed = remediation_plan(
        evidence_type="Email",
        action_status=REVIEW_REQUIRED,
        findings=["urgent account verification"],
        incident_context={"credentials_shared": True},
    )
    assert not any("Change the password" in item["action"] for item in baseline)
    assert any("Change the password" in item["action"] for item in exposed)
