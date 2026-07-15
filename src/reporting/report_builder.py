"""Report builders for the AI-FDS report generator page."""

from __future__ import annotations

import html
import io
import json
from collections import Counter
from typing import Literal

from src.utils.time_utils import formatted_now, now_for_app


THEME = {
    "navy": "#0B1220",
    "surface": "#111827",
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "violet": "#7C3AED",
    "green": "#059669",
    "orange": "#F97316",
    "red": "#DC2626",
    "muted": "#475569",
    "border": "#CBD5E1",
    "soft_blue": "#DBEAFE",
    "soft_violet": "#EDE9FE",
    "soft_orange": "#FFEDD5",
    "soft_green": "#DCFCE7",
}


DEFAULT_RECOMMENDATION = (
    "Verify suspicious contact through official channels, do not share OTP/passwords, "
    "preserve the evidence, and report urgent financial or identity-related requests "
    "to the relevant campus or organisation support team."
)

DEFAULT_SECTIONS = {
    "summary": True,
    "evidence": True,
    "explanations": True,
    "risk": True,
    "recommendations": True,
    "appendix": True,
}


def _text(value: object, fallback: str = "-") -> str:
    if value is None:
        return fallback
    value_text = str(value).strip()
    return value_text or fallback


def _percent(value: object) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _date_text(value: object) -> str:
    text = _text(value)
    return text.replace("T", " ")[:19]


def _flags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return _flags(json.loads(value))
        except json.JSONDecodeError:
            return [value] if value.strip() else []
    if isinstance(value, list):
        return [_text(item, "") for item in value if _text(item, "")]
    return [_text(value)]


def _evidence_family(row: dict[str, object]) -> str:
    text = " ".join(
        [
            _text(row.get("scan_type"), ""),
            _text(row.get("source_name"), ""),
            _text(row.get("model_name"), ""),
            _text(row.get("prediction"), ""),
        ]
    ).lower()
    if any(term in text for term in ("phone", "caller", "omkar", "carrier", "fallback dataset")):
        return "Phone"
    if any(term in text for term in ("audio", "voice", "deepfake", "mfcc", "speaker", "recording")):
        return "Audio"
    if any(term in text for term in ("transcript", "call", "meeting", "whisper")):
        return "Transcript"
    if any(term in text for term in ("email", "message", "mail", "sms")):
        return "Email"
    return _text(row.get("scan_type"), "Evidence")


def _is_unknown_result(row: dict[str, object]) -> bool:
    prediction = _text(row.get("prediction"), "").lower()
    if any(term in prediction for term in ("unknown", "unavailable", "not found", "reputation unknown")):
        return True
    return _evidence_family(row) == "Phone" and _percent(row.get("confidence")) <= 0


def _score_available(row: dict[str, object]) -> bool:
    if _is_unknown_result(row):
        return False
    if _evidence_family(row) == "Phone":
        return False
    return _percent(row.get("confidence")) > 0


def _score_text(row: dict[str, object]) -> str:
    return f"{_percent(row.get('confidence')):.1f}%" if _score_available(row) else "N/A"


def _risk_bucket(row: dict[str, object]) -> str:
    prediction = _text(row.get("prediction"), "").lower()
    confidence = _percent(row.get("confidence"))
    if _is_unknown_result(row):
        return "Unknown"
    if any(term in prediction for term in ("legitimate", "lower risk", "real human", "benign", "safe")):
        return "Lower concern"
    if any(term in prediction for term in ("suspicious", "scam", "phishing", "ai-generated", "high risk", "chunk")):
        return "High concern" if confidence >= 65 else "Needs review"
    if confidence >= 75:
        return "High concern"
    if confidence >= 40:
        return "Needs review"
    return "Lower concern"


def _risk_counts(rows: list[dict[str, object]]) -> Counter:
    return Counter(_risk_bucket(row) for row in rows)


def _prediction_color(prediction: object) -> str:
    text = _text(prediction, "").lower()
    if any(term in text for term in ("unknown", "unavailable", "not found")):
        return THEME["orange"]
    if any(term in text for term in ("lower risk", "legitimate", "real human", "safe")):
        return THEME["green"]
    if any(term in text for term in ("suspicious", "scam", "phishing", "ai-generated", "high risk", "chunk")):
        return THEME["red"]
    return THEME["orange"]


def _evidence_counts(rows: list[dict[str, object]]) -> Counter:
    return Counter(_evidence_family(row) for row in rows)


def _short_source(row: dict[str, object]) -> str:
    source = _text(row.get("source_name"), "")
    if source:
        return source[:70]
    family = _evidence_family(row)
    if family == "Phone":
        return "Omkar lookup / local fallback"
    if family == "Audio":
        return "Recorded or uploaded audio"
    if family == "Transcript":
        return "Uploaded or pasted transcript"
    if family == "Email":
        return "Uploaded or pasted message"
    return "-"


def _engine_text(row: dict[str, object]) -> str:
    engine = _text(row.get("model_name"), "")
    if engine:
        return engine
    if _evidence_family(row) == "Phone":
        return "Omkar + local fallback"
    return "-"


def _indicator_label(value: str) -> str:
    text = " ".join(value.replace("_", " ").split()).strip(" .:-")
    if not text:
        return "Evidence indicator"
    if len(text) > 54:
        text = text[:51].rstrip() + "..."
    return text[:1].upper() + text[1:]


def _indicator_counts(rows: list[dict[str, object]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        for flag in _flags(row.get("flags")):
            counts[_indicator_label(flag)] += 1
        if _is_unknown_result(row) and _evidence_family(row) == "Phone":
            counts["Phone ownership unconfirmed"] += 1
    return counts


def _combined_findings(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    evidence_by_indicator: dict[str, set[str]] = {}
    severity_by_indicator: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        family = _evidence_family(row)
        bucket = _risk_bucket(row)
        labels = [_indicator_label(flag) for flag in _flags(row.get("flags"))]
        if not labels and _is_unknown_result(row) and family == "Phone":
            labels = ["Phone ownership unconfirmed"]
        for label in labels:
            evidence_by_indicator.setdefault(label, set()).add(f"{family} #{index}")
            current = severity_by_indicator.get(label, "Informative")
            severity_by_indicator[label] = _stronger_severity(current, _severity_from_bucket(bucket))

    rows_out = []
    for label, evidence_set in sorted(evidence_by_indicator.items(), key=lambda item: (-len(item[1]), item[0]))[:10]:
        rows_out.append(
            {
                "Finding": label,
                "Evidence involved": ", ".join(sorted(evidence_set)),
                "Severity": severity_by_indicator.get(label, "Informative"),
            }
        )
    return rows_out


def _severity_from_bucket(bucket: str) -> str:
    if bucket == "High concern":
        return "High"
    if bucket == "Needs review":
        return "Review"
    if bucket == "Unknown":
        return "Unknown"
    return "Informative"


def _stronger_severity(left: str, right: str) -> str:
    order = {"Informative": 0, "Unknown": 1, "Review": 2, "High": 3}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _summary_rows(rows: list[dict[str, object]]) -> list[tuple[str, str]]:
    counts = _risk_counts(rows)
    evidence_counts = _evidence_counts(rows)
    return [
        ("Evidence selected", str(len(rows))),
        ("Email evidence", str(evidence_counts.get("Email", 0))),
        ("Transcript evidence", str(evidence_counts.get("Transcript", 0))),
        ("Audio evidence", str(evidence_counts.get("Audio", 0))),
        ("Phone evidence", str(evidence_counts.get("Phone", 0))),
        ("High concern", str(counts.get("High concern", 0))),
        ("Needs review", str(counts.get("Needs review", 0))),
        ("Lower concern", str(counts.get("Lower concern", 0))),
        ("Unknown", str(counts.get("Unknown", 0))),
    ]


def _confidence_chart_png(rows: list[dict[str, object]]) -> bytes | None:
    """Render available classification scores as PNG for PDF/DOCX exports."""

    score_rows = [(index, row) for index, row in enumerate(rows, 1) if _score_available(row)]
    if not score_rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return None

    labels = [f"{_evidence_family(row)} #{index}" for index, row in score_rows]
    values = [_percent(row.get("confidence")) for _, row in score_rows]
    predictions = [_text(row.get("prediction"), "Unknown") for _, row in score_rows]
    colors = [_prediction_color(prediction) for prediction in predictions]

    fig_height = max(3.2, min(7.5, 0.45 * len(score_rows) + 2.2))
    fig, ax = plt.subplots(figsize=(8.2, fig_height), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(range(len(score_rows)), values, color=colors, height=0.56)
    ax.set_title("Available Classification Scores", fontsize=12, fontweight="bold", color=THEME["navy"], pad=12)
    ax.set_xlabel("Risk / confidence score (%)", fontsize=9, color="#334155")
    ax.set_xlim(0, 100)
    ax.set_yticks(range(len(score_rows)))
    ax.set_yticklabels(labels, fontsize=8.5, color="#334155")
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8, colors="#334155")
    ax.grid(axis="x", color=THEME["border"], linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME["border"])
    ax.spines["bottom"].set_color(THEME["border"])

    legend_items: list[Patch] = []
    seen: set[str] = set()
    for prediction in predictions:
        if prediction in seen:
            continue
        seen.add(prediction)
        legend_items.append(Patch(facecolor=_prediction_color(prediction), label=prediction))
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", fontsize=7.5, frameon=False)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _confidence_chart_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = ["Available Classification Scores"]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"- {_evidence_family(row)} #{index}: {_text(row.get('prediction'))} "
            f"({_score_text(row)})"
        )
    return lines


def _count_chart_png(title: str, counts: Counter, color: str) -> bytes | None:
    if not counts:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    items = counts.most_common(8)
    labels = [label for label, _count in items]
    values = [count for _label, count in items]
    fig_height = max(2.8, min(6.8, 0.38 * len(items) + 1.8))
    fig, ax = plt.subplots(figsize=(8.0, fig_height), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(range(len(items)), values, color=color, height=0.55)
    ax.set_title(title, fontsize=12, fontweight="bold", color=THEME["navy"], pad=12)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=8.2, color="#334155")
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8, colors="#334155")
    ax.grid(axis="x", color=THEME["border"], linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME["border"])
    ax.spines["bottom"].set_color(THEME["border"])
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _evidence_distribution_chart_png(rows: list[dict[str, object]]) -> bytes | None:
    return _count_chart_png("Evidence Type Distribution", _evidence_counts(rows), THEME["violet"])


def _indicator_chart_png(rows: list[dict[str, object]]) -> bytes | None:
    return _count_chart_png("Indicator Categories", _indicator_counts(rows), THEME["orange"])


def _filename(extension: str) -> str:
    stamp = now_for_app().strftime("%Y%m%d_%H%M%S")
    return f"AIFDS_Report_{stamp}.{extension.lower()}"


def _overview_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = ["Investigation Summary"]
    lines.extend(f"- {label}: {value}" for label, value in _summary_rows(rows))
    lines.append("")
    lines.append("Evidence Outcome Distribution")
    counts = _risk_counts(rows)
    for label in ("High concern", "Needs review", "Lower concern", "Unknown"):
        lines.append(f"- {label}: {counts.get(label, 0)}")
    return lines


def _selected_evidence_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = ["Selected Evidence Overview"]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"{index}. {_evidence_family(row)} | Source: {_short_source(row)} | "
            f"Result: {_text(row.get('prediction'))} | Risk/Confidence: {_score_text(row)} | "
            f"Engine: {_engine_text(row)}"
        )
    return lines


def _evidence_specific_action(family: str) -> str:
    actions = {
        "Email": "Verify sender domain, links, attachments, and account requests through official channels.",
        "Transcript": "Check for OTP requests, secrecy, urgency, payment pressure, and impersonation cues.",
        "Audio": "Do not rely on voice familiarity alone; compare voice authenticity with transcript behavior.",
        "Phone": "Carrier metadata does not confirm identity; verify the caller through an official number.",
    }
    return actions.get(family, "Preserve the original evidence and verify through an independent trusted source.")


def _individual_evidence_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = ["Individual Evidence Results"]
    for index, row in enumerate(rows, 1):
        family = _evidence_family(row)
        lines.extend(
            [
                "",
                "-" * 72,
                f"Evidence {index} of {len(rows)} - {family}",
                "-" * 72,
                f"Evidence type: {family}",
                f"Source: {_short_source(row)}",
                f"Result: {_text(row.get('prediction'))}",
                f"Risk/Confidence: {_score_text(row)}",
                f"Engine: {_engine_text(row)}",
            ]
        )
        flags = ", ".join(_flags(row.get("flags")))
        if flags:
            lines.append(f"Detected indicators: {flags}")
        explanation = _text(row.get("explanation"), "")
        if explanation:
            lines.append(f"Explanation: {explanation}")
        preview = _text(row.get("preview"), "")
        if preview:
            lines.append(f"Evidence preview: {preview[:650]}")
        lines.append(f"Recommended action: {_evidence_specific_action(family)}")
    return lines


def _combined_findings_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = ["Combined Investigation Findings"]
    findings = _combined_findings(rows)
    if not findings:
        lines.append("- No combined indicators were available from the selected evidence.")
        return lines
    for item in findings:
        lines.append(
            f"- {item['Finding']} | Evidence involved: {item['Evidence involved']} | "
            f"Severity: {item['Severity']}"
        )
    return lines


def _recommendation_lines(report_note: str) -> list[str]:
    note = report_note.strip() or DEFAULT_RECOMMENDATION
    return [
        "Recommendations",
        "Immediate Actions",
        "- Do not send money or disclose credentials.",
        "- Verify the sender or caller through an official channel.",
        "- Preserve the original files, transcripts, screenshots, and timestamps.",
        "",
        "Evidence-Specific Actions",
        "- Email: Verify sender domains, links, and attachments.",
        "- Transcript: Look for OTP, secrecy, urgency, and payment requests.",
        "- Audio: Do not rely on voice familiarity alone.",
        "- Phone: Carrier information does not confirm caller identity.",
        "",
        "Reviewer Note",
        note,
    ]


def _scope_lines() -> list[str]:
    return [
        "Scope and Limitations",
        "- Email: TF-IDF with trained email classifiers.",
        "- Transcript: TF-IDF text classification after manual input or Whisper transcription.",
        "- Audio: MFCC voice-authenticity analysis, behavioral audio features, and transcript analysis.",
        "- Phone: Omkar carrier metadata, local reputation fallback, and transparent rules.",
        "- This report is an educational capstone prototype output, not legal or forensic proof.",
    ]


def build_preview(rows: list[dict[str, object]], report_note: str, sections: dict[str, bool]) -> str:
    lines = [
        "AI-based Spam and Caller Fraud Detection System",
        "AI Analysis Evidence Report",
        f"Generated: {formatted_now()}",
        f"Records included: {len(rows)}",
        "",
    ]
    if sections.get("summary", True):
        lines.extend(_overview_lines(rows))
        lines.append("")
    if sections.get("evidence", True):
        lines.extend(_selected_evidence_lines(rows))
        lines.append("")
    if sections.get("explanations", True):
        lines.extend(_individual_evidence_lines(rows))
        lines.append("")
        lines.extend(_combined_findings_lines(rows))
        lines.append("")
    if sections.get("risk", True):
        lines.extend(_confidence_chart_lines(rows))
        lines.append("")
        lines.append("Evidence Type Distribution")
        for label, count in _evidence_counts(rows).most_common():
            lines.append(f"- {label}: {count}")
        lines.append("")
        lines.append("Indicator Categories")
        indicator_counts = _indicator_counts(rows)
        if indicator_counts:
            for label, count in indicator_counts.most_common(8):
                lines.append(f"- {label}: {count}")
        else:
            lines.append("- No indicator categories were saved with the selected evidence.")
        lines.append("")
    if sections.get("recommendations", True):
        lines.extend(_recommendation_lines(report_note))
        lines.append("")
    if sections.get("appendix", True):
        lines.extend(_scope_lines())
    return "\n".join(lines)


def build_txt(rows: list[dict[str, object]], report_note: str, sections: dict[str, bool]) -> tuple[bytes, str]:
    body = build_preview(rows, report_note, sections)
    return body.encode("utf-8"), _filename("txt")


def build_pdf(rows: list[dict[str, object]], report_note: str, sections: dict[str, bool]) -> tuple[bytes, str]:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="AI-FDS Analysis Evidence Report",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Muted",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallMono",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#334155"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="AIFDSHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=17,
            textColor=colors.HexColor(THEME["blue"]),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AIFDSSubHeading",
            parent=styles["Heading3"],
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor(THEME["violet"]),
            spaceBefore=8,
            spaceAfter=5,
        )
    )

    def p(value: object, style_name: str = "Muted") -> Paragraph:
        return Paragraph(html.escape(_text(value)), styles[style_name])

    def table_style(header: bool = True, left_band: str = "soft_blue") -> TableStyle:
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(THEME["border"])),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]
        if header:
            commands.extend(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME["navy"])),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        else:
            commands.extend(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(THEME[left_band])),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ]
            )
        return TableStyle(commands)

    story = [
        Paragraph("AI-based Spam and Caller Fraud Detection System", styles["Title"]),
        Paragraph("AI Analysis Evidence Report", styles["AIFDSHeading"]),
        Paragraph(html.escape(f"Generated: {formatted_now()}"), styles["Muted"]),
        Paragraph("Educational capstone prototype. Not legal or forensic proof.", styles["Muted"]),
        Spacer(1, 12),
    ]

    if sections.get("summary", True):
        story.append(Paragraph("1. Investigation Summary", styles["AIFDSHeading"]))
        summary_rows = [[label, value] for label, value in _summary_rows(rows)]
        table = Table(summary_rows, colWidths=[6 * cm, 9 * cm])
        table.setStyle(table_style(header=False, left_band="soft_violet"))
        story.extend([table, Spacer(1, 12)])
        outcome_chart = _count_chart_png("Evidence Outcome Distribution", _risk_counts(rows), THEME["blue"])
        if outcome_chart:
            story.append(Image(io.BytesIO(outcome_chart), width=14.2 * cm, height=5.2 * cm))
            story.append(Spacer(1, 10))

    if sections.get("evidence", True):
        story.append(Paragraph("2. Selected Evidence Overview", styles["AIFDSHeading"]))
        table_rows = [["#", "Evidence", "Source", "Result", "Risk/Conf.", "Engine"]]
        for index, row in enumerate(rows, 1):
            table_rows.append(
                [
                    str(index),
                    _evidence_family(row),
                    _short_source(row)[:34],
                    _text(row.get("prediction")),
                    _score_text(row),
                    _engine_text(row)[:28],
                ]
            )
        table = Table(table_rows, repeatRows=1, colWidths=[0.7 * cm, 2.4 * cm, 4.2 * cm, 3 * cm, 2 * cm, 3.2 * cm])
        table.setStyle(table_style())
        story.extend([table, Spacer(1, 12)])

    if sections.get("explanations", True):
        story.append(Paragraph("3. Individual Evidence Results", styles["AIFDSHeading"]))
        for index, row in enumerate(rows, 1):
            family = _evidence_family(row)
            story.append(Paragraph(html.escape(f"Evidence {index} of {len(rows)} - {family}"), styles["AIFDSSubHeading"]))
            profile_rows = [
                ["Evidence type", family],
                ["Source", _short_source(row)],
                ["Result", _text(row.get("prediction"))],
                ["Risk/Confidence", _score_text(row)],
                ["Engine", _engine_text(row)],
                ["Timestamp", _date_text(row.get("scanned_at"))],
            ]
            profile_table = Table(profile_rows, colWidths=[4 * cm, 11 * cm])
            profile_table.setStyle(table_style(header=False, left_band="soft_blue"))
            story.extend([profile_table, Spacer(1, 5)])
            flags = ", ".join(_flags(row.get("flags")))
            if flags:
                story.append(Paragraph(html.escape(f"Detected indicators: {flags}"), styles["Muted"]))
            explanation = _text(row.get("explanation"), "")
            if explanation:
                story.append(Paragraph(html.escape(explanation), styles["Muted"]))
            preview = _text(row.get("preview"), "")
            if preview:
                story.append(Paragraph(html.escape(preview[:350]), styles["SmallMono"]))
            story.append(Paragraph(html.escape(f"Recommended action: {_evidence_specific_action(family)}"), styles["Muted"]))
            story.append(Spacer(1, 8))

        combined = _combined_findings(rows)
        story.append(Paragraph("Combined Investigation Findings", styles["AIFDSSubHeading"]))
        if combined:
            table_rows = [["Finding", "Evidence involved", "Severity"]] + [
                [item["Finding"], item["Evidence involved"], item["Severity"]] for item in combined
            ]
            table = Table(table_rows, repeatRows=1, colWidths=[6 * cm, 6 * cm, 3 * cm])
            table.setStyle(table_style())
            story.extend([table, Spacer(1, 12)])
        else:
            story.append(Paragraph("No combined indicators were available from the selected evidence.", styles["Muted"]))

    if sections.get("risk", True):
        story.append(Paragraph("4. Visual Evidence Summary", styles["AIFDSHeading"]))
        chart = _confidence_chart_png(rows)
        if chart:
            story.append(Image(io.BytesIO(chart), width=15.2 * cm, height=6.2 * cm))
            story.append(Spacer(1, 8))
        distribution_chart = _evidence_distribution_chart_png(rows)
        if distribution_chart:
            story.append(Image(io.BytesIO(distribution_chart), width=14.2 * cm, height=5.2 * cm))
            story.append(Spacer(1, 8))
        indicator_chart = _indicator_chart_png(rows)
        if indicator_chart:
            story.append(Image(io.BytesIO(indicator_chart), width=14.2 * cm, height=5.2 * cm))
            story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                "Scores across different evidence types are not always equivalent. Phone lookup Unknown is reported as N/A, "
                "because carrier metadata and local fallback evidence are not trained-model probabilities.",
                styles["Muted"],
            )
        )
        story.append(Spacer(1, 12))

    if sections.get("recommendations", True):
        story.append(Paragraph("5. Recommendations", styles["AIFDSHeading"]))
        for line in _recommendation_lines(report_note):
            if line in {"Recommendations", "Immediate Actions", "Evidence-Specific Actions", "Reviewer Note"}:
                if line != "Recommendations":
                    story.append(Paragraph(html.escape(line), styles["AIFDSSubHeading"]))
            elif line:
                story.append(Paragraph(html.escape(line), styles["Muted"]))
            else:
                story.append(Spacer(1, 4))
        story.append(Spacer(1, 12))

    if sections.get("appendix", True):
        story.append(PageBreak())
        story.append(Paragraph("6. Scope and Limitations", styles["AIFDSHeading"]))
        appendix = [
            ["System", "AI-FDS Capstone Prototype"],
            ["Email", "TF-IDF with trained email classifiers"],
            ["Transcript", "TF-IDF text classification after manual input or Whisper transcription"],
            ["Audio", "MFCC voice-authenticity analysis, behavioral audio features, and transcript analysis"],
            ["Phone", "Omkar carrier metadata, local reputation fallback, and transparent rules"],
            ["Scope", "Educational scam awareness support. Not enterprise security, telecom verification, or legal evidence."],
        ]
        table = Table(appendix, colWidths=[4 * cm, 11 * cm])
        table.setStyle(table_style(header=False, left_band="soft_green"))
        story.append(table)

    doc.build(story)
    return buffer.getvalue(), _filename("pdf")


def build_docx(rows: list[dict[str, object]], report_note: str, sections: dict[str, bool]) -> tuple[bytes, str]:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    document = Document()
    title = document.add_heading("AI-based Spam and Caller Fraud Detection System", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if title.runs:
        title.runs[0].font.color.rgb = RGBColor(37, 99, 235)
    subtitle = document.add_paragraph("AI Analysis Evidence Report")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if subtitle.runs:
        subtitle.runs[0].font.color.rgb = RGBColor(124, 58, 237)
        subtitle.runs[0].font.bold = True
    meta = document.add_paragraph(f"Generated: {formatted_now()}\nRecords included: {len(rows)}\nEducational capstone prototype. Not legal or forensic proof.")
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def heading(text: str, level: int = 1) -> None:
        paragraph = document.add_heading(text, level=level)
        if paragraph.runs:
            paragraph.runs[0].font.color.rgb = (
                RGBColor(37, 99, 235)
                if level == 1
                else RGBColor(124, 58, 237)
            )

    def body(text: str) -> None:
        paragraph = document.add_paragraph(text)
        for run in paragraph.runs:
            run.font.size = Pt(10)

    def shade_cell(cell, fill: str, text_color: RGBColor | None = None, bold: bool = False) -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), fill)
        tc_pr.append(shading)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                if text_color is not None:
                    run.font.color.rgb = text_color
                run.font.bold = bold

    def set_table_header(table) -> None:
        for cell in table.rows[0].cells:
            shade_cell(cell, "0B1220", RGBColor(255, 255, 255), True)

    def two_column_table(items: list[tuple[str, str]]) -> None:
        table = document.add_table(rows=0, cols=2)
        table.style = "Table Grid"
        for label, value in items:
            cells = table.add_row().cells
            cells[0].text = label
            cells[1].text = value
            shade_cell(cells[0], "EDE9FE", RGBColor(15, 23, 42), True)

    if sections.get("summary", True):
        heading("1. Investigation Summary")
        two_column_table(_summary_rows(rows))
        chart = _count_chart_png("Evidence Outcome Distribution", _risk_counts(rows), THEME["blue"])
        if chart:
            document.add_picture(io.BytesIO(chart), width=Inches(6.1))

    if sections.get("evidence", True):
        heading("2. Selected Evidence Overview")
        table = document.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        for index, label in enumerate(["#", "Evidence", "Source", "Result", "Risk/Conf.", "Engine"]):
            table.rows[0].cells[index].text = label
        set_table_header(table)
        for index, row in enumerate(rows, 1):
            cells = table.add_row().cells
            values = [
                str(index),
                _evidence_family(row),
                _short_source(row)[:60],
                _text(row.get("prediction")),
                _score_text(row),
                _engine_text(row)[:45],
            ]
            for cell_index, value in enumerate(values):
                cells[cell_index].text = value

    if sections.get("explanations", True):
        heading("3. Individual Evidence Results")
        for index, row in enumerate(rows, 1):
            family = _evidence_family(row)
            heading(f"Evidence {index} of {len(rows)} - {family}", level=2)
            two_column_table(
                [
                    ("Evidence type", family),
                    ("Source", _short_source(row)),
                    ("Result", _text(row.get("prediction"))),
                    ("Risk/Confidence", _score_text(row)),
                    ("Engine", _engine_text(row)),
                    ("Timestamp", _date_text(row.get("scanned_at"))),
                ]
            )
            flags = ", ".join(_flags(row.get("flags")))
            if flags:
                body(f"Detected indicators: {flags}")
            explanation = _text(row.get("explanation"), "")
            if explanation:
                body(f"Explanation: {explanation}")
            preview = _text(row.get("preview"), "")
            if preview:
                body(f"Evidence preview: {preview[:350]}")
            body(f"Recommended action: {_evidence_specific_action(family)}")

        combined = _combined_findings(rows)
        heading("Combined Investigation Findings", level=2)
        if combined:
            table = document.add_table(rows=1, cols=3)
            table.style = "Table Grid"
            for index, label in enumerate(["Finding", "Evidence involved", "Severity"]):
                table.rows[0].cells[index].text = label
            set_table_header(table)
            for item in combined:
                cells = table.add_row().cells
                cells[0].text = item["Finding"]
                cells[1].text = item["Evidence involved"]
                cells[2].text = item["Severity"]
        else:
            body("No combined indicators were available from the selected evidence.")

    if sections.get("risk", True):
        heading("4. Visual Evidence Summary")
        chart = _confidence_chart_png(rows)
        if chart:
            document.add_picture(io.BytesIO(chart), width=Inches(6.4))
        distribution_chart = _evidence_distribution_chart_png(rows)
        if distribution_chart:
            document.add_picture(io.BytesIO(distribution_chart), width=Inches(6.1))
        indicator_chart = _indicator_chart_png(rows)
        if indicator_chart:
            document.add_picture(io.BytesIO(indicator_chart), width=Inches(6.1))
        body(
            "Scores across different evidence types are not always equivalent. Phone lookup Unknown is reported as N/A, "
            "because carrier metadata and local fallback evidence are not trained-model probabilities."
        )

    if sections.get("recommendations", True):
        heading("5. Recommendations")
        for line in _recommendation_lines(report_note):
            if line in {"Recommendations"}:
                continue
            if line in {"Immediate Actions", "Evidence-Specific Actions", "Reviewer Note"}:
                heading(line, level=2)
            elif line.startswith("- "):
                document.add_paragraph(line[2:], style="List Bullet")
            elif line:
                body(line)

    if sections.get("appendix", True):
        document.add_page_break()
        heading("6. Scope and Limitations")
        two_column_table(
            [
                ("System", "AI-FDS Capstone Prototype"),
                ("Email", "TF-IDF with trained email classifiers"),
                ("Transcript", "TF-IDF text classification after manual input or Whisper transcription"),
                ("Audio", "MFCC voice-authenticity analysis, behavioral audio features, and transcript analysis"),
                ("Phone", "Omkar carrier metadata, local reputation fallback, and transparent rules"),
                ("Scope", "Educational scam awareness support. Not enterprise security, telecom verification, or legal evidence."),
            ]
        )

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue(), _filename("docx")


def build_report(
    report_format: Literal["TXT", "PDF", "DOCX"],
    rows: list[dict[str, object]],
    report_note: str,
    sections: dict[str, bool],
) -> tuple[bytes, str, str]:
    """Build a report and return bytes, filename, and MIME type."""

    if report_format == "TXT":
        payload, filename = build_txt(rows, report_note, sections)
        return payload, filename, "text/plain"
    if report_format == "PDF":
        payload, filename = build_pdf(rows, report_note, sections)
        return payload, filename, "application/pdf"
    if report_format == "DOCX":
        payload, filename = build_docx(rows, report_note, sections)
        return payload, filename, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    raise ValueError(f"Unsupported report format: {report_format}")
