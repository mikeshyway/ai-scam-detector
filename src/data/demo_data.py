"""Curated, non-personal demo evidence for app demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEMO_DATA_NOTICE = "CURATED_DEMO_EVIDENCE_RESERVED_OR_FICTIONAL"
DEMO_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "demo"


@dataclass(frozen=True)
class QuizQuestion:
    question: str
    options: tuple[str, ...]
    answer: str
    explanation: str


def _load_demo_csv(filename: str) -> pd.DataFrame | None:
    path = DEMO_DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def build_demo_emails(seed: int = 22057764) -> pd.DataFrame:
    del seed
    loaded = _load_demo_csv("demo_emails.csv")
    if loaded is not None:
        return loaded
    return pd.DataFrame(
        [
            {
                "sample_id": "DEMO-EMAIL-SCAM-001",
                "text": "From: University IT Helpdesk <notice@example.edu> Subject: Action required: mailbox suspension Your student mailbox will be suspended in two hours. Confirm your password and OTP at https://student-mail-verify.example.invalid to keep access.",
                "label": "Suspicious",
                "expected_native_prediction": "Suspicious",
                "expected_action_status": "Immediate Action Required",
                "expected_concern_score": 88,
                "demo_file_name": "demo_email_urgent_mailbox.txt",
                "seeded_report_row": "yes",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-EMAIL-LEGIT-001",
                "text": "From: Library Services <library@example.edu> Subject: Requested book is ready The book you requested is ready for collection at the circulation desk. Please bring your student card.",
                "label": "Legitimate",
                "expected_native_prediction": "Legitimate",
                "expected_action_status": "No Immediate Action",
                "expected_concern_score": 10,
                "demo_file_name": "demo_email_library_notice.txt",
                "seeded_report_row": "yes",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-EMAIL-SCAM-002",
                "text": "Subject: Scholarship selected notice Congratulations, you were selected for a student grant. Pay the RM150 processing fee today by instant transfer or your award will be cancelled.",
                "label": "Suspicious",
                "expected_native_prediction": "Suspicious",
                "expected_action_status": "Immediate Action Required",
                "expected_concern_score": 84,
                "demo_file_name": "demo_email_scholarship_fee.txt",
                "seeded_report_row": "yes",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-EMAIL-LEGIT-002",
                "text": "Subject: Workshop schedule Career services has shared the public workshop schedule for next week. Register through the normal student portal if you want to attend.",
                "label": "Legitimate",
                "expected_native_prediction": "Legitimate",
                "expected_action_status": "No Immediate Action",
                "expected_concern_score": 12,
                "demo_file_name": "demo_email_workshop_schedule.txt",
                "seeded_report_row": "no",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-EMAIL-SCAM-003",
                "text": "Subject: Confidential professor request Do not discuss this with anyone. Buy RM300 gift card codes and reply with the numbers within one hour.",
                "label": "Suspicious",
                "expected_native_prediction": "Suspicious",
                "expected_action_status": "Immediate Action Required",
                "expected_concern_score": 90,
                "demo_file_name": "demo_email_gift_card_request.txt",
                "seeded_report_row": "no",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-EMAIL-LEGIT-003",
                "text": "Subject: Assignment feedback uploaded Your assignment feedback has been uploaded to the learning management system. Review it when you are free.",
                "label": "Legitimate",
                "expected_native_prediction": "Legitimate",
                "expected_action_status": "No Immediate Action",
                "expected_concern_score": 9,
                "demo_file_name": "demo_email_assignment_feedback.txt",
                "seeded_report_row": "no",
                "source": DEMO_DATA_NOTICE,
            },
        ]
    )


def build_demo_transcripts(seed: int = 22057764) -> pd.DataFrame:
    del seed
    loaded = _load_demo_csv("demo_transcripts.csv")
    if loaded is not None:
        return loaded
    return pd.DataFrame(
        [
            {
                "sample_id": "DEMO-TRANSCRIPT-SCAM-001",
                "transcript": "Caller: I am from the bank fraud team. Your account is frozen. Do not end this call. Read me the TAC and OTP now so I can cancel the suspicious transfer.",
                "label": "Scam",
                "expected_native_prediction": "Scam",
                "expected_action_status": "Immediate Action Required",
                "expected_concern_score": 92,
                "demo_file_name": "demo_transcript_bank_otp.txt",
                "seeded_report_row": "yes",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-TRANSCRIPT-LEGIT-001",
                "transcript": "Advisor: Your course registration form is due Friday. Please submit it through the official student portal. Email me if you have questions about the module list.",
                "label": "Non-scam",
                "expected_native_prediction": "Non-scam",
                "expected_action_status": "No Immediate Action",
                "expected_concern_score": 14,
                "demo_file_name": "demo_transcript_advisor_portal.txt",
                "seeded_report_row": "yes",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-TRANSCRIPT-SCAM-002",
                "transcript": "Caller: Your parcel is connected to a police case. Keep this confidential. Transfer RM500 to the secure account today so we can clear your name.",
                "label": "Scam",
                "expected_native_prediction": "Scam",
                "expected_action_status": "Immediate Action Required",
                "expected_concern_score": 90,
                "demo_file_name": "demo_transcript_police_case.txt",
                "seeded_report_row": "yes",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-TRANSCRIPT-LEGIT-002",
                "transcript": "Lecturer: Tomorrow's class will move online. The official meeting link is in the LMS announcement. There is no fee and no password request.",
                "label": "Non-scam",
                "expected_native_prediction": "Non-scam",
                "expected_action_status": "No Immediate Action",
                "expected_concern_score": 11,
                "demo_file_name": "demo_transcript_online_class.txt",
                "seeded_report_row": "no",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-TRANSCRIPT-SCAM-003",
                "transcript": "Caller: The scholarship office selected you, but you must pay a processing fee within 24 hours. Do not contact the university office because this is a private offer.",
                "label": "Scam",
                "expected_native_prediction": "Scam",
                "expected_action_status": "Immediate Action Required",
                "expected_concern_score": 87,
                "demo_file_name": "demo_transcript_scholarship_fee.txt",
                "seeded_report_row": "no",
                "source": DEMO_DATA_NOTICE,
            },
            {
                "sample_id": "DEMO-TRANSCRIPT-LEGIT-003",
                "transcript": "Team member: Let us review the project timeline, divide the documentation tasks, and upload the final slides before Friday.",
                "label": "Non-scam",
                "expected_native_prediction": "Non-scam",
                "expected_action_status": "No Immediate Action",
                "expected_concern_score": 8,
                "demo_file_name": "demo_transcript_project_meeting.txt",
                "seeded_report_row": "no",
                "source": DEMO_DATA_NOTICE,
            },
        ]
    )


def build_demo_phone_reputation(seed: int = 22057764) -> pd.DataFrame:
    del seed
    loaded = _load_demo_csv("demo_phone_numbers.csv")
    if loaded is not None:
        return loaded
    rows = [
        ("DEMO-PHONE-SCAM-001", "+447700900101", "scam", 31, 86, "bank impersonation", "High concern", "Immediate Action Required", "yes"),
        ("DEMO-PHONE-SCAM-002", "+447700900103", "scam", 27, 78, "parcel scam", "High concern", "Immediate Action Required", "no"),
        ("DEMO-PHONE-SCAM-003", "+447700900105", "scam", 24, 76, "authority impersonation", "High concern", "Immediate Action Required", "no"),
        ("DEMO-PHONE-SCAM-004", "+447700900107", "scam", 19, 72, "remote access pressure", "High concern", "Immediate Action Required", "no"),
        ("DEMO-PHONE-SCAM-005", "+447700900109", "scam", 16, 70, "investment pitch", "High concern", "Immediate Action Required", "no"),
        ("DEMO-PHONE-LEGIT-001", "+447700900102", "innocent", 0, 8, "campus office", "Lower concern", "No Immediate Action", "yes"),
        ("DEMO-PHONE-LEGIT-002", "+447700900104", "innocent", 1, 12, "library desk", "Lower concern", "No Immediate Action", "no"),
        ("DEMO-PHONE-LEGIT-003", "+447700900106", "innocent", 0, 6, "course advisor", "Lower concern", "No Immediate Action", "no"),
        ("DEMO-PHONE-LEGIT-004", "+447700900108", "innocent", 1, 14, "delivery reminder", "Lower concern", "No Immediate Action", "no"),
        ("DEMO-PHONE-LEGIT-005", "+447700900110", "innocent", 0, 7, "student services", "Lower concern", "No Immediate Action", "no"),
    ]
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "phone_number": phone_number,
                "demo_group": demo_group,
                "reports": reports,
                "risk_score": risk_score,
                "tag": tag,
                "expected_native_prediction": prediction,
                "expected_action_status": action_status,
                "seeded_report_row": seeded,
                "source": DEMO_DATA_NOTICE,
                "notes": "Reserved drama/demo number",
            }
            for (
                sample_id,
                phone_number,
                demo_group,
                reports,
                risk_score,
                tag,
                prediction,
                action_status,
                seeded,
            ) in rows
        ]
    )


def build_model_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model / Method": "TF-IDF",
                "Used for": "Email and transcript text features",
                "Strength": "Fast, explainable word and phrase representation",
                "Limitation": "Does not understand deep context like transformer models",
            },
            {
                "Model / Method": "Naive Bayes",
                "Used for": "Email phishing and transcript scam classification",
                "Strength": "Very fast baseline for text classification",
                "Limitation": "Assumes features are conditionally independent",
            },
            {
                "Model / Method": "Decision Tree",
                "Used for": "Explainable email comparison model",
                "Strength": "Readable decision logic for student explanation",
                "Limitation": "Can overfit small datasets",
            },
            {
                "Model / Method": "MFCC",
                "Used for": "Audio feature extraction",
                "Strength": "Compact speech frequency representation",
                "Limitation": "Does not localize exact suspicious timestamps",
            },
            {
                "Model / Method": "SVM",
                "Used for": "AI-generated speech detection",
                "Strength": "Strong small-dataset classifier with clear boundary",
                "Limitation": "Needs representative real/fake voice samples",
            },
        ]
    )


def build_quiz_questions() -> list[QuizQuestion]:
    return [
        QuizQuestion(
            question="A caller says your bank account is suspended and asks for your OTP immediately.",
            options=("Likely scam", "Likely normal"),
            answer="Likely scam",
            explanation="OTP requests plus urgency are strong scam indicators.",
        ),
        QuizQuestion(
            question="Your lecturer emails that class is moved online and points you to the official LMS announcement.",
            options=("Likely scam", "Likely normal"),
            answer="Likely normal",
            explanation="The message uses an official channel and does not ask for credentials or payment.",
        ),
        QuizQuestion(
            question="A scholarship message says you were selected but must pay a processing fee today.",
            options=("Likely scam", "Likely normal"),
            answer="Likely scam",
            explanation="Unexpected reward plus payment pressure is a common student-targeted scam pattern.",
        ),
        QuizQuestion(
            question="A teammate asks to reschedule the project meeting and does not request money or login details.",
            options=("Likely scam", "Likely normal"),
            answer="Likely normal",
            explanation="Routine scheduling without pressure or sensitive data requests is lower risk.",
        ),
        QuizQuestion(
            question="A caller says to keep the conversation confidential and transfer money to avoid a police case.",
            options=("Likely scam", "Likely normal"),
            answer="Likely scam",
            explanation="Secrecy, threats, and money transfer requests are high-risk signals.",
        ),
    ]


def build_demo_bundle() -> dict[str, object]:
    emails = build_demo_emails()
    transcripts = build_demo_transcripts()
    phones = build_demo_phone_reputation()
    models = build_model_comparison()
    return {
        "notice": DEMO_DATA_NOTICE,
        "emails": emails,
        "transcripts": transcripts,
        "phones": phones,
        "models": models,
        "quiz": build_quiz_questions(),
    }
