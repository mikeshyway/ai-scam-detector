"""Temporary synthetic data for app demonstrations before official datasets are inserted."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEMO_DATA_NOTICE = "TEMPORARY_SYNTHETIC_DEMO_DATA_REMOVE_AFTER_OFFICIAL_DATASET_INSERTION"


@dataclass(frozen=True)
class QuizQuestion:
    question: str
    options: tuple[str, ...]
    answer: str
    explanation: str


def build_demo_emails(seed: int = 22057764) -> pd.DataFrame:
    rng = random.Random(seed)
    scam_templates = [
        "Urgent: your university email account will be suspended within 24 hours. Verify your account at {url}.",
        "Scholarship selected notice. Kindly pay RM{amount} processing fee today to release your award.",
        "Final warning from finance office: tuition payment failed. Send your bank account details immediately.",
        "Internship offer confirmed. Click here and upload your password and OTP to activate your placement.",
        "Professor request: do not tell anyone, buy gift card codes worth RM{amount} and reply now.",
    ]
    safe_templates = [
        "Reminder: your lab session starts at 10 AM. Please review the slides before class.",
        "The library book you requested is ready for collection at the front desk.",
        "Your assignment feedback has been uploaded to the learning management system.",
        "Student council meeting notes are attached for your review.",
        "Career services has shared a public workshop schedule for next week.",
    ]

    rows = []
    for i in range(18):
        is_scam = i % 2 == 0
        template = rng.choice(scam_templates if is_scam else safe_templates)
        rows.append(
            {
                "sample_id": f"DEMO-EMAIL-{i + 1:03d}",
                "text": template.format(url="https://example-login.invalid", amount=rng.choice([80, 150, 300])),
                "label": "Suspicious" if is_scam else "Legitimate",
                "source": DEMO_DATA_NOTICE,
            }
        )
    return pd.DataFrame(rows)


def build_demo_transcripts(seed: int = 22057764) -> pd.DataFrame:
    rng = random.Random(seed + 7)
    scam_lines = [
        "Caller: I am from the bank fraud team. Your account is suspended. Share the OTP immediately.",
        "Caller: This is confidential. Do not tell anyone. Transfer RM500 to secure your student visa.",
        "Caller: Your parcel is linked to a police case. Verify your IC and bank account now.",
        "Caller: The scholarship office selected you, but you must pay a processing fee within 24 hours.",
    ]
    safe_lines = [
        "Lecturer: Tomorrow's class will move online. The official meeting link is in the LMS announcement.",
        "Advisor: Please submit your course registration form before Friday through the student portal.",
        "Team member: Let us review the project timeline and divide the documentation tasks.",
        "Parent: Call me after class when you are free. No rush.",
    ]
    rows = []
    for i in range(16):
        is_scam = i % 2 == 0
        text = rng.choice(scam_lines if is_scam else safe_lines)
        rows.append(
            {
                "sample_id": f"DEMO-CALL-{i + 1:03d}",
                "transcript": text,
                "label": "Scam" if is_scam else "Non-scam",
                "source": DEMO_DATA_NOTICE,
            }
        )
    return pd.DataFrame(rows)


def build_demo_phone_reputation(seed: int = 22057764) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    prefixes = ["+60 11", "+60 12", "+60 13", "+60 16", "+65 8", "+44 20"]
    tags = ["bank impersonation", "parcel scam", "unknown recruiter", "campus office", "family", "delivery"]
    rows = []
    for i in range(20):
        reports = int(rng.integers(0, 44))
        risk_score = min(99, reports * 2 + int(rng.integers(0, 25)))
        rows.append(
            {
                "phone_number": f"{prefixes[i % len(prefixes)]} {int(rng.integers(1000, 9999))} {int(rng.integers(1000, 9999))}",
                "reports": reports,
                "risk_score": risk_score,
                "tag": tags[i % len(tags)],
                "source": DEMO_DATA_NOTICE,
            }
        )
    return pd.DataFrame(rows)


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
