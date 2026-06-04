"""Scenario content for the Scam Simulation Lab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioOption:
    label: str
    feedback: str
    safe: bool


@dataclass(frozen=True)
class ScenarioQuiz:
    question: str
    options: tuple[str, ...]
    answer: str
    explanation: str


@dataclass(frozen=True)
class ScamScenario:
    scenario_id: str
    title: str
    channel: str
    difficulty: str
    content: str
    attacker_motive: str
    indicators: tuple[str, ...]
    defense_steps: tuple[str, ...]
    options: tuple[ScenarioOption, ...]
    quiz: tuple[ScenarioQuiz, ...]
    time_limit_seconds: int = 45


SCENARIOS: tuple[ScamScenario, ...] = (
    ScamScenario(
        scenario_id="tuition-phishing",
        title="Tuition Payment Panic",
        channel="Email",
        difficulty="Beginner",
        content=(
            "Subject: Final warning - tuition payment failed\n\n"
            "Dear student, your tuition payment could not be verified. Your university account "
            "will be suspended within 24 hours unless you verify your account and update bank "
            "details here: https://student-finance-verify.invalid. This is confidential."
        ),
        attacker_motive=(
            "The attacker is trying to create urgency and impersonate a university finance office "
            "so the student shares credentials or banking details before thinking carefully."
        ),
        indicators=(
            "Urgent deadline",
            "Threat of account suspension",
            "External verification link",
            "Bank detail request",
            "Confidentiality pressure",
        ),
        defense_steps=(
            "Do not click the link.",
            "Open the official student portal manually.",
            "Contact university finance through a verified channel.",
            "Report the message to IT/security support.",
        ),
        options=(
            ScenarioOption("Click the link and verify quickly", "That follows the attacker's pressure tactic.", False),
            ScenarioOption("Reply with your student ID and bank details", "That gives sensitive data to the attacker.", False),
            ScenarioOption("Use the official portal or call finance directly", "Good. You moved to a verified channel.", True),
        ),
        quiz=(
            ScenarioQuiz(
                "Which indicator is strongest here?",
                ("The sender says dear student", "The message demands verification within 24 hours", "The email has a subject line"),
                "The message demands verification within 24 hours",
                "Urgency is used to bypass careful checking.",
            ),
            ScenarioQuiz(
                "What is the safest next step?",
                ("Use the official portal manually", "Click the link", "Forward your password"),
                "Use the official portal manually",
                "Manual navigation avoids attacker-controlled links.",
            ),
        ),
    ),
    ScamScenario(
        scenario_id="otp-call",
        title="Bank OTP Caller",
        channel="Call transcript",
        difficulty="Intermediate",
        content=(
            "Caller: I am from the bank fraud team. Your account is currently locked because "
            "of suspicious activity. I need your one-time password now to stop the transfer. "
            "Do not tell anyone because this is an active investigation."
        ),
        attacker_motive=(
            "The attacker wants an OTP to approve a transaction or account takeover. They combine "
            "authority, fear, secrecy, and time pressure."
        ),
        indicators=(
            "OTP request",
            "Bank impersonation",
            "Secrecy instruction",
            "Fear of losing money",
            "Immediate action demand",
        ),
        defense_steps=(
            "Never share OTP codes.",
            "Hang up calmly.",
            "Call the bank using the number on the official card or website.",
            "Freeze or review the account through official channels.",
        ),
        options=(
            ScenarioOption("Read the OTP because the caller sounds official", "Banks should not ask for OTP codes.", False),
            ScenarioOption("Ask for the caller's name and still continue", "Asking is not enough if the channel is unverified.", False),
            ScenarioOption("Hang up and call the official bank number", "Correct. You broke the attacker's control of the conversation.", True),
        ),
        quiz=(
            ScenarioQuiz(
                "Why is secrecy suspicious?",
                ("It isolates the victim", "It saves time", "It proves the caller is official"),
                "It isolates the victim",
                "Scammers use secrecy to stop victims from asking trusted people for help.",
            ),
            ScenarioQuiz(
                "Should OTP ever be shared with a caller?",
                ("Yes, if urgent", "No", "Only if they know your name"),
                "No",
                "OTP codes are authentication secrets.",
            ),
        ),
    ),
    ScamScenario(
        scenario_id="internship-offer",
        title="Fake Internship Offer",
        channel="Chat message",
        difficulty="Advanced",
        content=(
            "Hi, we selected you for a remote cybersecurity internship. Kindly fill this form "
            "with your IC, bank account, login email, and pay RM80 onboarding fee today. "
            "Slots are limited, so confirm immediately."
        ),
        attacker_motive=(
            "The attacker is using career pressure and reward framing to collect identity data, "
            "bank details, and a small irreversible payment."
        ),
        indicators=(
            "Unexpected selection",
            "Payment before employment",
            "Identity and bank data request",
            "Limited slot pressure",
            "Vague recruiter identity",
        ),
        defense_steps=(
            "Verify the company and recruiter independently.",
            "Do not pay onboarding fees.",
            "Share only required data through official HR systems.",
            "Ask career services or a lecturer before proceeding.",
        ),
        options=(
            ScenarioOption("Pay RM80 because the opportunity may disappear", "Scarcity pressure is part of the scam.", False),
            ScenarioOption("Submit IC and bank details but skip payment", "Identity data is still sensitive.", False),
            ScenarioOption("Verify the recruiter through official channels first", "Correct. Verification protects both identity and money.", True),
        ),
        quiz=(
            ScenarioQuiz(
                "What attacker motive is most likely?",
                ("Help students find jobs", "Collect identity data and money", "Schedule a normal interview"),
                "Collect identity data and money",
                "The message asks for sensitive data and payment before verification.",
            ),
            ScenarioQuiz(
                "Which defense fits best?",
                ("Pay first", "Verify independently", "Send all requested details"),
                "Verify independently",
                "Independent verification removes the attacker's control.",
            ),
        ),
    ),
)


def get_scenarios() -> tuple[ScamScenario, ...]:
    return SCENARIOS


def get_scenario(scenario_id: str) -> ScamScenario:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    return SCENARIOS[0]
