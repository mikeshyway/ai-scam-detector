"""Explainability helpers for student-facing scam awareness feedback."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

import numpy as np


@dataclass(frozen=True)
class SuspiciousPattern:
    pattern: str
    category: str
    specific_tactic: str
    is_regex: bool = False


@dataclass(frozen=True)
class ContextPattern:
    pattern: str
    category: str
    intention: str
    is_regex: bool = False


PATTERNS: tuple[SuspiciousPattern, ...] = (
    SuspiciousPattern(r"\burgent(ly)?\b|\bimmediately\b|\bwithin\s+\d+\s+(hours?|days?)\b|\btoday only\b|\bfinal notice\b", "Urgency", "Time-pressure urgency", True),
    SuspiciousPattern(r"\bfinal warning\b|\bsuspended\b|\blocked\b|\blegal action\b|\bpenalty\b|\bblacklist(ed)?\b|\bpolice report\b", "Threat & Intimidation", "Consequence pressure", True),
    SuspiciousPattern(r"\bpassword\b|\botp\b|\bone[-\s]?time password\b|\blog[\s-]?in\b|\bverify\b.{0,40}\baccount\b|\breset\b.{0,40}\bcredentials?\b", "Credential Request", "Account access request", True),
    SuspiciousPattern(r"\bbank transfer\b|\bwire transfer\b|\bgift card\b|\bcrypto(currency)?\b|\bprocessing fee\b|\brefund fee\b|\btuition payment\b", "Financial Request", "Financial transfer request", True),
    SuspiciousPattern(r"\bwinner\b|\bselected\b|\bscholarship\b|\bgrant\b|\bprize\b|\bfree money\b|\bcongratulations\b", "Reward Bait", "Unexpected reward bait", True),
    SuspiciousPattern(r"\b(bank officer|bank department|bank security|bank support|bank verification|banking department|university finance|university admin|professor|dean|government officer|police officer|courier service|hr department|finance department)\b", "Authority Impersonation", "Trusted-entity impersonation", True),
    SuspiciousPattern(r"\binternship offer\b|\brecruiter\b|\bit department\b|\bsupport team\b|\bstudent office\b|\badmin office\b", "Impersonation", "Trusted-person or organisation impersonation", True),
    SuspiciousPattern(r"\bdo not tell\b|\bdo not share\b|\bconfidential\b|\bkeep\s+(this\s+)?private\b|\bsecret\b", "Secrecy", "Isolation pressure", True),
    SuspiciousPattern(r"\bwhats\s?app\b|\btelegram\b|\bcall this number\b|\breply outside\b|\bcontact me directly\b|\bphonetoken\b", "Communication Redirect", "Off-platform communication request", True),
    SuspiciousPattern(r"\bclick here\b|\burltoken\b", "External Link / Domain", "External link prompt", True),
    SuspiciousPattern(r"\bemailtoken\b|\bfrom:\b|\breply-to:\b|\bgmail\.com\b|\byahoo\.com\b|\boutlook\.com\b", "Sender Identity", "Sender identity clue", True),
    SuspiciousPattern(r"\.(exe|zip|scr|js|bat|vbs|docm|xlsm)\b|\benable macros\b|\bpassword[-\s]?protected attachment\b", "Attachment Risk", "Executable or protected attachment", True),
    SuspiciousPattern(r"\bpersonal account\b|\baccount holder\b|\bunusual currency\b|\bwestern union\b|\bmoneygram\b|\bmoneytoken\b", "Account Mismatch", "Unusual payment destination", True),
    SuspiciousPattern(r"(?-i:\b[A-Z]{5,}\b)|[!?]{3,}|,\S|\.\.", "Writing Style", "Unprofessional formatting", True),
    SuspiciousPattern(r"\bkindly\b|\bdear valued customer\b|\bdear customer\b|\bwe value your security\b|\bplease be informed\b", "Template Style", "Template-like or unnatural wording", True),
    SuspiciousPattern(r"\b(emergency|urgent help|please help|help me now|i need your help|desperate|struggling|anything helps|please donate|medical emergency|hospital bills?|funeral expenses?|my family|my child|my mother|my father|i have no money|cannot afford|financial hardship|stranded|lost my wallet|stuck overseas|sympathy|kindness|good heart|bless you|save my life|life saving|critical condition|only you can help|please have mercy)\b", "Emotional Manipulation", "Distress, begging, or sympathy pressure", True),
    SuspiciousPattern(r"\b(ic number|nric|passport|student id|matric number|date of birth|home address|full name|phone number|personal details)\b", "Personal Information Request", "Sensitive identity data request", True),
)


LEGITIMATE_PATTERNS: tuple[ContextPattern, ...] = (
    ContextPattern(r"\b(sunway\.edu\.my|edu\.my|gov\.my|official portal|intranet)\b", "Internal Domain", "Show an official organisational web domain.", True),
    ContextPattern(r"\b(meeting agenda|agenda|agenda item|calendar invite|scheduled meeting|meeting minutes)\b", "Meeting Agenda", "Suggest routine meeting coordination.", True),
    ContextPattern(r"\b(microsoft teams|sharepoint|google meet|zoom meeting|official channel|student portal)\b", "Official Work Platform", "Reference recognised workplace collaboration tools.", True),
    ContextPattern(r"\b(regards|kind regards|sincerely|best regards)\b", "Professional Signature", "Show a normal professional closing.", True),
    ContextPattern(r"\b(finance team|operations team|it support|hr department|project update|weekly update|team meeting|department workflow|follow up|as discussed)\b", "Department Workflow", "Suggest normal internal department activity.", True),
    ContextPattern(r"\b(purchase order|invoice reference|reconciliation)\b", "Department Workflow", "Suggest normal internal department activity.", True),
    ContextPattern(r"\b(do not enter your password|official company systems)\b", "Safe Credential Warning", "Warn against unsafe credential sharing.", True),
)


TYPE_INTENTIONS = {
    "Urgency": "Push action before careful verification.",
    "Threat & Intimidation": "Pressure through fear of consequences.",
    "Credential Request": "Collect passwords, OTPs, or login details.",
    "Financial Request": "Prompt fast or unusual money transfer.",
    "Reward Bait": "Lure attention with fake benefits.",
    "Authority Impersonation": "Borrow trust from a recognised authority.",
    "Impersonation": "Pretend to be a trusted contact.",
    "Secrecy": "Prevent independent verification by others.",
    "Communication Redirect": "Move conversation away from safer channels.",
    "External Link / Domain": "Send user to an external website.",
    "Shortened URL": "Hide the true destination behind a redirect.",
    "Suspicious Domain Ending": "Use uncommon domains for phishing pages.",
    "IP Address Link": "Bypass recognizable domain names with numbers.",
    "Credential Landing Page": "Lead users toward account verification pages.",
    "Brand Imitation Domain": "Mimic a trusted brand or service.",
    "Sender Domain Mismatch": "Hide identity behind an inconsistent sender.",
    "Free Email Brand Sender": "Impersonate a brand through free email.",
    "Sender Identity": "Confuse the sender's real identity.",
    "Writing Style": "Suggest careless or unnatural message construction.",
    "Template Style": "Reuse generic wording from phishing templates.",
    "Attachment Risk": "Encourage opening a risky attachment.",
    "Account Mismatch": "Route payment to an unauthorized account.",
    "Emotional Manipulation": "Exploit emotions to influence decisions.",
    "Personal Information Request": "Collect sensitive personal identity information.",
    "Internal Domain": "Show an official organisational web domain.",
    "Professional Signature": "Show a normal professional closing.",
    "Meeting Agenda": "Suggest routine meeting coordination.",
    "Official Work Platform": "Reference recognised workplace collaboration tools.",
    "Department Workflow": "Suggest normal internal department activity.",
    "Safe Credential Warning": "Warn against unsafe credential sharing.",
    "Model Term": "Influence the AI model prediction.",
}


TRUSTED_BRANDS = {
    "google": ["google.com", "gmail.com"],
    "microsoft": ["microsoft.com", "office.com", "outlook.com"],
    "paypal": ["paypal.com"],
    "amazon": ["amazon.com"],
    "maybank": ["maybank2u.com.my", "maybank.com"],
    "cimb": ["cimb.com.my"],
    "sunway": ["sunway.edu.my"],
}

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
}

SUSPICIOUS_TLDS = {
    "xyz",
    "top",
    "click",
    "site",
    "live",
    "buzz",
    "tk",
    "ml",
    "ga",
    "cf",
}

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "ow.ly",
}

URL_PATTERN = re.compile(
    r"(?:https?|hxxps?)://[^\s<>\"]+|"
    r"www\.[^\s<>\"]+|"
    r"(?<!@)\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+/[^\s<>\"]*|"
    r"(?<!@)\b[a-z0-9][a-z0-9-]*\.(?:xyz|top|click|site|live|buzz|tk|ml|ga|cf)\b(?:/[^\s<>\"]*)?",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)


def _pattern_reason(pattern: SuspiciousPattern) -> str:
    return TYPE_INTENTIONS.get(pattern.category, pattern.specific_tactic)


def type_intention(category: object) -> str:
    return TYPE_INTENTIONS.get(str(category), "Supports manual review of this evidence.")


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _domain_from_url(url: str) -> str:
    candidate = _normalise_url_for_analysis(url)
    candidate = candidate.strip().rstrip(".,;:!?)\"]}")
    if candidate.startswith("www."):
        candidate = "https://" + candidate
    elif not re.match(r"^[a-z][a-z0-9+.-]*://", candidate, re.IGNORECASE):
        candidate = "https://" + candidate

    parsed = urlparse(candidate)
    domain = parsed.netloc.lower()
    if "@" in domain:
        domain = domain.rsplit("@", 1)[-1]
    if ":" in domain:
        domain = domain.split(":", 1)[0]
    return domain[4:] if domain.startswith("www.") else domain


def _normalise_url_for_analysis(url: str) -> str:
    return (
        url.strip()
        .replace("hxxps://", "https://")
        .replace("hxxp://", "http://")
        .replace("[.]", ".")
        .replace("(.)", ".")
    )


def _extract_urls(text: str) -> list[str]:
    return [match.group(0) for match in URL_PATTERN.finditer(text)]


def _extract_emails(text: str) -> list[str]:
    return [match.group(0) for match in EMAIL_PATTERN.finditer(text)]


def _is_ip_domain(domain: str) -> bool:
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))


def _matches_official_domain(domain: str, official_domains: list[str]) -> bool:
    return any(domain == official or domain.endswith("." + official) for official in official_domains)


def _domain_tokens(domain: str) -> set[str]:
    label = domain.split(".", 1)[0]
    return {part for part in re.split(r"[^a-z0-9]+", label.lower()) if part}


def _looks_like_brand(domain: str, official_domains: list[str], brand: str) -> bool:
    if _matches_official_domain(domain, official_domains):
        return False

    tokens = _domain_tokens(domain)
    official_roots = {official.split(".", 1)[0].lower() for official in official_domains}
    candidates = tokens.union({domain.split(".", 1)[0].lower()})
    target_terms = official_roots.union({brand.lower()})

    if brand.lower() in domain.lower():
        return True

    return any(
        0.78 <= _similarity(candidate, target) < 1.0
        for candidate in candidates
        for target in target_terms
    )


def _has_credential_landing_words(url: str, domain: str) -> bool:
    normalized = _normalise_url_for_analysis(url).lower()
    domain_words = _domain_tokens(domain)
    credential_words = {"login", "verify", "verification", "account", "secure", "reset", "auth", "portal"}
    if domain_words.intersection(credential_words):
        return True
    return any(word in normalized for word in credential_words)


def _link_category(url: str, domain: str) -> str:
    tld = domain.split(".")[-1] if "." in domain else ""

    if domain in SHORTENERS:
        return "Shortened URL"
    if _is_ip_domain(domain):
        return "IP Address Link"

    for brand, official_domains in TRUSTED_BRANDS.items():
        if _looks_like_brand(domain, official_domains, brand):
            return "Brand Imitation Domain"

    if _has_credential_landing_words(url, domain):
        return "Credential Landing Page"
    if tld in SUSPICIOUS_TLDS:
        return "Suspicious Domain Ending"
    return "External Link / Domain"


def analyse_domain_indicators(text: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []

    for match in URL_PATTERN.finditer(text):
        url = match.group(0)
        domain = _domain_from_url(url)
        if not domain:
            continue

        category = _link_category(url, domain)
        findings.append(
            {
                "phrase": domain,
                "category": category,
                "specific_tactic": category,
                "label": category,
                "intention": type_intention(category),
                "reason": category,
                "start": match.start(),
                "end": match.end(),
            }
        )

    for match in EMAIL_PATTERN.finditer(text):
        email_addr = match.group(0)
        domain = email_addr.split("@")[-1].lower()
        label = None

        brand_in_address = any(brand in email_addr.lower() for brand in TRUSTED_BRANDS)

        if domain in FREE_EMAIL_DOMAINS and brand_in_address:
            label = "Free Email Brand Sender"
        elif domain.split(".")[-1] in SUSPICIOUS_TLDS:
            label = "Suspicious Domain Ending"

        for brand, official_domains in TRUSTED_BRANDS.items():
            if brand in email_addr.lower() and not _matches_official_domain(domain, official_domains):
                label = "Free Email Brand Sender" if domain in FREE_EMAIL_DOMAINS else "Sender Domain Mismatch"

        if label:
            findings.append(
                {
                    "phrase": email_addr,
                    "category": label,
                    "specific_tactic": label,
                    "label": label,
                    "intention": type_intention(label),
                    "reason": label,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    return findings


def find_suspicious_phrases(text: str) -> list[dict[str, object]]:
    if not text:
        return []

    findings: list[dict[str, object]] = []
    for pattern in PATTERNS:
        regex_text = pattern.pattern if pattern.is_regex else rf"(?<!\w){re.escape(pattern.pattern)}(?!\w)"
        regex = re.compile(regex_text, re.IGNORECASE)

        for match in regex.finditer(text):
            findings.append(
                {
                    "phrase": text[match.start() : match.end()],
                    "category": pattern.category,
                    "specific_tactic": pattern.specific_tactic,
                    "intention": type_intention(pattern.category),
                    "reason": _pattern_reason(pattern),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    findings.extend(analyse_domain_indicators(text))
    findings.sort(key=lambda item: (int(item["start"]), -int(item["end"])))
    semantic_spans = [
        (int(item["start"]), int(item["end"]))
        for item in findings
        if str(item.get("category")) != "Writing Style"
    ]
    filtered: list[dict[str, object]] = []
    for item in findings:
        indicator = str(item.get("phrase", "")).strip().casefold()
        category = str(item.get("category", "")).strip()
        if not indicator:
            continue

        if category == "Writing Style":
            start = int(item["start"])
            end = int(item["end"])
            if any(start >= span_start and end <= span_end for span_start, span_end in semantic_spans):
                continue

        filtered.append(item)
    return filtered


def find_legitimate_indicators(text: str) -> list[dict[str, object]]:
    if not text:
        return []

    indicators: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for pattern in LEGITIMATE_PATTERNS:
        regex_text = pattern.pattern if pattern.is_regex else rf"(?<!\w){re.escape(pattern.pattern)}(?!\w)"
        regex = re.compile(regex_text, re.IGNORECASE)

        for match in regex.finditer(text):
            indicator = text[match.start() : match.end()].strip()
            key = (indicator.casefold(), pattern.category)
            if not indicator or key in seen:
                continue

            indicators.append(
                {
                    "indicator": indicator,
                    "phrase": indicator,
                    "category": pattern.category,
                    "Type": pattern.category,
                    "source": "Context indicator",
                    "intention": pattern.intention,
                    "reason": pattern.intention,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            seen.add(key)

    return indicators


def highlighted_html(text: str, findings: list[dict[str, object]]) -> str:
    if not text:
        return ""
    if not findings:
        return html.escape(text).replace("\n", "<br>")

    spans = sorted(
        [(int(item["start"]), int(item["end"]), str(item["category"])) for item in findings],
        key=lambda item: item[0],
    )
    merged: list[tuple[int, int, str]] = []
    cursor_end = -1
    for start, end, category in spans:
        if start < cursor_end:
            continue
        merged.append((start, end, category))
        cursor_end = end

    pieces: list[str] = []
    cursor = 0
    for start, end, category in merged:
        pieces.append(html.escape(text[cursor:start]))
        title = html.escape(category)
        phrase = html.escape(text[start:end])
        pieces.append(f'<mark title="{title}">{phrase}</mark>')
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces).replace("\n", "<br>")


def educational_summary(
    label: str,
    confidence: float | None,
    findings: list[dict[str, object]],
) -> str:
    count = len(findings)
    if confidence is None:
        if count:
            return f"Demo mode found {count} suspicious pattern(s). Train the ML model for a real prediction."
        return "No strong rule-based scam signals were found. Train the ML model for a real prediction."

    percent = round(confidence * 100, 1)
    if "Suspicious" in label or "AI-generated" in label:
        return (
            f"The model classified this as suspicious with {percent}% confidence. "
            f"It also found {count} explainable warning pattern(s)."
        )
    return (
        f"The model classified this as lower risk with {percent}% confidence. "
        f"Still verify sender identity and links before acting."
    )


def top_model_terms(
    text: str,
    vectorizer: Any,
    model: Any,
    *,
    top_n: int = 10,
) -> list[dict[str, object]]:
    try:
        X = vectorizer.transform([text])
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        active = X.toarray()[0]
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            return []

        model_name = type(model).__name__.casefold()
        model_module = type(model).__module__.casefold()
        is_xgboost = "xgb" in model_name or "xgboost" in model_module
        weights = None
        directional = True
        method = "model_weight"
        if hasattr(model, "feature_log_prob_") and model.feature_log_prob_.shape[0] >= 2:
            weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
            method = "naive_bayes_log_probability_delta"
        elif hasattr(model, "coef_"):
            weights = np.ravel(model.coef_)
            method = "linear_coefficient"
        elif hasattr(model, "feature_importances_"):
            weights = model.feature_importances_
            directional = not is_xgboost
            method = "xgboost_feature_importance" if is_xgboost else "tree_feature_importance"

        if weights is None:
            return []

        scores = active[active_indices] * weights[active_indices]
        order = np.argsort(np.abs(scores))[::-1][:top_n]
        return [
            {
                "term": str(feature_names[active_indices[index]]),
                "score": float(scores[index]),
                "directional": directional,
                "method": method,
                "model_family": "xgboost" if is_xgboost else "standard",
            }
            for index in order
        ]
    except Exception:
        return []
