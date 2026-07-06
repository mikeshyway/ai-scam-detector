"""Text loading and preprocessing utilities."""

from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
MONEY_RE = re.compile(r"(?:(?:rm|usd|myr|\$)\s*)?\b\d+(?:,\d{3})*(?:\.\d{2})?\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
NON_WORD_RE = re.compile(r"[^a-z_\s]")
WHITESPACE_RE = re.compile(r"\s+")

NEGATIVE_LABELS = {
    "0",
    "ham",
    "legit",
    "legitimate",
    "safe",
    "normal",
    "non-scam",
    "non scam",
    "nonscam",
    "not scam",
    "real",
    "human",
    "bonafide",
    "genuine",
    "false",
    "no",
}

POSITIVE_LABELS = {
    "1",
    "spam",
    "phishing",
    "phish",
    "scam",
    "fraud",
    "fake",
    "spoof",
    "synthetic",
    "ai",
    "malicious",
    "suspicious",
    "true",
    "yes",
}


def _nltk_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return set(ENGLISH_STOP_WORDS)


STOPWORDS = _nltk_stopwords()


def _lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except Exception:
        return list(tokens)


def clean_text(text: object, *, remove_stopwords: bool = True, lemmatize: bool = False) -> str:
    if text is None:
        return ""

    cleaned = str(text).lower()
    cleaned = URL_RE.sub(" urltoken ", cleaned)
    cleaned = EMAIL_RE.sub(" emailtoken ", cleaned)
    cleaned = PHONE_RE.sub(" phonetoken ", cleaned)
    cleaned = MONEY_RE.sub(" moneytoken ", cleaned)
    cleaned = NON_WORD_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()

    tokens = cleaned.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    if lemmatize:
        tokens = _lemmatize_tokens(tokens)

    return " ".join(tokens)


def build_tfidf_vectorizer(max_features: int = 8000) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean_text,
        token_pattern=r"(?u)\b[a-z_][a-z_]+\b",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
    )


def extract_email_body(path: Path) -> str:
    raw = path.read_bytes()
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        return raw.decode("utf-8", errors="ignore")

    parts: list[str] = []
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition()
            if content_type == "text/plain" and disposition != "attachment":
                try:
                    parts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    parts.append(payload.decode("utf-8", errors="ignore"))
    else:
        try:
            parts.append(message.get_content())
        except Exception:
            payload = message.get_payload(decode=True) or raw
            parts.append(payload.decode("utf-8", errors="ignore"))

    text = "\n".join(part for part in parts if part)
    return text if text.strip() else raw.decode("utf-8", errors="ignore")


def load_spamassassin_dataset(raw_dir: str | Path) -> pd.DataFrame:
    root = Path(raw_dir)
    rows: list[dict[str, object]] = []

    for folder_name, label in (("ham", 0), ("spam", 1)):
        folder = root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                rows.append({"text": extract_email_body(path), "label": label, "source": str(path)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No SpamAssassin emails found under {root}. Expected spam/ and ham/ folders."
        )
    return df


def label_to_binary(value: object) -> int:
    label = str(value).strip().lower().replace("_", " ").replace("/", " ")
    label = WHITESPACE_RE.sub(" ", label)

    if label in NEGATIVE_LABELS or any(marker in label for marker in ("non scam", "not scam")):
        return 0
    if label in POSITIVE_LABELS:
        return 1
    if "bonafide" in label or "legit" in label or label == "ham":
        return 0
    if "scam" in label or "spam" in label or "phish" in label or "fraud" in label:
        return 1

    try:
        return int(float(label) > 0)
    except ValueError as exc:
        raise ValueError(f"Cannot convert label value {value!r} to binary 0/1.") from exc


def infer_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "conversation",
        "transcript",
        "call_transcript",
        "call transcript",
        "text",
        "message",
        "dialogue",
        "content",
        "body",
    ]
    lowered = {column.lower().strip(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    object_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) and df[column].astype(str).str.len().mean() > 20
    ]
    if object_columns:
        return object_columns[0]

    raise ValueError("Could not infer a text column. Pass --text-column explicitly.")


def infer_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "class", "target", "is_scam", "scam", "fraud", "category", "type", "outcome"]
    lowered = {column.lower().strip(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in df.columns:
        values = set(df[column].dropna().astype(str).str.lower().str.strip().head(20))
        if values and all(value in NEGATIVE_LABELS | POSITIVE_LABELS for value in values):
            return column

    raise ValueError("Could not infer a label column. Pass --label-column explicitly.")


def load_labeled_text_csv(
    csv_path: str | Path,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")

    text_col = text_column or infer_text_column(df)
    label_col = label_column or infer_label_column(df)

    loaded = pd.DataFrame(
        {
            "text": df[text_col].fillna("").astype(str),
            "label": df[label_col].map(label_to_binary),
        }
    )
    loaded = loaded[loaded["text"].str.strip().astype(bool)]
    if loaded["label"].nunique() < 2:
        raise ValueError("Training requires both scam and non-scam labels.")
    return loaded
