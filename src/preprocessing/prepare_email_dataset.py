from pathlib import Path
import pandas as pd
import re
from email import policy
from email.parser import BytesParser

ROOT = Path(__file__).resolve().parents[2]

RAW_EMAIL_DIR = ROOT / "data" / "raw" / "email"
SPAMASSASSIN_DIR = RAW_EMAIL_DIR / "spamassassin_public_corpus"
ENRON_FILE = RAW_EMAIL_DIR / "the_enron_email_dataset" / "emails.csv"
PHISHING_FILE = (
    RAW_EMAIL_DIR
    / "phishing_and_legitimate_emails_dataset_for_ml_2026"
    / "phishing_legit_dataset_KD_10000.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "email"
OUTPUT_FILE = OUTPUT_DIR / "email_dataset.csv"

SPAMASSASSIN_FOLDERS = {
    "easy_ham": 0,
    "hard_ham": 0,
    "spam_2": 1,
}


def extract_email_text(file_path: Path) -> str:
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = msg.get("subject", "")
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body += part.get_content()
                    except Exception:
                        pass
        else:
            try:
                body = msg.get_content()
            except Exception:
                body = ""

        return f"{subject}\n{body}"

    except Exception:
        try:
            return file_path.read_text(errors="ignore")
        except Exception:
            return ""


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"hxxps?://\S+|https?://\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-zA-Z0-9$£€%@.!? ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower().strip(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for col in df.columns:
        col_l = col.lower()
        if any(candidate.lower() in col_l for candidate in candidates):
            return col

    return None


def load_spamassassin(rows: list[dict[str, object]]) -> None:
    print("\nLoading SpamAssassin...")

    for folder_name, label in SPAMASSASSIN_FOLDERS.items():
        folder_path = SPAMASSASSIN_DIR / folder_name

        if not folder_path.exists():
            print(f"[MISSING] {folder_path}")
            continue

        files = [p for p in folder_path.iterdir() if p.is_file()]
        print(f"[FOUND] {folder_name}: {len(files)} files")

        for file_path in files:
            text = clean_text(extract_email_text(file_path))

            if len(text) < 20:
                continue

            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": f"spamassassin_{folder_name}",
                    "file_name": file_path.name,
                }
            )


def load_enron(rows: list[dict[str, object]], max_rows: int = 20000) -> None:
    print("\nLoading Enron...")

    if not ENRON_FILE.exists():
        print(f"[MISSING] {ENRON_FILE}")
        return

    df = pd.read_csv(ENRON_FILE, nrows=max_rows)

    text_col = find_column(df, ["message", "text", "email", "body", "content"])

    if text_col is None:
        print(f"[SKIPPED] Enron text column not found. Columns: {list(df.columns)}")
        return

    for index, value in df[text_col].fillna("").astype(str).items():
        text = clean_text(value)

        if len(text) < 20:
            continue

        rows.append(
            {
                "text": text,
                "label": 0,
                "source": "enron_legitimate",
                "file_name": f"enron_{index}",
            }
        )

    print(f"[FOUND] Enron rows loaded: {min(len(df), max_rows)}")


def load_phishing2026(rows: list[dict[str, object]]) -> None:
    print("\nLoading Phishing 2026...")

    if not PHISHING_FILE.exists():
        print(f"[MISSING] {PHISHING_FILE}")
        return

    df = pd.read_csv(PHISHING_FILE)

    text_col = find_column(df, ["text", "email", "body", "message", "content"])
    label_col = find_column(df, ["label", "class", "target", "type", "status"])

    if text_col is None or label_col is None:
        print(f"[SKIPPED] Phishing 2026 required columns not found. Columns: {list(df.columns)}")
        return

    for index, row in df.iterrows():
        text = clean_text(row[text_col])
        raw_label = str(row[label_col]).lower().strip()

        if len(text) < 20:
            continue

        if raw_label in {"1", "phishing", "spam", "suspicious", "malicious"}:
            label = 1
        elif raw_label in {"0", "legitimate", "legit", "ham", "safe"}:
            label = 0
        else:
            continue

        rows.append(
            {
                "text": text,
                "label": label,
                "source": "phishing2026",
                "file_name": f"phishing2026_{index}",
            }
        )

    print(f"[FOUND] Phishing 2026 rows loaded: {len(df)}")


def main():
    rows: list[dict[str, object]] = []

    load_spamassassin(rows)
    load_enron(rows)
    load_phishing2026(rows)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No email data was loaded. Check dataset paths and CSV columns.")

    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset prepared successfully")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print("\nLabel counts:")
    print(df["label"].value_counts())
    print("\nSource counts:")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()
