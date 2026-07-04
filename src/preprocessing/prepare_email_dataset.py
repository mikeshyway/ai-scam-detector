from pathlib import Path
import pandas as pd
import re
from email import policy
from email.parser import BytesParser

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "spamassassin"
OUTPUT_FILE = ROOT / "data" / "processed" / "email_dataset.csv"

FOLDERS = {
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
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-zA-Z0-9$£€%@.!? ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def main():
    rows = []

    for folder_name, label in FOLDERS.items():
        folder_path = RAW_DIR / folder_name

        if not folder_path.exists():
            print(f"[MISSING] {folder_path}")
            continue

        files = [p for p in folder_path.iterdir() if p.is_file()]

        print(f"[FOUND] {folder_name}: {len(files)} files")

        for file_path in files:
            raw_text = extract_email_text(file_path)
            clean = clean_text(raw_text)

            if len(clean) < 20:
                continue

            rows.append({
                "text": clean,
                "label": label,
                "source": folder_name,
                "file_name": file_path.name,
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset prepared successfully")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()