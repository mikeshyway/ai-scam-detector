# AI-based Spam and Caller Fraud Detection System

Educational AI scam detection platform for Capstone Project 2. The system runs locally with
Python and Streamlit and demonstrates how machine learning can flag suspicious emails,
scam-style transcripts, suspicious phone-number patterns, and AI-generated speech.

This project is intentionally scoped as a learning and awareness tool, not as a commercial
cybersecurity or caller-ID product. It focuses on explainability, confidence scores,
visual dashboards, and clear feedback for students.

## App Pages

- Home: project focus, main feature pillars, and demo-data status.
- Dashboard: model readiness, temporary demo-data coverage, and session activity.
- Email Detection: phishing-style email/message analysis.
- Transcript Detection: scam transcript analysis for calls, Zoom, Teams, and Google Meet.
- Audio Detection: uploaded `.wav` and `.flac` analysis with waveform and spectrogram views.
- Phone Risk Demo: manual phone-number reputation demo using synthetic data.
- Model Comparison: TF-IDF, Naive Bayes, Decision Tree, MFCC, and SVM comparison.
- Explainability: pipeline explanation and source-code reference.
- Student Quiz: interactive scam-awareness practice questions.
- Session History: temporary browser-session detection log.

## Temporary Demo Data

The app includes self-forged synthetic demo data because official datasets may not be inserted
yet. Demo data is generated in `src/demo_data.py` and marked with:

```text
TEMPORARY_SYNTHETIC_DEMO_DATA_REMOVE_AFTER_OFFICIAL_DATASET_INSERTION
```

Remove this demo dependency from screenshots, dashboard claims, and demonstrations once the
official datasets and trained models are inserted.

## Features

- Email phishing detection using TF-IDF with Naive Bayes and Decision Tree models.
- Scam transcript detection using TF-IDF with Naive Bayes.
- AI-generated speech detection using MFCC audio features with an SVM classifier.
- Confidence scoring and Streamlit warning banners.
- Suspicious phrase highlighting.
- Audio playback, waveform visualization, and spectrogram visualization.
- Session-only history using `st.session_state`.
- Upload support for `.txt`, `.csv`, `.wav`, and `.flac`.

## Project Structure

```text
ai-scam-detector/
├── app/                 Streamlit web application pages
├── data/                Raw, processed, and demo data notes
├── models/              Trained model artifacts
├── notebooks/           Jupyter walkthrough notebooks
├── scripts/             Dataset preparation and training scripts
└── src/                 Shared preprocessing, model, demo-data, and explainability code
```

## Setup

```bash
cd ai-scam-detector
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/00_setup_check.py
```

On Windows PowerShell:

```powershell
cd C:\Users\user\Documents\Codex\ai-scam-detector
pip install -r requirements.txt
streamlit run app/main.py
```

## Dataset Placement

Place official datasets in the following folders before training:

```text
data/raw/spamassassin/spam/
data/raw/spamassassin/ham/
data/raw/transcripts/scam_nonscam_calls.csv
data/raw/transcripts/youtube_scam_transcripts.csv
data/raw/asvspoof_subset/labels.csv
data/raw/asvspoof_subset/*.flac
```

Expected audio labels:

- `0`, `real`, `human`, or `bonafide` means real speech.
- `1`, `fake`, `synthetic`, `spoof`, or `ai` means AI-generated/synthetic speech.

## Training

Run the scripts in order:

```bash
python scripts/02_train_email_model.py
python scripts/03_train_transcript_model.py
python scripts/01_prepare_audio.py --max-real 300 --max-fake 300
python scripts/04_train_audio_model.py
```

Generated artifacts are saved in `models/` and `data/processed/`.

## Run the Streamlit App

```bash
streamlit run app/main.py
```

For Streamlit Cloud, set the main file path to:

```text
app/main.py
```

Keep `requirements.txt` at the repository root.

## Truecaller-Style Scope Note

This Streamlit app cannot automatically record phone calls, intercept caller ID, or monitor
live phone conversations. A real Truecaller-style system would need a separate mobile app,
telephony integration, explicit user consent, platform permissions, and legal review.

The included Phone Risk Demo is only a manual, synthetic reputation checker for educational
discussion.

## Ethical Use

Predictions may be wrong. Do not use this project as the sole basis for security, legal,
financial, or disciplinary decisions.
