# AI-based Spam and Caller Fraud Detection System

Educational AI scam detection platform for Capstone Project 2. The system runs locally with
Python and Streamlit and demonstrates how machine learning can flag suspicious uploaded
emails, scam-style transcripts, suspicious phone-number patterns, and AI-generated speech.

This project is intentionally scoped as a learning and awareness tool, not as a commercial
cybersecurity, caller-ID, or telecom product. It focuses on explainability, confidence scores,
clean visual feedback, and downloadable evidence summaries.

## App Pages

- Scam Simulation Lab: uploaded call/meeting recording chunk analysis.
- Live Audio Detection: browser Voice Recorder plus in-device short audio recording, both
  with transcript/voice scoring and report-history saving.
- Detection Center: email, transcript, AI voice/deepfake, and phone-number risk checkers.
- AI Report Generator: downloadable TXT/PDF/DOCX evidence summary.

Removed by request: Home, Dashboard, Turn-Based Scenario, Transparency Hub, and Session History.

## Temporary Demo Data

The app includes self-forged synthetic demo data because official datasets may not be inserted
yet. Demo data is generated in `src/demo_data.py` and marked with:

```text
TEMPORARY_SYNTHETIC_DEMO_DATA_REMOVE_AFTER_OFFICIAL_DATASET_INSERTION
```

Remove this demo dependency from screenshots and demonstrations once the official datasets and
trained models are inserted.

## Features

- Email phishing detection using TF-IDF with Naive Bayes and Decision Tree models.
- Scam transcript detection using TF-IDF with Naive Bayes.
- AI-generated speech detection using MFCC audio features with an SVM classifier.
- Uploaded meeting/call recording chunk analysis with 5-10 second confidence results.
- Browser voice recording using Streamlit's built-in microphone recorder.
- In-device 5-10 second recording chunks using `audio-recorder-streamlit`.
- Optional local Whisper speech-to-text for combining spoken content and voice signals.
- AI report generation with TXT/PDF/DOCX downloads when dependencies are installed.
- Confidence scoring and Streamlit warning banners.
- Suspicious phrase highlighting.
- Audio playback, waveform visualization, and spectrogram visualization.
- Upload support for `.txt`, `.csv`, `.wav`, `.flac`, `.mp3`, and `.m4a`.

## Project Structure

```text
ai-scam-detector/
|-- app/                 Streamlit web application pages
|-- data/                Raw, processed, and demo data notes
|-- models/              Trained model artifacts
|-- notebooks/           Jupyter walkthrough notebooks
|-- scripts/             Dataset preparation and training scripts
`-- src/                 Shared preprocessing, model, demo-data, and explainability code
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

For local audio recording and Whisper transcription on the Live Audio Detection page:

```bash
sudo apt update
sudo apt install ffmpeg
pip install -r requirements.txt
```

The Live Audio Detection page opens in **Voice Recorder** mode. This uses Streamlit's browser
recorder. **Device Audio Monitor** is a separate in-device short recording mode using
`audio-recorder-streamlit`. It records manual 5-10 second chunks, saves each temporary WAV for
Whisper transcription, runs transcript and audio-risk analysis, and lets the user record another
chunk to simulate near-real-time monitoring.

If Whisper is unavailable, Device Audio Monitor keeps a demo fallback transcript mode so the
page remains usable for presentation.

On Windows PowerShell:

```powershell
cd C:\Users\user\Documents\Codex\ai-scam-detector
pip install -r requirements.txt
streamlit run app/main.py
```

## Dataset Placement

Detailed methodology setup instructions are in:

```text
data/DATASET_SETUP.md
```

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

## Detection Scope

The system supports uploaded evidence and local short audio recording. The Live Audio Detection
page can analyse manually recorded 5-10 second clips, but it does not join meetings through
platform APIs, capture mobile phone calls, monitor email/SMS delivery, or block communications
automatically. The local recorder is educational near-real-time chunk analysis, not a commercial
interception or prevention system.

## Change Log

Implementation changes and removed/ineligible proposal items are documented in:

```text
changes.md
```

## Ethical Use

Predictions may be wrong. Do not use this project as the sole basis for security, legal,
financial, or disciplinary decisions.
