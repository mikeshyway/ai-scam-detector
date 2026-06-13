# AI-based Spam and Caller Fraud Detection System

Educational AI scam detection platform for Capstone Project 2. The system runs locally with
Python and Streamlit and demonstrates how machine learning can flag suspicious uploaded
emails, scam-style transcripts, suspicious phone-number patterns, and AI-generated speech.

This project is intentionally scoped as a learning and awareness tool, not as a commercial
cybersecurity, caller-ID, or telecom product. It focuses on explainability, confidence scores,
clean visual feedback, and downloadable evidence summaries.

## App Pages

- Scam Simulation Lab: uploaded call/meeting recording chunk analysis.
- Live Audio Detection: browser-microphone chunk analysis with optional local Whisper
  transcription, transcript scam scoring, MFCC/SVM voice scoring, and report-history saving.
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
- Browser microphone analysis using configurable 3-10 second chunks.
- Reliable built-in microphone recording that avoids WebRTC/STUN/TURN negotiation.
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

For optional local Whisper transcription on the Live Audio Detection page:

```bash
sudo apt update
sudo apt install ffmpeg
pip install -r requirements-live.txt
```

The first recorded clip downloads the selected Whisper model. The model is then cached and
reused for later clips. The `tiny` model is recommended for CPU demonstrations. Without
Whisper, the page supports audio-only MFCC/SVM analysis or a clearly labelled manual transcript
fallback.

Browser microphone access works on `localhost`. Remote deployments require HTTPS, and some
restricted networks or cloud hosts also require a configured TURN server for WebRTC.

The Live Audio Detection page uses Streamlit's built-in `st.audio_input` as its primary path.
Record for roughly 5-10 seconds and stop; the app automatically transcribes and divides the
recording into analysis chunks. Select **Record next clip** to continue the same conversation
session. Earlier transcript, frequency, MFCC, and risk results remain visible, and the complete
session can be saved to the AI Report Generator. This path does not use WebRTC and is recommended
for local and Streamlit Cloud demonstrations.

**Advanced local WebRTC experiment** is collapsed and disabled by default. It remains available
for continuous local chunk updates, but it requires a working TURN relay on many hosted or
restricted networks and is not the supported presentation path.

### TURN Setup for Hosted Live Audio

The default STUN-only connection may stall on Streamlit Community Cloud, university Wi-Fi,
VPNs, corporate firewalls, or carrier-grade NAT. For a hosted demonstration, configure TURN
credentials in Streamlit Cloud under **Settings > Secrets**.

Recommended Twilio Network Traversal Service configuration:

```toml
[webrtc]
twilio_account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
twilio_auth_token = "replace-with-real-token"
```

The app uses Twilio's Tokens API to request short-lived ICE servers and refreshes the cached
configuration hourly. No Twilio credentials are sent to the browser.

A static TURN provider or self-hosted `coturn` server is also supported:

```toml
[webrtc]
turn_urls = [
  "turn:turn.example.com:3478?transport=udp",
  "turn:turn.example.com:3478?transport=tcp",
  "turns:turn.example.com:443?transport=tcp",
]
turn_username = "replace-with-turn-username"
turn_credential = "replace-with-turn-password"
```

See `.streamlit/secrets.toml.example`. Never commit the real `.streamlit/secrets.toml` file.
For local use through `http://localhost`, TURN is normally unnecessary.

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

The system supports uploaded evidence and an educational browser-microphone demonstration.
The live page analyses only the microphone selected by the browser. It does not automatically
intercept phone calls, meeting platforms, system audio, emails, or messages. Capturing another
speaker requires speaker playback into the microphone or a separately configured virtual audio
cable. Automatic pre-delivery monitoring remains outside the Streamlit prototype.

## Change Log

Implementation changes and removed/ineligible proposal items are documented in:

```text
changes.md
```

## Ethical Use

Predictions may be wrong. Do not use this project as the sole basis for security, legal,
financial, or disciplinary decisions.
