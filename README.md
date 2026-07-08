# AI-based Spam and Caller Fraud Detection System

Educational AI scam detection platform for Capstone Project 2. The system runs locally with
Python and Streamlit and demonstrates how machine learning can flag suspicious uploaded
emails, scam-style transcripts, suspicious phone-number patterns, and AI-generated speech.

This project is intentionally scoped as a learning and awareness tool, not as a commercial
cybersecurity, caller-ID, or telecom product. It focuses on explainability, confidence scores,
clean visual feedback, and downloadable evidence summaries.

## App Pages

- Scam Simulation Lab: uploaded call/meeting recording chunk analysis.
- Live Audio Detection: browser Voice Recorder plus local-only internal audio capture, both
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
- Local-only internal audio capture from system output, monitor sources, or virtual audio
  devices using `sounddevice`.
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

For local internal audio capture and Whisper transcription on the Live Audio Detection page:

```bash
python -m pip install sounddevice soundfile
python -m pip install -r requirements.txt
# Optional when FFmpeg is missing and Conda is available:
conda install -c conda-forge ffmpeg
```

These commands install into the currently active Python environment and do not require `sudo`.
After installation, restart Streamlit and press **Refresh diagnostics** in Device Audio Monitor.

The Live Audio Detection page opens in **Voice Recorder** mode. This uses Streamlit's browser
recorder. **Device Audio Monitor** is a separate local-only internal audio mode using
`sounddevice`. It captures manual 5-10 second chunks from system speaker output, BlackHole,
PulseAudio/PipeWire monitor sources, Stereo Mix, or virtual-cable devices, then saves each
temporary WAV for Whisper transcription and risk analysis.

Device Audio Monitor does not use the physical microphone. It cannot work properly on
Streamlit Cloud because cloud servers cannot access your laptop's Zoom/Meet/Teams audio.
Use the WAV upload fallback when internal capture is unavailable.

The Device Audio Monitor includes a collapsed **Audio setup and diagnostics** panel. It checks
`sounddevice`, `soundfile`, `ffmpeg`, Whisper, default input/output devices, and detected
system-audio or virtual-cable sources. Missing dependencies show exact installation commands.
The **Test selected device for 3 seconds** action verifies duration, RMS/peak level, silence,
WAV playback, and capture permission. Diagnostic snapshots are appended to
`logs/system_diagnostics.log`; log files are excluded from Git.

On Windows, test an input-capable source such as Stereo Mix or a virtual-cable output before
starting monitoring. The capture helper probes compatible sample rates, mono/stereo channels,
and latency settings. If every attempt fails, the page displays and logs the exact PortAudio
exception instead of a generic device message.

Windows capture backends are prioritized in this order: **WASAPI**, **DirectSound**, then
**MME**. Windows WDM-KS devices remain visible in diagnostics but are excluded from the capture
selector because PortAudio reports `Blocking API not supported yet` for this workflow. When the
same device name is exposed by several host APIs, the host API remains visible in its label.

Devices containing `CABLE`, `VB-Cable`, `VoiceMeeter`, or `BlackHole` are grouped as
**Meeting Capture Devices**, receive a **Recommended** badge, and are prioritized in the audio
device selector. Input-capable virtual endpoints are placed first because they can feed live
transcription and scam detection; playback-only routing endpoints remain labelled but cannot be
captured directly.

Both Live Audio tabs include a recording carousel so previous clips can be reviewed one at a
time with their own transcript, flags, risk score, MFCC heatmap, and spectrum.

If Whisper is unavailable, Device Audio Monitor keeps a demo fallback transcript mode so the
page remains usable for presentation. Whisper also requires the `ffmpeg` executable to be
installed and available on PATH when transcribing temporary audio files.

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

### Email Metrics Source

Email model metrics shown in Streamlit are loaded from:

```text
reports/metrics/email_model_metrics.json
```

That file is generated by the canonical training pipeline:

```text
data/raw/email/
-> src/preprocessing/prepare_email_dataset.py
-> data/processed/email/email_dataset.csv
-> src/training/train_email_model.py
-> reports/metrics/email_model_metrics.json
```

The legacy `scripts/02_train_email_model.py` now delegates to the canonical trainer above.
It should not create separate demo or stale email metrics.

In the Email tab:

- `Accuracy`, `Precision`, `Recall`, `F1 Score`, `ROC-AUC`, and `Training Time` are saved
  held-out evaluation metrics from the latest training run.
- `Risk Score`, `Confidence`, and `Prediction Time` are live values for the current uploaded
  or pasted evidence.
- `email_best.pkl` is saved as a benchmark artifact, but live analysis uses the selected
  multi-model comparison rather than only the benchmark model.

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

The system supports uploaded evidence, browser voice samples, and local internal-audio chunks.
The Live Audio Detection page can analyse manually recorded 5-10 second clips, but it does not
join meetings through platform APIs, capture mobile phone calls, monitor email/SMS delivery, or
block communications automatically. The internal-audio monitor is educational near-real-time
chunk analysis, not a commercial interception or prevention system.

## Change Log

Implementation changes and removed/ineligible proposal items are documented in:

```text
changes.md
```

## Ethical Use

Predictions may be wrong. Do not use this project as the sole basis for security, legal,
financial, or disciplinary decisions.
