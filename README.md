# AI-based Spam and Caller Fraud Detection System

AI-FDS is a local Streamlit capstone application for educational scam
detection and awareness. It analyses written messages, call transcripts,
uploaded voice evidence, and phone-number reputation data. Results combine
trained model predictions with human-readable evidence; the application is
not a commercial security, telecom, or forensic product.

## Active Features

- **Emails and Messages**: parses pasted text and uploaded email/document
  evidence, compares trained text classifiers, and explains suspicious and
  legitimate indicators.
- **Voice Transcript**: analyses pasted/uploaded transcripts and optional
  `.wav`, `.mp3`, `.m4a`, or `.flac` evidence using Whisper, the transcript
  classifiers, MFCC + calibrated SVM, and the optional behavioral model.
- **Phone Number**: queries PenipuMY when configured, falls back to the local
  educational dataset, and explains reputation evidence.
- **AI Report Generator**: builds downloadable evidence reports from saved
  detections.

## Directory Structure

```text
ai-scam-detector/
|-- main.py                  Optional root Streamlit entry point
|-- app/                     Active Streamlit pages and UI components
|-- src/
|   |-- audio/               Audio loading, features, inference, recording helpers
|   |-- data/                Synthetic/demo data helpers
|   |-- phone/               API client, fallback lookup, rules, explanations
|   |-- preprocessing/       Canonical dataset preparation logic
|   |-- reporting/           History persistence and report generation
|   |-- text/                Text preprocessing, inference, explainability
|   |-- training/            Canonical model training/evaluation logic
|   `-- utils/               Time and system diagnostic helpers
|-- scripts/                 Thin command-line entry points
|-- data/
|   |-- raw/                 Original datasets, kept out of Git
|   `-- processed/           Generated channel-specific datasets
|-- models/                  Runtime model artifacts
|-- reports/metrics/         Saved evaluation metrics
|-- notebooks/               Exploratory capstone notebooks
|-- tests/                   Automated unit tests
|-- docs/                    Architecture and setup documentation
`-- archive/deprecated/      Superseded files kept for reference
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership rules and the
complete execution flow.

## Installation

From Windows PowerShell:

```powershell
cd C:\Users\user\Documents\Codex\ai-scam-detector
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py scripts\00_setup_check.py
```

FFmpeg must also be installed and available on `PATH` for Whisper and
compressed audio formats such as MP3/M4A.

## Run the App

Use either entry point:

```powershell
py -m streamlit run app\main.py
```

```powershell
py -m streamlit run main.py
```

`app/main.py` remains the canonical Streamlit Cloud entry point.

## Prepare Datasets

Place datasets under the channel folders described in
[docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md), then run:

```powershell
py scripts\01_prepare_email_dataset.py
py scripts\02_prepare_transcript_dataset.py
py scripts\03_prepare_audio_dataset.py
```

Generated datasets are written to:

```text
data/processed/email/email_dataset.csv
data/processed/transcript/transcript_dataset.csv
data/processed/audio/labels.csv
data/processed/audio/train/
data/processed/audio/dev/
data/processed/phone/phone_dataset.csv
```

## Train Models

```powershell
py scripts\04_train_email_model.py
py scripts\05_train_transcript_model.py
py scripts\06_train_audio_model.py
py scripts\07_train_audio_behavior_model.py
```

The numbered workflow is deliberate: `00` validates the environment, `01-03`
prepare datasets, and `04-07` train models. Scripts are thin launchers only.
Reusable functions and classes belong under `src/`; do not place heavy logic
directly in `scripts/`. See
[docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) for inputs and outputs.

## Phone API Configuration

The phone module reads the PenipuMY API key from `PENIPU_API_KEY`.

```powershell
$env:PENIPU_API_KEY="your-key"
py -m streamlit run app\main.py
```

Alternatively, store it in the untracked
`.streamlit/secrets.toml` file. Never commit real API keys. The full fallback
behavior is documented in [docs/PHONE_MODULE.md](docs/PHONE_MODULE.md).

## Generated Artifacts

- Trained models: `models/`
- Evaluation JSON: `reports/metrics/`
- Local diagnostics: `logs/`
- Report/history database: generated locally and ignored by Git

Large datasets, model binaries, local databases, logs, caches, and secrets are
ignored. Keep only the artifacts required for the final offline demonstration
when packaging the capstone submission.

## Detection Scope and Limitations

- Predictions can be wrong and require human verification.
- Explainability indicators describe evidence; they do not override trained
  model predictions.
- Uploaded audio is analysed after recording/upload, not intercepted from live
  calls or meetings.
- Phone results represent API/local reputation evidence, not telecom identity
  verification.
- Whisper transcription quality depends on language, noise, and FFmpeg.
- Do not use AI-FDS as the sole basis for financial, legal, security, or
  disciplinary action.

Implementation history and scope changes are recorded in
[changes.md](changes.md).
