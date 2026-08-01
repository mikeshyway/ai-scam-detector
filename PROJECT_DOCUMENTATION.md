# AI-FDS Project Documentation

This document is the merged technical companion for the final AI-FDS capstone
prototype. It connects the final report narrative to the files that exist in
the code project: Streamlit pages, processed datasets, model artifacts, metric
JSON files, curated demo evidence, SQLite history, and report exports.

## Contents

- [Final Prototype Evidence Summary](#final-prototype-evidence-summary)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Phone Module](#phone-module)

---

## Final Prototype Evidence Summary

The maintained evidence record for the final AI-FDS prototype is:

```text
notebooks/00_final_prototype_evidence_notebook.ipynb
```

That notebook is the main reviewer-facing proof. It explains the methodology,
runs verification cells, displays saved metrics and charts, links source-code
ownership, and records the final evidence flow. This document is a text companion
for readers who want the final scope before opening the notebook.

### Final Prototype Scope

The final prototype has two top-level Streamlit pages:

1. Detection Center
2. AI Report Generator

The Detection Center contains three evidence tabs:

1. Email/message analysis
2. Transcript and uploaded/recorded audio analysis
3. Phone-number evidence checking

All three detection paths can produce structured evidence that is later used by
the AI Report Generator to build TXT, PDF, or DOCX reports.

### Evidence Strategy

| Channel | Main Evidence | Runtime Strategy | Output |
| --- | --- | --- | --- |
| Email/message | Pasted text or uploaded email/document content | Saved trained text classifiers plus suspicious-content indicators | Label, confidence, highlighted evidence, history row |
| Transcript | Pasted/uploaded transcript or transcribed speech | Saved transcript classifiers plus DistilBERT contextual artifact | Scam-risk result, confidence, explanation, history row |
| Audio | Uploaded or recorded voice evidence | MFCC/statistical audio features, calibrated SVM, behavior model, trained voice-evidence calibrator, and transcript-risk signals | Voice/audio concern signals and explanation |
| Phone number | Caller number and optional claimed identity | Veriphone.io metadata, PenipuMY reputation/report fields where configured, and transparent evidence rules | Concern priority, provider status, evidence score, recommended verification action |

Phone evidence uses provider-pulled metadata/reputation fields interpreted with
transparent rules.

### Training And Model Selection

Email and transcript models use processed labeled text datasets and a stratified
80/20 train/test split. Stratification keeps class ratios similar between the
training and held-out evaluation portions.

Audio uses the ASVspoof train/dev style because the audio task is closer to a
signal-processing authenticity problem than a normal text classification task.

Training scripts prepare datasets, train candidate models, save model artifacts
under `models/`, and write metric JSON files under `reports/metrics/`. The app
loads those saved artifacts for runtime detection.

For audio, the upstream MFCC model and behavior model produce raw signals, then
`scripts/08_train_audio_voice_evidence_calibrator.py` trains the second-stage
voice-evidence calibrator that converts raw voice, behavior, and quality fields
into the dashboard-facing voice evidence risk.

### AI Report Generator Workflow

The report generator packages evidence produced by the Detection Center. It
uses saved scan rows from `data/session_history.db`, then builds a selected
case preview and export payload.

| Step | Purpose | Reviewer-facing output |
| --- | --- | --- |
| Save detection evidence | Email, transcript/audio, and phone tabs write structured rows to local history, including prediction, concern score, explanation, provenance, and dashboard artifacts. | A traceable scan-history row. |
| Filter and select rows | The report page loads `data/session_history.db`, excludes simulation rows, and lets the user filter by date, evidence type, and action status. | A clean selected-evidence list for the case. |
| Preview the report | `build_preview()` converts selected rows into a readable text preview before export. | Early check that the report matches what the reviewer intends to share. |
| Export the Student Brief | `build_report()` dispatches TXT, PDF, or DOCX output. | Downloadable report with summary, evidence, explanations, visuals/data, risk notes, recommendations, and appendix. |
| Log export metadata | `log_export()` records format, filename, selected row ids, SHA-256, and report manifest. | Evidence that the exported file can be traced back to selected history rows. |

The important presentation point is simple: the report generator is an evidence
packaging tool. It makes dashboard results shareable, traceable, and easier to
review during the capstone demonstration.

### Reviewer Explanation

The professional explanation should be:

AI-FDS collects scam-risk evidence across message text, call transcript/audio,
and phone-provider evidence. Each input type uses the model or evidence strategy
appropriate to that data type. Text models produce labels and probabilities,
audio models produce voice/authenticity concern signals, and phone checks use
provider evidence plus deterministic concern rules. Selected results are saved
as report-ready evidence.

The plain-language explanation should be:

The dashboard highlights warning signs, shows where the concern came from, and
helps a reviewer prepare a clear report from selected evidence.

### Companion Sections

- `README.md`: short project overview, setup, and final feature summary.
- Architecture: code ownership, runtime flow, and artifact layout.
- Data Pipeline: dataset purpose and preparation flow.
- Model Training: training commands, model families, metrics, and artifact handoff.
- Phone Module: phone provider flow, API keys, rules, history, and demo evidence.
- `data/DATASET_SETUP.md`: quick raw dataset placement guide.
- `notebooks/01_email_eda_model.ipynb`: email-focused appendix.
- `notebooks/02_transcript_eda_model.ipynb`: transcript-focused appendix.
- `notebooks/03_audio_eda_model.ipynb`: audio-focused appendix.

### Verified Repository Evidence

| Evidence Area | Current File Proof | Current Count Or Artifact |
| --- | --- | --- |
| Email dataset | `data/processed/email/email_dataset.csv` | 33,884 processed rows |
| Transcript dataset | `data/processed/transcript/transcript_dataset.csv` | 578 processed rows |
| Audio dataset index | `data/processed/audio/labels.csv` | 2,600 indexed ASVspoof rows |
| Email models | `models/email_*.pkl` and `reports/metrics/email_model_metrics.json` | NB, DT, SVM, RF, XGBoost, selected benchmark |
| Transcript models | `models/transcript_*.pkl`, `models/transcript_distilbert/`, and `reports/metrics/transcript_model_metrics.json` | NB, SVM, DistilBERT |
| Audio models | `models/audio_svm.pkl`, `models/audio_behavior_rf.pkl`, `models/audio_voice_evidence_calibrator.pkl` | MFCC SVM, behavior RF, voice calibrator |
| Demo uploads | `data/demo/uploads/` | 16 upload-ready email/transcript text files |
| Seeded history | `data/session_history.db` | 8 demo scan rows and export metadata |

---

## Architecture

The final prototype architecture is summarized and verified in
`notebooks/00_final_prototype_evidence_notebook.ipynb`. This file keeps the
codebase ownership and execution-flow view, while the notebook remains the
main evidence record for reviewers.

### Ownership Rules

| Directory | Responsibility |
| --- | --- |
| `app/` | Streamlit rendering, navigation, session state, and user actions |
| `src/audio/` | Audio decoding, feature extraction, inference, and recording helpers |
| `src/text/` | Text preprocessing, classifier loading, rules, and explainability |
| `src/phone/` | Veriphone.io carrier lookup, PenipuMY reputation lookup, rules, and explanations |
| `src/reporting/` | Saved history, evidence snapshots, export logging, and TXT/PDF/DOCX report generation |
| `src/data/` | Synthetic/demo data helpers only |
| `src/utils/` | Cross-cutting time and system diagnostic helpers |
| `src/preprocessing/` | Reusable `*_preprocessor.py` dataset workflows |
| `src/training/` | Reusable `*_trainer.py` training/evaluation workflows |
| `scripts/` | Numbered thin entry points that only call `src` modules |
| `tests/` | Unit tests against current public module paths |

The UI may import `src` modules. Source modules must not import Streamlit page
modules. Do not place heavy reusable logic directly in `scripts/`; reusable
functions and classes belong in `src/`. Script files should contain no
training or preprocessing logic beyond environment setup and calling a
canonical `main()` function.

### Application Flow

```text
app/main.py
  -> app/detection_center_page.py or app/report_page.py
  -> channel-specific app tab
  -> src runtime modules
  -> models/, data/processed/, reports/metrics/
```

The active page graph contains two top-level pages:

1. Detection Center
2. AI Report Generator

Detection Center routes internally to three evidence workflows:

1. Email/message analysis
2. Transcript and uploaded/recorded audio analysis
3. Phone-number evidence checking

Detection results can be stored as structured evidence and handed to the AI
Report Generator for TXT, PDF, or DOCX export.

The report generator uses the existing evidence history rather than producing
new detections. Detection tabs call the reporting/history helpers to persist
rows in `data/session_history.db`; `app/report_page.py` then syncs session
history, filters reportable rows, lets the user select evidence, previews the
case material, and calls `src/reporting/report_builder.py` for TXT, PDF, or
DOCX output. Completed exports are logged in the `report_exports` table with
format, filename, selected row ids, and SHA-256 metadata.

### Runtime Evidence Flow

```text
User evidence
  -> Streamlit tab in app/
  -> src/ domain logic where applicable
  -> saved model artifact, provider evidence, or deterministic rules
  -> result explanation and history row
  -> report generator filter/select/preview/export
  -> export metadata log
```

Email and transcript tabs load saved text model artifacts from `models/`.
Audio analysis uses audio feature/inference helpers and saved audio artifacts
including the second-stage voice-evidence calibrator. Phone checks use provider
evidence from Veriphone.io and PenipuMY where configured, plus transparent
rules.

### Reporting Modules

| File | Responsibility |
| --- | --- |
| `src/reporting/history_db.py` | Creates and queries `scan_history`, syncs Streamlit session history, records individual detections, and logs report exports. |
| `src/reporting/evidence_snapshot.py` | Normalizes dashboard evidence, provenance, action status, remediation, and report-ready evidence bundles. |
| `src/reporting/report_builder.py` | Builds report previews and TXT/PDF/DOCX Student Brief or technical outputs from selected history rows. |
| `src/reporting/chart_renderer.py` | Renders saved Plotly chart artifacts into images for document exports when possible. |

### Python Packages

Every importable source directory contains `__init__.py`. Runtime imports use
fully qualified paths such as:

```python
from src.audio.live_audio_analysis import analyse_live_chunk
from src.text.explainability import find_suspicious_phrases
from src.phone.phone_lookup import lookup_phone
```

`__pycache__/`, `.pyc`, Numba caches, virtual environments, logs, secrets, and
temporary files are generated locally and excluded from Git.

### Stable Artifact Layout

Model files remain flat under `models/` because current runtime loaders use
stable artifact names such as `email_nb.pkl`, `transcript_svm.pkl`, and
`audio_svm.pkl`. Moving them into channel subdirectories would create a broad
and unnecessary migration risk.

Metrics are already grouped under `reports/metrics/`. Raw and processed
datasets are grouped by channel under `data/`.

Notebook evidence and EDA appendices remain under `notebooks/`. The final
evidence notebook should be treated as the main documentation reference, while
the channel-specific notebooks provide focused appendices.

---

## Data Pipeline

The final evidence notebook,
`notebooks/00_final_prototype_evidence_notebook.ipynb`, is the main
reviewer-facing proof for dataset readiness and charts. This document keeps the
channel-by-channel data flow and dataset-purpose notes.

### Lifecycle

```text
Raw dataset
  -> preprocessing script
  -> processed channel dataset
  -> training script
  -> model artifact
  -> optional calibration artifact
  -> evaluation metrics
  -> Streamlit inference
```

Raw files are treated as source material and are not modified. Processed files
can be regenerated.

### Dataset Purpose

| Channel | Dataset Purpose | Dashboard Use |
| --- | --- | --- |
| Email/message | Train phishing/legitimate text classifiers from email/message examples. | Provides the Email and Messages tab with saved classifier predictions, confidence, and highlighted text evidence. |
| Transcript | Train scam-intent classifiers from labeled call or conversation text. | Provides the Voice Transcript tab with NB, SVM, and DistilBERT transcript predictions. |
| Audio | Train bonafide/spoof voice classifiers and calibrate voice evidence risk using ASVspoof-style audio evidence. | Provides MFCC, behavior, and calibrated voice-evidence signals for uploaded or recorded audio. |
| Phone | Provide traceable provider-style evidence fields for lookup demonstration. | Provides metadata/reputation rows for phone evidence walkthroughs and curated demos. |

### Email

```text
data/raw/email/
  -> scripts/01_prepare_email_dataset.py
  -> data/processed/email/email_dataset.csv
  -> scripts/04_train_email_model.py
  -> models/email_*.pkl
  -> reports/metrics/email_model_metrics.json
```

Expected raw collections are the SpamAssassin corpus, Enron legitimate email,
and the phishing/legitimate email dataset used by the preprocessing module.

### Transcript

```text
data/raw/voice_transcript/
|-- call_transcripts_scam_determinations/
`-- youtube_scam_phone_call_transcripts/
  -> scripts/02_prepare_transcript_dataset.py
  -> data/processed/transcript/transcript_dataset.csv
  -> scripts/05_train_transcript_model.py
```

The labeled call dataset supplies both classes. The YouTube collection adds
scam-language examples for reviewer walkthroughs and transcript demonstrations.

### Audio

```text
data/raw/asvspoof_2019_dataset_subset/
  -> scripts/03_prepare_audio_dataset.py
  -> data/processed/audio/{train,dev,labels.csv}
  -> scripts/06_train_audio_model.py
  -> scripts/07_train_audio_behavior_model.py
  -> scripts/08_train_audio_voice_evidence_calibrator.py
```

The preparation workflow creates a balanced capstone-sized ASVspoof subset.
MFCC features are used for the calibrated SVM. Behavioral features are used
for the optional Random Forest layer. The final calibrator converts those raw
voice and behavior signals, plus speech-quality fields, into the
dashboard-facing voice evidence risk.

### Phone

`data/processed/phone/phone_dataset.csv` is for real, traceable fallback
records using the provider-style schema.

Fictional presentation records belong in `data/demo/phone_demo_dataset.csv` and
are queried when the Phone Number tab's Demo Mode is explicitly enabled.
Demo records must be labelled with `record_type=demo`, `is_demo=true`,
`source_reference`, and `last_verified`.

The final notebook may also read `data/demo/demo_phone_numbers.csv` as a small
repeatable demonstration file. These demo rows are for presentation support
only and are labelled as reserved demo inputs.

### Data Safety

- Keep licensed or large raw datasets in the local `data/raw/` evidence tree.
- Keep personal phone numbers out of public fallback/demo files.
- Keep labels and source provenance in processed datasets.
- Regenerate processed data after changing preprocessing logic.

---

## Model Training

Run commands from the repository root with the same Python environment used by
Streamlit.

The final evidence notebook,
`notebooks/00_final_prototype_evidence_notebook.ipynb`, reads saved artifacts
and metrics instead of retraining models. Use this file for reviewer evidence,
and use this document for the training command boundary and model-selection
logic.

### Recommended Order

```powershell
py scripts\01_prepare_email_dataset.py
py scripts\04_train_email_model.py

py scripts\02_prepare_transcript_dataset.py
py scripts\05_train_transcript_model.py

py scripts\03_prepare_audio_dataset.py
py scripts\06_train_audio_model.py
py scripts\07_train_audio_behavior_model.py
py scripts\08_train_audio_voice_evidence_calibrator.py
```

### Email Models

Input: `data/processed/email/email_dataset.csv`

Outputs include the TF-IDF vectorizer, Naive Bayes, Decision Tree, calibrated
SVM, Random Forest, XGBoost (when available), and the selected benchmark model
under `models/email_*.pkl`.

Metrics: `reports/metrics/email_model_metrics.json`

Training principle:

- Text is converted into TF-IDF features.
- Candidate classifiers are trained and compared with held-out metrics.
- The saved `email_best.pkl` artifact represents the selected benchmark model
  for the current dataset and metric evidence.

### Transcript Models

Input: `data/processed/transcript/transcript_dataset.csv`

Outputs use the `transcript_*.pkl` naming convention under `models/`.

Metrics: `reports/metrics/transcript_model_metrics.json`

Training principle:

- Transcript text is prepared separately from email because call language,
  urgency, threats, and OTP/payment requests have a different pattern.
- TF-IDF NB/SVM models are supported, and the saved DistilBERT artifact provides
  contextual transcript classification.
- The dashboard loads saved transcript artifacts from `models/` for runtime
  comparison.

### Audio Models

Input: `data/processed/audio/labels.csv` plus the `train/` and `dev/` audio
folders.

- `scripts/06_train_audio_model.py`: MFCC + calibrated SVM
- `scripts/07_train_audio_behavior_model.py`: behavioral features + Random Forest
- `scripts/08_train_audio_voice_evidence_calibrator.py`: second-stage voice-evidence calibration from raw audio, behavior, and quality signals

Outputs:

```text
models/audio_svm.pkl
models/audio_behavior_rf.pkl
models/audio_voice_evidence_calibrator.pkl
reports/metrics/audio_model_metrics.json
reports/metrics/audio_behavior_metrics.json
reports/metrics/audio_voice_evidence_metrics.json
```

Training principle:

- The MFCC/statistical SVM estimates raw bonafide/spoof-style voice risk.
- The behavior Random Forest estimates secondary speech-behavior risk.
- The voice-evidence calibrator is trained last so it can convert raw voice,
  behavior, and quality fields into the user-facing `voice_evidence_risk`.
- The dashboard presents the result as "voice evidence risk" or "voice
  authenticity concern".

### Validation

After training:

```powershell
py -m compileall app src scripts tests
py -m unittest discover -s tests
```

Restart the Streamlit process after replacing model artifacts so cached
resources are reloaded.

### Split And Selection Principles

| Channel | Split Strategy | Selection Evidence | Runtime Benefit |
| --- | --- | --- | --- |
| Email | Stratified 80/20 train/test split | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix where saved | Provides a stronger default classifier for email/message risk evidence. |
| Transcript | Stratified 80/20 train/test split | Same text-classification metrics, plus transformer metrics when trained | Helps score scam intent in call transcripts or transcribed speech. |
| Audio | ASVspoof train/dev style split | Audio metric JSON, confusion matrix, ROC where saved, behavior feature importance, voice-evidence calibrator metrics | Produces calibrated voice-authenticity concern signals for uploaded/recorded audio. |
| Phone | Provider-rule workflow | Provider response fields and deterministic evidence weights | Keeps caller evidence transparent through carrier, reputation, and rule evidence. |

The "best" or "recommended" model means the strongest candidate among the
saved validation results for the current capstone datasets and metric files.

### Expected Terminal Evidence

The numbered scripts should print enough information to confirm the workflow
without opening the dashboard:

| Command | Expected Evidence |
| --- | --- |
| `py scripts\00_setup_check.py` | Python version, dependency status, and expected folder readiness. |
| `py scripts\01_prepare_email_dataset.py` | Loaded email sources, row counts, label counts, processed CSV path. |
| `py scripts\02_prepare_transcript_dataset.py` | Loaded transcript sources, scam/non-scam label counts, processed CSV path. |
| `py scripts\03_prepare_audio_dataset.py` | ASVspoof subset readiness, train/dev counts, label index path. |
| `py scripts\04_train_email_model.py` | Candidate email model metrics, selected benchmark, saved artifact paths. |
| `py scripts\05_train_transcript_model.py` | Candidate transcript metrics, optional transformer status, saved artifact paths. |
| `py scripts\06_train_audio_model.py` | MFCC/SVM metrics and saved audio model/metric paths. |
| `py scripts\07_train_audio_behavior_model.py` | Behavior RF metrics, feature importance, saved model/metric paths. |
| `py scripts\08_train_audio_voice_evidence_calibrator.py` | Voice-evidence train/dev row counts, threshold metrics, ROC-AUC, feature importances, saved calibrator path. |

The Streamlit app then loads the saved artifacts from `models/` and saved metric
summaries from `reports/metrics/` during normal demo use.

---

## Phone Module

### Flow

```text
User phone number
  -> normalize to canonical international format
  -> configured live provider checks
  -> Veriphone.io metadata evidence
  -> PenipuMY reputation/report evidence where configured
  -> shared status and evidence record
  -> shared normalized record
  -> transparent rules and explanation
  -> Streamlit result/history
```

The final prototype evidence notebook treats phone checks as provider evidence,
with carrier, line, reputation, report, and rule-based concern fields. Veriphone.io
supplies carrier/line metadata. PenipuMY supplies reputation or report-oriented
fields where a valid key is configured.

### Phone Number Normalization

The UI accepts common local and international formats, then converts them into
one canonical E.164-style internal format before lookup.

Accepted examples:

```text
012-345 6789
0123456789
+60 12-345 6789
60123456789
(03) 1234 5678
```

Canonical internal examples:

```text
+60123456789
+60312345678
```

Veriphone.io receives the canonical E.164-style value, such as `+60123456789`. The app
rejects clearly invalid text, repeated plus signs, and alphabetic input.

### Live Provider

#### Veriphone.io Carrier Lookup

Veriphone.io is used as the active carrier and number metadata provider. It can
return validity, carrier, line type, E.164 phone number, national/local
formatting, country, country code, calling country code, region, timezone, and
current-carrier metadata when the selected lookup mode supports it.

Documentation: <https://veriphone.io/v3>

API keys: <https://veriphone.io/app>

Free-tier note: Veriphone.io currently gives free accounts 1,000 credits per
month with no credit card required. Standard validation uses 1 credit. Current
carrier lookup uses 10 credits, and the API returns HTTP 402 when credits are
exhausted. See <https://veriphone.io/pricing> and
<https://veriphone.io/v3>.

Carrier metadata contributes the line-profile part of the phone evidence view.
The concern score is built from provider reputation fields, claim-consistency
rules, and source provenance.

#### PenipuMY

PenipuMY is used as a reputation/report evidence provider when configured. Its
fields support the phone concern explanation and the report-ready evidence
bundle.

Free-tier note: PenipuMY currently lists a Free API tier at 100 requests per
day, authenticated with the `X-API-Key` header, with the daily limit resetting
at midnight Malaysia time. See <https://penipu.my/api/v1/docs>.

### API Key

For normal dashboard use, configure provider keys directly inside the Phone
Number tab. Each provider card lets the user enable the provider, paste a
session-only key, test the connection, and view diagnostics. This is the
simplest path for capstone demonstrations.

Streamlit secrets are also supported:

```toml
VERIPHONE_API_KEY = "..."
PENIPUMY_API_KEY = "..."

[veriphone]
api_key = "..."

[penipumy]
api_key = "..."
```

Keep real API keys in session input, environment variables, or Streamlit
secrets. The repository includes `.env.example` with blank provider placeholders.

### Legacy Local And Demo Evidence

Path: `data/processed/phone/phone_dataset.csv`

This file is retained for helper workflows and traceable historical records.
The active dashboard phone investigation uses Veriphone.io and PenipuMY provider
evidence, with demo rows kept in the dedicated demo files below.

Required columns:

```text
phone
police_report_count
verified_report_count
spam
fraud
business_tier
business_name
spoofing_report_count
source
record_type
is_demo
source_reference
last_verified
```

Rows in this file should use `record_type=real` and `is_demo=false`.

Path: `data/demo/phone_demo_dataset.csv`

This file contains reserved-number capstone fixtures. The Phone Number tab
searches it when Demo Mode is explicitly enabled. Demo results are labelled as
demonstration data and kept separate from dashboard headline KPIs.

Path: `data/demo/demo_phone_numbers.csv`

This smaller demo file may be used by notebook/demo workflows as repeatable
presentation input. It separates high-concern and lower-concern examples, uses
reserved drama/demo numbers, and documents expected dashboard outcomes.

Path: `data/demo/uploads/`

This folder contains individual `.txt` email and transcript examples that can
be uploaded directly to the dashboard. The matching CSV rows document expected
native predictions, action statuses, and curated concern scores.

Path: `data/session_history.db`

The lecturer demo database can be rebuilt with:

```powershell
py scripts\seed_demo_history.py
```

The seeded rows use `record_scope='seed_demo'`, display as demo ID `0` on the
report page, and keep negative internal row IDs so future user-created scans
begin at positive ID `1`.

Active dashboard order:

```text
Configured live providers
  -> normalized provider/demo evidence record
```

### Output Principles

- `Valid` means number format/routing appears valid.
- `Metadata available` means carrier or line information was returned.
- `Unknown` is the neutral status used for provider evidence that returns no
  report-oriented reputation fields.
- `High Risk` appears only when real reputation evidence or explicit fallback
  records support it.

The UI shows provenance for each result:

```text
Live provider: Veriphone.io / PenipuMY where configured
Provider returned: Carrier or validation metadata / No usable carrier metadata
Scam reputation available: Yes/No
```

### Charts

The Phone Number tab may show:

- Lookup Evidence Coverage
- Caller Claim Consistency
- Provider Response Completeness
- Session Lookup History after multiple phone lookups

These charts summarize available evidence and complement the final lookup
result.

### Phone Evidence Method

The phone module intentionally remains:

```text
Veriphone.io API
+ PenipuMY API where configured
+ normalization
+ transparent consistency rules
+ explainability
```

### Unknown Result

The Unknown result is the neutral phone-evidence status. The UI pairs it with
verification guidance and reminders about OTPs, passwords, banking details, and
personal information.

### History Database

Phone rows saved in `data/session_history.db` are report evidence rows. They can
contain masked phone numbers, provider status, provider-derived fields, concern
labels, concern scores, and recommended verification actions.

Stored phone outcomes come from provider-pulled evidence plus deterministic
rules at lookup time. Each row keeps the provider status and provenance needed
for report review.

### Module Responsibilities

- `veriphone_client.py`: Veriphone.io HTTP communication and response parsing
- `penipumy_client.py`: PenipuMY HTTP communication and response parsing
- `phone_lookup.py`: provider -> local -> demo/status orchestration
- `phone_rules.py`: transparent evidence-based reputation/context level
- `phone_explainability.py`: readable evidence and recommendations
- `ipqs_client.py`: legacy provider client kept for source-history context
