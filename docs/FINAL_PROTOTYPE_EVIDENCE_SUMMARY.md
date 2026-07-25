# Final Prototype Evidence Summary

The maintained evidence record for the final AI-FDS prototype is:

```text
notebooks/00_final_prototype_evidence_notebook.ipynb
```

That notebook is the main reviewer-facing proof. It explains the methodology,
runs verification cells, displays saved metrics and charts, links source-code
ownership, and states the final limitations. This document is a text companion
for readers who want the final scope before opening the notebook.

## Final Prototype Scope

The final prototype has two top-level Streamlit pages:

1. Detection Center
2. AI Report Generator

The Detection Center contains three evidence tabs:

1. Email/message analysis
2. Transcript and uploaded/recorded audio analysis
3. Phone-number evidence checking

All three detection paths can produce structured evidence that is later used by
the AI Report Generator to build TXT, PDF, or DOCX reports.

## Evidence Strategy

| Channel | Main Evidence | Runtime Strategy | Output |
| --- | --- | --- | --- |
| Email/message | Pasted text or uploaded email/document content | Saved trained text classifiers plus suspicious-content indicators | Label, confidence, highlighted evidence, history row |
| Transcript | Pasted/uploaded transcript or transcribed speech | Saved transcript classifiers or optional transformer artifact | Scam-risk result, confidence, explanation, history row |
| Audio | Uploaded or recorded voice evidence | MFCC/statistical audio features, calibrated SVM, optional behavior model, and transcript-risk signals where available | Voice/audio concern signals and explanation |
| Phone number | Caller number and optional claimed identity | Veriphone metadata, PenipuMY reputation/report fields where configured, local/demo fallback where allowed, and transparent evidence rules | Concern priority, provider status, evidence score, recommended verification action |

Phone evidence is not a locally trained phone-risk model. It is provider-pulled
metadata/reputation evidence interpreted with transparent rules.

## Training And Model Selection

Email and transcript models use processed labeled text datasets and a stratified
80/20 train/test split. Stratification keeps class ratios similar between the
training and held-out evaluation portions.

Audio uses the ASVspoof train/dev style because the audio task is closer to a
signal-processing authenticity problem than a normal text classification task.

The dashboard does not retrain models when it opens. Training scripts prepare
datasets, train candidate models, save model artifacts under `models/`, and
write metric JSON files under `reports/metrics/`. The app loads those saved
artifacts for runtime detection.

## Reviewer Explanation

The professional explanation should be:

AI-FDS collects scam-risk evidence across message text, call transcript/audio,
and phone-provider evidence. Each input type uses the model or evidence strategy
appropriate to that data type. Text models produce labels and probabilities,
audio models produce voice/authenticity concern signals, and phone checks use
provider evidence plus deterministic concern rules. Selected results are saved
as report-ready evidence.

The plain-language explanation should be:

The dashboard does not decide guilt or safety by itself. It highlights warning
signs, shows where the concern came from, and helps a trained human reviewer
prepare a clear report.

## Companion Documents

- `README.md`: short project overview, setup, and final feature summary.
- `docs/ARCHITECTURE.md`: code ownership, runtime flow, and artifact layout.
- `docs/DATA_PIPELINE.md`: dataset purpose and preparation flow.
- `docs/MODEL_TRAINING.md`: training commands, model families, metrics, and artifact handoff.
- `docs/PHONE_MODULE.md`: phone provider flow, API keys, rules, history, and limitations.
- `data/DATASET_SETUP.md`: quick raw dataset placement guide.
- `notebooks/01_email_eda_model.ipynb`: email-focused appendix.
- `notebooks/02_transcript_eda_model.ipynb`: transcript-focused appendix.
- `notebooks/03_audio_eda_model.ipynb`: audio-focused appendix.

## Limitations

- Saved metrics describe the available validation data, not universal real-world accuracy.
- Email and transcript behavior depends on the processed datasets and feature coverage.
- Audio analysis can be affected by noise, accents, recording quality, and missing FFmpeg or Whisper dependencies.
- Phone provider evidence may be incomplete, unavailable, rate-limited, or different on a later lookup.
- Carrier metadata does not prove who is holding the phone.
- The report generator summarizes evidence; it does not create legal proof.
