# Data Pipeline

The final evidence notebook,
`notebooks/00_final_prototype_evidence_notebook.ipynb`, is the main
reviewer-facing proof for dataset readiness and charts. This document keeps the
channel-by-channel data flow and dataset-purpose notes.

## Lifecycle

```text
Raw dataset
  -> preprocessing script
  -> processed channel dataset
  -> training script
  -> model artifact
  -> evaluation metrics
  -> Streamlit inference
```

Raw files are treated as source material and are not modified. Processed files
can be regenerated.

## Dataset Purpose

| Channel | Dataset Purpose | Important Boundary |
| --- | --- | --- |
| Email/message | Train phishing/legitimate text classifiers from email/message examples. | Metrics describe the available dataset mix, not universal accuracy. |
| Transcript | Train scam-intent classifiers from labeled call or conversation text. | Scam-only transcript sources are useful for demos, not as the only binary training source. |
| Audio | Train bonafide/spoof voice classifiers using ASVspoof-style audio evidence. | Audio uses train/dev style validation rather than the text 80/20 split. |
| Phone | Provide traceable fallback/provider-style evidence fields for lookup demonstration. | Phone evidence is not local ML training data. |

## Email

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

## Transcript

```text
data/raw/voice_transcript/
|-- call_transcripts_scam_determinations/
`-- youtube_scam_phone_call_transcripts/
  -> scripts/02_prepare_transcript_dataset.py
  -> data/processed/transcript/transcript_dataset.csv
  -> scripts/05_train_transcript_model.py
```

The YouTube collection contains scam examples and is not sufficient as the
only binary-classification source. The labeled call dataset supplies both
classes.

## Audio

```text
data/raw/asvspoof_2019_dataset_subset/
  -> scripts/03_prepare_audio_dataset.py
  -> data/processed/audio/{train,dev,labels.csv}
  -> scripts/06_train_audio_model.py
  -> scripts/07_train_audio_behavior_model.py
```

The preparation workflow creates a balanced capstone-sized ASVspoof subset.
MFCC features are used for the calibrated SVM. Behavioral features are used
for the optional Random Forest layer.

## Phone

`data/processed/phone/phone_dataset.csv` is for real, traceable fallback
records only. It is queried only when the selected live provider is unavailable,
rate-limited, unauthorized, or has no matching record.

Fictional presentation records belong in `data/demo/phone_demo_dataset.csv` and
are queried only when the Phone Number tab's Demo Mode is explicitly enabled.
Demo records must be labelled with `record_type=demo`, `is_demo=true`,
`source_reference`, and `last_verified`.

The final notebook may also read `data/demo/demo_phone_numbers.csv` as a small
repeatable demonstration file. These demo rows are for presentation support
only and should not be described as trained phone-risk values.

## Data Safety

- Do not commit licensed or large raw datasets.
- Do not place real personal phone numbers in the fallback CSV.
- Keep labels and source provenance in processed datasets.
- Regenerate processed data after changing preprocessing logic.
