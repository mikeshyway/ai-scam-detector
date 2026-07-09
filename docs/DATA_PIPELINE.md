# Data Pipeline

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

`data/processed/phone/phone_dataset.csv` is a fictional educational fallback,
not ML training data. It is queried only when the PenipuMY API is unavailable,
rate-limited, unauthorized, or has no matching record.

## Data Safety

- Do not commit licensed or large raw datasets.
- Do not place real personal phone numbers in the fallback CSV.
- Keep labels and source provenance in processed datasets.
- Regenerate processed data after changing preprocessing logic.
