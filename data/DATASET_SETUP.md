# Dataset Setup

This file is a quick placement reference. The maintained pipeline explanation
is in `docs/DATA_PIPELINE.md`. The final reviewer evidence is in
`notebooks/00_final_prototype_evidence_notebook.ipynb`.

## Email

Place the downloaded email collections under:

```text
data/raw/email/
|-- spamassassin_public_corpus/
|-- the_enron_email_dataset/
`-- phishing_and_legitimate_emails_dataset_for_ml_2026/
```

Then run:

```powershell
py scripts\01_prepare_email_dataset.py
py scripts\04_train_email_model.py
```

Purpose: train phishing/legitimate message classifiers. The final notebook
uses saved metrics and charts instead of retraining these models.

## Voice Transcripts

```text
data/raw/voice_transcript/
|-- call_transcripts_scam_determinations/
`-- youtube_scam_phone_call_transcripts/
```

CSV and TXT sources are supported by the preprocessing workflow. The YouTube
dataset is demo/scam-only evidence and must not be the sole source for binary
model training.

```powershell
py scripts\02_prepare_transcript_dataset.py
py scripts\05_train_transcript_model.py
```

Purpose: train scam-intent classifiers for call or meeting transcript text.
Scam-only transcript sources are useful for examples but should not be the only
binary training source.

## ASVspoof Audio

Place the ASVspoof 2019 LA protocol and audio folders under:

```text
data/raw/asvspoof_2019_dataset_subset/
|-- ASVspoof2019_LA_cm_protocols/
|-- ASVspoof2019_LA_train/flac/
`-- ASVspoof2019_LA_dev/flac/
```

Then run:

```powershell
py scripts\03_prepare_audio_dataset.py
py scripts\06_train_audio_model.py
py scripts\07_train_audio_behavior_model.py
```

The preparation command creates:

```text
data/processed/audio/labels.csv
data/processed/audio/train/
data/processed/audio/dev/
```

Purpose: train and validate bonafide/spoof voice models. Audio uses an
ASVspoof train/dev style workflow rather than the text 80/20 split.

## Phone Fallback

Real, traceable fallback records belong at:

```text
data/processed/phone/phone_dataset.csv
```

Do not place synthetic demo rows in the normal fallback file. Fictional
presentation rows belong at:

```text
data/demo/phone_demo_dataset.csv
data/demo/demo_phone_numbers.csv
```

Demo records are used only when the Phone Number tab's Demo Mode is explicitly
enabled. See `docs/PHONE_MODULE.md` for the required schemas.

Purpose: phone records support provider-style evidence and repeatable
demonstrations. They are not local ML training data.

## Repository Policy

Raw and processed datasets are ignored by Git because they are large,
generated, or subject to dataset licenses. Preserve `.gitkeep` placeholders
when sharing the repository, and document the official download sources in the
capstone report.
