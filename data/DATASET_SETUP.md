# Dataset and Methodology Setup Guide

This guide maps the current implementation to the Chapter 3 methodology.

## 3.1 Project Flow Overview

Use this workflow:

1. Confirm requirements and learning outcomes.
2. Insert official datasets.
3. Clean and preprocess text/audio data.
4. Run EDA notebooks.
5. Train text and audio models.
6. Integrate predictions into Streamlit pages.
7. Evaluate and tune models.
8. Deploy the Streamlit app for demonstration.

## 3.2 Phase 1: Business Requirements

Primary user need:

- Help students recognise scam pressure tactics.
- Explain why a message, transcript, number, or recording is suspicious.
- Let students practice safe responses through the Scam Simulation Lab.

System outcomes:

- Detection confidence score.
- Suspicious phrase highlights.
- Educational explanation.
- Uploaded recording chunk-confidence results.
- Simulation score and session history.
- Dashboard and transparency views.

Constraints:

- Runs locally with Python and Streamlit.
- Uses lightweight ML models suitable for capstone timeline.
- Supports uploaded-evidence detection only.
- Does not perform automatic pre-delivery monitoring.

## 3.3 Phase 2: Data Collection

Place official datasets here:

```text
data/raw/spamassassin/spam/
data/raw/spamassassin/ham/
data/raw/transcripts/scam_nonscam_calls.csv
data/raw/transcripts/youtube_scam_transcripts.csv
data/raw/asvspoof_subset/labels.csv
data/raw/asvspoof_subset/*.flac
```

Temporary demo data is stored in:

```text
data/demo/
```

Remove demo data from final claims once official data is inserted. The demo marker is:

```text
TEMPORARY_SYNTHETIC_DEMO_DATA_REMOVE_AFTER_OFFICIAL_DATASET_INSERTION
```

## 3.3.1 Datasets Used

Use these datasets:

- SpamAssassin Public Corpus for email spam/ham training.
- teeconnie scam-and-non-scam call conversation dataset for binary transcript training.
- YouTube scam transcript dataset for educational scenario examples only.
- ASVspoof 2019 LA subset for real/fake speech training.

The YouTube scam transcript dataset should not be used alone for binary classification because it only contains scam samples.

## 3.3.2 Data Storage Structure

Raw datasets remain in `data/raw/`.
Processed audio arrays are saved in:

```text
data/processed/audio_X.npy
data/processed/audio_y.npy
```

Trained models are saved in:

```text
models/
```

## 3.4 Phase 3: Data Cleaning and Preprocessing

Text preprocessing:

- Lowercase text.
- Replace URLs with `urltoken`.
- Replace email addresses with `emailtoken`.
- Replace phone numbers with `phonetoken`.
- Replace money values with `moneytoken`.
- Remove punctuation/noise.
- Apply TF-IDF vectorization.

Audio preprocessing:

- Load uploaded `.wav`, `.flac`, `.mp3`, or `.m4a` audio where supported.
- Convert to mono 16 kHz audio.
- Extract MFCC mean and standard deviation features.
- Save training features once to `.npy` files for faster training.
- Split uploaded recordings into 5-10 second chunks for simulation analysis.

## 3.4.3 Class Imbalance Handling

The audio preparation script supports balanced subsampling:

```bash
python scripts/01_prepare_audio.py --max-real 300 --max-fake 300
```

Text models use stratified train/test splits.

## 3.5 Phase 4: Exploratory Data Analysis

Use notebooks:

```text
notebooks/01_email_eda_model.ipynb
notebooks/02_transcript_eda_model.ipynb
notebooks/03_audio_eda_model.ipynb
```

Recommended EDA:

- Class distribution.
- Common suspicious words.
- Scam vs non-scam examples.
- Audio waveform and MFCC plots.
- Spectrogram inspection.

## 3.6 Phase 5: Model Selection and Training

Text models:

- TF-IDF feature extraction.
- Naive Bayes classifier.
- Decision Tree classifier for explainable comparison.

Audio models:

- MFCC feature extraction.
- SVM classifier.

Training commands:

```bash
python scripts/02_train_email_model.py
python scripts/03_train_transcript_model.py
python scripts/01_prepare_audio.py --max-real 300 --max-fake 300
python scripts/04_train_audio_model.py
```

## 3.6.3 Model Justification

The project uses lightweight models because they are:

- Fast enough for local demonstration.
- Explainable to non-technical students.
- Aligned with the educational capstone objective.
- Easier to evaluate within the project timeline.

## 3.7 Phase 6: Detection System and Response

Streamlit modules:

- `app/home_page.py`
- `app/dashboard_page.py`
- `app/simulation_lab_page.py`
- `app/detection_center_page.py`
- `app/report_page.py`
- `app/explainability_page.py`
- `app/history_tab.py`

Response style:

- Streamlit warning banners.
- Confidence charts.
- Keyword highlights.
- Uploaded recording chunk tables.
- Attacker motive explanation.
- Defense steps.
- Session history.

## 3.8 Phase 7: Model Evaluation and Tuning

Evaluate text models with:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Evaluate audio model with:

- Accuracy
- F1-score
- Confusion matrix
- Optional EER/ROC-AUC if you extend evaluation scripts.

## 3.9 Phase 8: Model Deployment

Run locally:

```bash
streamlit run app/main.py
```

For Streamlit Cloud:

```text
app/main.py
```

Keep `requirements.txt` in the repository root.

## 3.9.2 Planned System Architecture

```text
User input
  |-- Simulation scenario decisions
  |-- Uploaded meeting/call recordings
  |-- Email/message text
  |-- Transcript text
  |-- Audio upload
  `-- Manual phone-number check

Preprocessing
  |-- Text cleaning + TF-IDF
  `-- Audio loading + MFCC

Models / logic
  |-- Naive Bayes
  |-- Decision Tree
  |-- SVM
  |-- Uploaded recording chunk analysis
  `-- Scenario decision rules

Streamlit explanation layer
  |-- Confidence
  |-- Highlighted indicators
  |-- Attacker motive
  |-- Defense steps
  |-- Simulation score
  `-- Session history
```
