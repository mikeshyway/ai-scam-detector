# Model Training

Run commands from the repository root with the same Python environment used by
Streamlit.

The final evidence notebook,
`notebooks/00_final_prototype_evidence_notebook.ipynb`, reads saved artifacts
and metrics instead of retraining models. Use this file for reviewer evidence,
and use this document for the training command boundary and model-selection
logic.

## Recommended Order

```powershell
py scripts\01_prepare_email_dataset.py
py scripts\04_train_email_model.py

py scripts\02_prepare_transcript_dataset.py
py scripts\05_train_transcript_model.py

py scripts\03_prepare_audio_dataset.py
py scripts\06_train_audio_model.py
py scripts\07_train_audio_behavior_model.py
```

## Email Models

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

## Transcript Models

Input: `data/processed/transcript/transcript_dataset.csv`

Outputs use the `transcript_*.pkl` naming convention under `models/`.

Metrics: `reports/metrics/transcript_model_metrics.json`

Training principle:

- Transcript text is prepared separately from email because call language,
  urgency, threats, and OTP/payment requests have a different pattern.
- TF-IDF NB/SVM models are supported, and transformer models such as
  DistilBERT can be included when the required checkpoints and compute are
  available.
- The dashboard loads available saved artifacts and should explain when a
  preferred model is unavailable.

## Audio Models

Input: `data/processed/audio/labels.csv` plus the `train/` and `dev/` audio
folders.

- `scripts/06_train_audio_model.py`: MFCC + calibrated SVM
- `scripts/07_train_audio_behavior_model.py`: behavioral features + Random Forest

Outputs:

```text
models/audio_svm.pkl
models/audio_behavior_rf.pkl
reports/metrics/audio_model_metrics.json
reports/metrics/audio_behavior_metrics.json
```

## Validation

After training:

```powershell
py -m compileall app src scripts tests
py -m unittest discover -s tests
```

Restart the Streamlit process after replacing model artifacts so cached
resources are reloaded.

## Split And Selection Principles

| Channel | Split Strategy | Selection Evidence | Runtime Benefit |
| --- | --- | --- | --- |
| Email | Stratified 80/20 train/test split | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix where saved | Provides a stronger default classifier for email/message risk evidence. |
| Transcript | Stratified 80/20 train/test split | Same text-classification metrics, plus transformer metrics when trained | Helps score scam intent in call transcripts or transcribed speech. |
| Audio | ASVspoof train/dev style split | Audio metric JSON, confusion matrix, ROC where saved, behavior feature importance | Produces voice-authenticity concern signals for uploaded/recorded audio. |
| Phone | No local training split | Provider response fields and deterministic evidence weights | Keeps caller evidence transparent instead of inventing an ML probability. |

The "best" or "recommended" model means the strongest candidate among the
available saved validation results. It does not mean the model is perfect or
universally accurate.

## Expected Terminal Evidence

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

The Streamlit app should then load the saved artifacts from `models/` and saved
metric summaries from `reports/metrics/`; it should not retrain during normal
demo use.
