# Model Training

Run commands from the repository root with the same Python environment used by
Streamlit.

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

## Transcript Models

Input: `data/processed/transcript/transcript_dataset.csv`

Outputs use the `transcript_*.pkl` naming convention under `models/`.

Metrics: `reports/metrics/transcript_model_metrics.json`

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
