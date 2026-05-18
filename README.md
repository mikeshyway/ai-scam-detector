# AI Scam Detector

Educational AI scam detection platform for Capstone Project 2. The system runs locally with
Python and Streamlit and demonstrates how machine learning can flag suspicious emails,
scam-style transcripts, and AI-generated speech.

This project is intentionally scoped as a learning and awareness tool, not as a commercial
cybersecurity product. It focuses on explainability, confidence scores, visual dashboards,
and clear educational feedback for students.

## Features

- Email phishing detection using TF-IDF with Naive Bayes and Decision Tree models.
- Scam transcript detection for call, Zoom, Teams, and Google Meet transcript text.
- AI-generated speech detection using MFCC audio features with an SVM classifier.
- Confidence scoring and warning banners for suspicious predictions.
- Suspicious phrase highlighting to explain why text was flagged.
- Audio playback, waveform visualization, and spectrogram visualization.
- Session-only detection history using `st.session_state`.
- Upload support for `.txt`, `.csv`, `.wav`, and `.flac` files.

## Project Structure

```text
ai-scam-detector/
├── app/                 Streamlit web application
├── data/                Raw and processed datasets
├── models/              Trained model artifacts
├── notebooks/           Jupyter walkthrough notebooks
├── scripts/             Dataset preparation and training scripts
└── src/                 Shared preprocessing, model, and explainability code
```

## Kali Linux Setup

```bash
cd ai-scam-detector
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/00_setup_check.py
```

If NLTK resources are missing, the code automatically falls back to scikit-learn stopwords.
For the best preprocessing quality, you can optionally run:

```bash
python -m nltk.downloader stopwords wordnet omw-1.4
```

## Dataset Placement

Place datasets in the following folders before training:

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

For the transcript CSV, the training script will try to infer sensible text and label columns.
If your Kaggle file uses unusual column names, pass them explicitly:

```bash
python scripts/03_train_transcript_model.py --text-column conversation --label-column label
```

## Training

Run the scripts in order:

```bash
python scripts/02_train_email_model.py
python scripts/03_train_transcript_model.py
python scripts/01_prepare_audio.py --max-real 300 --max-fake 300
python scripts/04_train_audio_model.py
```

Generated artifacts are saved in `models/` and `data/processed/`.

## Run the Streamlit App

```bash
streamlit run app/main.py
```

The app can open before models are trained. Text tabs will run in educational demo mode using
suspicious phrase rules, while the audio tab will show playback, waveform, and spectrogram
visuals until `models/audio_svm.pkl` exists.

## Streamlit Cloud Deployment

If this folder is the GitHub repository root, set the Streamlit app path to:

```text
app/main.py
```

Keep `requirements.txt` at the repository root so Streamlit Cloud installs packages such as
`plotly`, `matplotlib`, `scikit-learn`, and `librosa`.

## Capstone Scope Notes

The updated version uses:

- `teeconnie/scam-and-non-scam-call-conversation-dataset` for binary transcript training.
- YouTube scam transcripts for educational examples only.
- A small ASVspoof 2019 LA subset, around 300 real and 300 fake files.
- Audio upload only, avoiding unreliable real-time microphone support on Kali Linux.
- Streamlit warnings instead of SMTP alerts.
- Temporary Streamlit session history instead of a persistent SQL database.
- Pre-extracted MFCC arrays so audio training and demos remain lightweight.

## Ethical Use

This project is for education and awareness. Predictions may be wrong and should not be used
as the sole basis for security, legal, or financial decisions.
