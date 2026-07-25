# Architecture

The final prototype architecture is summarized and verified in
`notebooks/00_final_prototype_evidence_notebook.ipynb`. This file keeps the
codebase ownership and execution-flow view, while the notebook remains the
main evidence record for reviewers.

## Ownership Rules

| Directory | Responsibility |
| --- | --- |
| `app/` | Streamlit rendering, navigation, session state, and user actions |
| `src/audio/` | Audio decoding, feature extraction, inference, and recording helpers |
| `src/text/` | Text preprocessing, classifier loading, rules, and explainability |
| `src/phone/` | Veriphone.io carrier lookup, PenipuMY reputation lookup, rules, and explanations |
| `src/reporting/` | Saved history and TXT/PDF/DOCX report generation |
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

## Application Flow

```text
main.py or app/main.py
  -> app/main.py
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

## Runtime Evidence Flow

```text
User evidence
  -> Streamlit tab in app/
  -> src/ domain logic where applicable
  -> saved model artifact, provider evidence, or deterministic rules
  -> result explanation and history row
  -> report generator preview/export
```

Email and transcript tabs load saved text model artifacts from `models/`.
Audio analysis uses audio feature/inference helpers and saved audio artifacts
where available. Phone checks use provider evidence from Veriphone.io and PenipuMY
where configured, plus transparent rules; they are not a locally trained phone
ML model.

## Python Packages

Every importable source directory contains `__init__.py`. Runtime imports use
fully qualified paths such as:

```python
from src.audio.live_audio_analysis import analyse_live_chunk
from src.text.explainability import find_suspicious_phrases
from src.phone.phone_lookup import lookup_phone
```

`__pycache__/`, `.pyc`, Numba caches, virtual environments, logs, secrets, and
temporary files are generated locally and excluded from Git.

## Archived Files

`archive/deprecated/` contains former pages that are not reachable from the
active route graph. They are retained only for implementation history. New
code must not import archived modules.

## Stable Artifact Layout

Model files remain flat under `models/` because current runtime loaders use
stable artifact names such as `email_nb.pkl`, `transcript_svm.pkl`, and
`audio_svm.pkl`. Moving them into channel subdirectories would create a broad
and unnecessary migration risk.

Metrics are already grouped under `reports/metrics/`. Raw and processed
datasets are grouped by channel under `data/`.

Notebook evidence and EDA appendices remain under `notebooks/`. The final
evidence notebook should be treated as the main documentation reference, while
the channel-specific notebooks provide focused appendices.
