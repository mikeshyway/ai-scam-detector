# Architecture

## Ownership Rules

| Directory | Responsibility |
| --- | --- |
| `app/` | Streamlit rendering, navigation, session state, and user actions |
| `src/audio/` | Audio decoding, feature extraction, inference, and recording helpers |
| `src/text/` | Text preprocessing, classifier loading, rules, and explainability |
| `src/phone/` | PenipuMY client, fallback lookup, rules, and explanations |
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

The active page graph contains the Detection Center and AI Report Generator.
Detection Center routes internally to email, transcript/audio, and phone
workflows.

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
