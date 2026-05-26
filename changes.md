# Changes

This document records project changes made after the original proposal so the implementation stays aligned with the capstone objective.

## Current Direction

The project now presents itself as an **uploaded-evidence scam awareness and detection lab**.

The strongest student-facing feature is the Scam Simulation Lab. It demonstrates how an uploaded meeting/call recording can be split into short chunks, analyzed, and explained.

## Current Page Structure

1. Home
2. Dashboard
3. Scam Simulation Lab
4. Detection Center
5. AI Report Generator
6. Transparency Hub
7. Session History

The old standalone quiz page was removed. Scenario questions now reside inside the turn-based section of the Scam Simulation Lab.

## Added

- `app/simulation_lab_page.py`
- `app/detection_center_page.py`
- `app/report_page.py`
- `src/scenarios.py`
- `src/recording_audio_simulation.py`
- Turn-based simulation state using `st.session_state`
- Start-session gate before scenario content appears
- Countdown-style phase timer
- Retry-from-checkpoint behavior
- Uploaded recording chunk analysis
- 5-10 second audio chunk processing
- PDF/DOCX/TXT report generator
- Malaysia-time report timestamps via `src/time_utils.py`
- Attacker motive explanations
- Defense steps and mini-quiz checks inside the simulation
- Simulation entries in session history
- `data/DATASET_SETUP.md`

## UI Design Change

The interface was rebuilt with a cleaner Streamlit layout:

- Text-only sidebar navigation buttons
- No radio-button page menu
- No emoji-heavy page labels
- System status moved into a collapsible sidebar section
- Tailwind-inspired spacing, borders, cards, and gradients through custom CSS
- Increased main-content top padding so Streamlit's top toolbar does not cover content

Direct `shadcn/ui` was not used because it is built for React/Tailwind applications. Streamlit can use custom HTML/CSS and custom components, but importing shadcn directly is outside the normal Streamlit Python workflow. The practical alternative is Streamlit CSS styling inspired by modern Tailwind/shadcn layouts.

## Real-Time Feature Decision

### Automatic Email/Text/Call Blocking

Not implemented.

Reason:

- Streamlit does not connect to email inboxes, SMS systems, meeting platforms, or telecom systems in this prototype.
- Real pre-delivery blocking would require mail-server integration, mobile OS permissions, browser/meeting-platform hooks, telecom or VoIP APIs, consent handling, and legal review.

Replacement:

- Uploaded-evidence detection only.
- Paste email/text.
- Upload `.txt`, `.csv`, `.wav`, `.flac`, `.mp3`, or `.m4a`.

### Phone Number Integration

Not implemented.

Reason:

- Streamlit does not access telecom identity systems.
- A true caller-ID system requires a separate mobile app or telecom/VoIP integration.

Replacement:

- Manual Phone Number Risk Checker using educational heuristics and synthetic demo reputation data.

### PC Meeting Platform Integration

Not implemented.

Reason:

- Streamlit does not connect directly to Zoom, Teams, or Google Meet audio streams.

Replacement:

- Upload exported meeting/call recording.
- Split audio into 5-10 second chunks.
- Run MFCC extraction and SVM/demo prediction per chunk.
- Show chunk confidence results as an educational replay-style analysis.

## Other Excluded Items

### Real-Time Microphone Recording

Excluded because Streamlit microphone support is not reliable across Kali Linux/local deployment without extra components and permissions.

Replacement:

- Audio upload only.

### Suspicious Timestamp Replay in Audio

Excluded because MFCC + SVM classification cannot reliably identify exact suspicious timestamps.

Replacement:

- Chunk-level uploaded-recording analysis.
- Full audio playback.
- Waveform.
- Spectrogram.

### SMTP Email Alerts

Excluded because email-server configuration adds avoidable risk and does not improve the educational objective.

Replacement:

- Downloadable TXT/PDF/DOCX reports.
- Streamlit warning and error banners.

### Persistent SQL History

Excluded for the capstone prototype because session-only learning logs are sufficient for demonstration.

Replacement:

- `st.session_state` temporary history.

### Full ASVspoof 2019 Dataset

Excluded because the full dataset is too large for practical capstone training and deployment.

Replacement:

- Subsample around 300 real and 300 fake files.

### YouTube Scam Transcript Dataset as Binary Training Data

Excluded for binary classifier training because it contains scam-only examples.

Replacement:

- Use it for educational examples and scenario content only.
- Use scam/non-scam call conversation data for binary training.

## Performance Changes

- `@st.cache_data` added for demo data, dataset/model status, dashboard summaries, transcript demo loading, uploaded audio parsing, and chunk analysis.
- `@st.cache_resource` retained for trained model artifacts.
- Page modules are lazy-loaded in `app/main.py`.
- Detection tools are grouped in a single Detection Center page to reduce sidebar clutter.
- The old standalone model-comparison page was removed because model comparison now belongs inside the Transparency Hub.

## Remaining Limitations

- Countdown display uses browser-side JavaScript for visual timing, while Python checks expiry when the student submits a turn.
- Free-text simulation answers use simple safety keyword heuristics, not a generative AI evaluator.
- Demo datasets are synthetic and must not be presented as official results.
