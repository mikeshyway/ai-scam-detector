# Changes

This document records project changes made after the original proposal so the implementation stays aligned with the capstone objective.

## Main Direction Change

The project now prioritises the **Scam Simulation Lab** as the main student-facing showcase feature.

Reason:

- A detection-only app can show a confidence score, but the proposal also promises cybersecurity education and student empowerment.
- The simulation makes students experience scam pressure, investigate attacker motive with AI support, and practice defense steps.
- The ML models remain important, but they now support a stronger educational workflow.

## Current Main Feature Priority

1. Scam Simulation Lab
2. Email phishing detection
3. Transcript scam detection
4. AI-generated speech upload detection
5. Phone-number risk demo
6. Dashboard, explainability, model comparison, and quiz support pages

## Added

- `app/simulation_lab_page.py`
- `app/detection_center_page.py`
- `app/report_page.py`
- `src/scenarios.py`
- `src/live_audio_simulation.py`
- Turn-based simulation state using `st.session_state`
- Countdown-style phase timer
- Retry-from-checkpoint behavior
- Live-call-style uploaded recording simulation
- 5-10 second audio chunk analysis
- PDF/DOCX/TXT report generator
- Attacker motive explanations
- Defense steps and mini-quiz checks
- Simulation entries in session history
- `data/DATASET_SETUP.md`

## UI Design Change

The interface now uses a cleaner, Notion-like Streamlit layout with:

- Sidebar navigation
- Visible app header
- Soft cards
- Rounded panels
- Light animation
- Better content spacing
- CSS-only shadcn-inspired styling

Direct `shadcn/ui` was not used because it is built for React/Tailwind applications. Streamlit can use custom HTML/CSS and custom components, but importing shadcn directly is outside the normal Streamlit Python workflow. The practical alternative is Streamlit CSS styling inspired by shadcn design.

## Page Structure Remodel

The app now follows this page structure:

1. Home Page
2. Dashboard
3. Scam Simulation Lab
4. Detection Center
5. AI Report Generator
6. Scenario-Based Scam Awareness Quiz
7. Transparency & Explainability Hub
8. Session History

## No Longer Eligible or Intentionally Excluded

### Real-Time Microphone Recording

Excluded because Streamlit microphone support is not reliable across Kali Linux/local deployment without extra components and permissions.

Replacement:

- Audio upload only with `.wav` and `.flac`.

### Automatic Phone Call Recording

Excluded because Streamlit cannot intercept real phone calls, access caller ID, or record mobile calls automatically.

Reason:

- Requires Android/iOS telephony APIs, permissions, consent handling, and legal review.
- Better suited for a separate mobile app or VoIP integration.

Replacement:

- Manual Phone Risk Demo using synthetic reputation data.
- Phone can access the same Streamlit app through LAN/ngrok during a demo, but this is not telecom integration.

### True Live Video Call Interception

Excluded because Streamlit cannot directly intercept Zoom/Teams/Google Meet calls in real time.

Replacement:

- Upload meeting/call recording.
- Split audio into 5-10 second chunks.
- Run MFCC extraction and SVM/demo prediction per chunk.
- Show rolling confidence results to simulate near-real-time detection.

### Suspicious Timestamp Replay in Audio

Excluded because MFCC + SVM classification cannot reliably identify exact suspicious timestamps.

Replacement:

- Full audio playback.
- Waveform.
- Spectrogram.
- File-level classification.

### SMTP Live Email Alerts

Excluded because email-server configuration adds avoidable risk and does not improve the educational objective.

Replacement:

- Streamlit `st.warning()` and `st.error()` banners.

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

- `@st.cache_data` added for demo data, dataset/model status, dashboard summaries, transcript demo loading, and uploaded audio parsing.
- `@st.cache_resource` retained for trained model artifacts.
- Page modules are lazy-loaded in `app/main.py` so heavier imports only run when the selected page needs them.
- Audio parsing is cached by uploaded bytes and suffix.
- Detection tools are grouped in a single Detection Center page to reduce sidebar clutter.
- The old standalone model-comparison page was removed because model comparison now belongs inside the Transparency & Explainability Hub.

## Remaining Limitations

- Countdown display uses browser-side JavaScript for visual timing, while Python checks expiry when the student submits a turn.
- Free-text simulation answers use simple safety keyword heuristics, not a generative AI evaluator.
- Demo datasets are synthetic and must not be presented as official results.
