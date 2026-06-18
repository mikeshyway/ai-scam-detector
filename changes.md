# Changes

This document records project changes made after the original proposal so the implementation
stays aligned with the current capstone demo scope.

## Current Direction

The project now combines uploaded-evidence detection with short local audio recording. The Live
Audio Detection page supports browser voice recording and an in-device short recording mode
that sends each completed WAV clip to Python for transcription, audio features, and scam-risk
analysis.

## Current Page Structure

1. Scam Simulation Lab
2. Live Audio Detection
3. Detection Center
4. AI Report Generator

Removed by request:

- Home
- Dashboard
- Turn-Based Scenario tab
- Transparency Hub
- Session History

## Added Or Retained

- `app/simulation_lab_page.py`
- `app/detection_center_page.py`
- `app/report_page.py`
- `src/recording_audio_simulation.py`
- Uploaded recording chunk analysis
- In-device short audio recording
- 5-10 second audio chunk processing
- PDF/DOCX/TXT report generator
- Malaysia-time report timestamps via `src/time_utils.py`
- `data/DATASET_SETUP.md`

## UI Design Change

The interface was rebuilt with a GuardAI-inspired visual system:

- Dark navy workspace
- Blue and teal security accents
- Text-only sidebar navigation
- Consistent page headers
- Styled cards, tabs, upload fields, metrics, alerts, and tables
- Increased main-content top padding so Streamlit's top toolbar does not cover content

Direct `shadcn/ui` was not used because it is built for React/Tailwind applications. Streamlit
can use custom HTML/CSS and custom components, but importing shadcn directly is outside the
normal Streamlit Python workflow. The practical alternative is Streamlit CSS styling inspired
by modern Tailwind/shadcn layouts.

## Real-Time Feature Decision

### Automatic Email/Text/Call Blocking

Not implemented.

Reason:

- Streamlit does not connect to email inboxes, SMS systems, meeting platforms, or telecom
  systems in this prototype.
- Real pre-delivery blocking would require mail-server integration, mobile OS permissions,
  browser/meeting-platform hooks, telecom or VoIP APIs, consent handling, and legal review.

Replacement:

- Uploaded-evidence detection.
- In-device short audio recording with manual 5-10 second chunks.
- Paste email/text.
- Upload `.txt`, `.csv`, `.wav`, `.flac`, `.mp3`, or `.m4a`.

### Phone Number Integration

Not implemented.

Reason:

- Streamlit does not access telecom identity systems.
- A true caller-ID system requires a separate mobile app or telecom/VoIP integration.

Replacement:

- Manual Phone Number Risk Checker using educational heuristics and synthetic demo reputation
  data.

### In-Device Audio Recording

Implemented scope:

- Record 5-10 second local microphone clips through `audio-recorder-streamlit`.
- Save each recording temporarily as a WAV for local Whisper transcription.
- Run transcript scam analysis and MFCC/SVM voice analysis.
- Display rolling confidence, transcript, frequency, MFCC, suspicious flags, and alert results.
- Let the user record another chunk to simulate near-real-time monitoring.

Limitations:

- This is manual short-chunk recording, not always-on call interception.
- It does not integrate directly with Zoom, Teams, Google Meet, phone calls, or OS-level caller ID.
- It depends on browser/local microphone permission.

## Other Excluded Items

### Near-Real-Time In-Device Recording

Implemented as short local recording chunks. This replaces the earlier continuous streaming
approach while keeping the original Voice Recorder as a separate default feature.

Implemented scope:

- Record local 5-10 second chunks with `audio-recorder-streamlit`.
- Save each chunk as a temporary WAV for Whisper.
- Extract MFCC and acoustic features for the existing SVM or educational heuristic.
- Optionally run local Whisper speech-to-text, then score the transcript with the existing
  transcript model and suspicious-phrase explanations.
- Save a session summary into the AI Report Generator history.

Limitations:

- Requires microphone/browser recording permission.
- Whisper can be heavy on CPU; demo fallback mode remains available.
- This is near-real-time chunk analysis, not zero-latency interception or automatic call
  blocking.

### Suspicious Timestamp Replay In Audio

Excluded because MFCC + SVM classification cannot reliably identify exact suspicious
timestamps.

Replacement:

- Chunk-level uploaded-recording analysis.
- Full audio playback.
- Waveform.
- Spectrogram.

### SMTP Email Alerts

Excluded because email-server configuration adds avoidable risk and does not improve the
educational objective.

Replacement:

- Downloadable TXT/PDF/DOCX reports.
- Streamlit warning and error banners.

### Persistent SQL History

Excluded for the capstone prototype because session-only logging is sufficient for report
generation during the demo.

Replacement:

- `st.session_state` temporary history used internally by the report generator.

### Full ASVspoof 2019 Dataset

Excluded because the full dataset is too large for practical capstone training and deployment.

Replacement:

- Subsample around 300 real and 300 fake files.

### YouTube Scam Transcript Dataset As Binary Training Data

Excluded for binary classifier training because it contains scam-only examples.

Replacement:

- Use it for educational transcript examples only.
- Use scam/non-scam call conversation data for binary training.

## Performance Changes

- `@st.cache_data` added for demo data, dataset/model status, transcript demo loading,
  uploaded audio parsing, and chunk analysis.
- `@st.cache_resource` retained for trained model artifacts.
- Page modules are lazy-loaded in `app/main.py`.
- Detection tools are grouped in a single Detection Center page to reduce sidebar clutter.

## Remaining Limitations

- Demo datasets are synthetic and must not be presented as official results.
- Uploaded-recording analysis is an educational replay, not a forensic or real-time call
  interception system.
