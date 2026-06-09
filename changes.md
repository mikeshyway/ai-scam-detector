# Changes

This document records project changes made after the original proposal so the implementation
stays aligned with the current capstone demo scope.

## Current Direction

The project now presents itself as an uploaded-evidence scam detection interface with a
professional GuardAI-style Streamlit UI. The strongest student-facing feature is the Scam
Simulation Lab, which demonstrates how an uploaded meeting or call recording can be split into
short chunks, analyzed, visualized, and explained.

## Current Page Structure

1. Scam Simulation Lab
2. Detection Center
3. AI Report Generator

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
- Browser-microphone educational analysis through WebRTC.
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

Implemented as an optional browser-microphone demonstration using `streamlit-webrtc`.

Implemented scope:

- Analyse the microphone selected by the browser in configurable 3-10 second chunks.
- Extract MFCC and acoustic features for the existing SVM or educational heuristic.
- Optionally run local Whisper speech-to-text, then score the transcript with the existing
  transcript model and suspicious-phrase explanations.
- Save a session summary into the AI Report Generator history.

Limitations:

- Requires browser microphone permission and the additional WebRTC dependencies.
- Does not intercept phone calls, Zoom, Teams, Google Meet, or operating-system audio.
- Capturing another participant requires speaker playback or a separately configured virtual
  audio cable.
- Whisper is optional because its model download and CPU cost are too heavy for the default
  capstone installation.
- Hosted deployments use STUN by default but support static TURN credentials or short-lived
  Twilio Network Traversal Service tokens through Streamlit secrets. TURN is required on many
  cloud, university, VPN, firewall, and carrier-grade NAT networks.

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
