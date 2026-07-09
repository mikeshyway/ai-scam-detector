# Changes

## 2026-07-09 - Option A source and script naming

- Renamed reusable preprocessing modules to `*_preprocessor.py`.
- Renamed reusable training modules to `*_trainer.py`.
- Renumbered the public workflow scripts from `00` through `07`.
- Archived superseded numbered compatibility scripts under
  `archive/deprecated_scripts/`.
- Kept ML behavior, metrics, model artifacts, datasets, reports, and UI logic
  unchanged.

## 2026-07-09 - Architecture audit

- Added stable root and script entry points while keeping `app/main.py` as the
  canonical Streamlit Cloud path.
- Converted stale numbered preprocessing/training scripts into compatibility
  wrappers around canonical `src` workflows.
- Added package markers for `src/preprocessing`, `src/training`, and `scripts`.
- Made transcript dataset preparation import-safe by moving execution into
  `prepare_transcript_dataset()` and `main()`.
- Archived unreferenced former page modules under `archive/deprecated/app/`.
- Updated tests, setup checks, dataset status paths, README, and architecture
  documentation to match the current project.

This document records project changes made after the original proposal so the implementation
stays aligned with the current capstone demo scope.

## Current Direction

The project now combines uploaded-evidence detection with short local audio workflows. The Live
Audio Detection page supports browser voice recording and a separate internal system-audio
capture mode that sends each completed WAV clip to Python for transcription, audio features,
and scam-risk analysis.

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
- Internal system-audio capture
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
- Internal system-audio capture with manual 5-10 second chunks.
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

### Internal Device Audio Capture

Implemented scope:

- Capture 5-10 second chunks from local system audio using `sounddevice`.
- Support Windows WASAPI output candidates, macOS BlackHole-style virtual devices, and
  Linux PulseAudio/PipeWire monitor sources where available.
- Block physical microphone inputs for this feature.
- Save each captured chunk temporarily as a WAV for local Whisper transcription.
- Provide WAV upload fallback when internal capture is unavailable.
- Run transcript scam analysis and MFCC/SVM voice analysis.
- Display rolling confidence, transcript, frequency, MFCC, suspicious flags, and alert results.
- Let the user capture another chunk to simulate near-real-time monitoring.

Limitations:

- This is manual short-chunk recording, not always-on call interception.
- It does not integrate directly with Zoom, Teams, Google Meet, phone calls, or OS-level caller ID.
- It must run locally; Streamlit Cloud cannot access the user's system speaker output.

## Other Excluded Items

### Near-Real-Time Internal Audio Capture

Implemented as short local internal-audio chunks. This replaces the earlier browser/microphone
recorder approach for Device Audio Monitor while keeping the original Voice Recorder as a
separate default feature.

Implemented scope:

- Capture local 5-10 second internal audio chunks with `sounddevice`.
- Reject likely microphone inputs.
- Run automatic dependency and audio-device diagnostics when the page first opens.
- Provide a refreshable setup panel with exact package/FFmpeg installation commands.
- Fall back to Python's built-in PCM WAV writer when `soundfile` is unavailable.
- Keep WAV upload, audio-only analysis, and manual/demo transcript paths available when local
  capture dependencies are missing.
- Test the selected source for three seconds and report duration, RMS, peak level, silence,
  playback, and pass/warning/error status.
- Probe multiple sample-rate, channel, and latency combinations for supported Windows Stereo Mix
  sources while letting PortAudio choose a compatible stream buffer size.
- Display device ID, host API, sample rate, channels, WASAPI/input readiness, and preserve exact
  PortAudio failures in `logs/system_diagnostics.log`.
- Detect CABLE/VB-Cable, VoiceMeeter, and BlackHole endpoints as Meeting Capture Devices, add a
  Recommended badge, and prioritize input-capable virtual devices for transcription/scam analysis.
- Exclude Windows WDM-KS devices from blocking capture after PortAudio reported error `-9999`,
  keep them visible as unsupported diagnostics rows, and prefer WASAPI, DirectSound, then MME.
- Append non-fatal diagnostic snapshots to `logs/system_diagnostics.log`.
- Save each chunk as a temporary WAV for Whisper.
- Extract MFCC and acoustic features for the existing SVM or educational heuristic.
- Optionally run local Whisper speech-to-text, then score the transcript with the existing
  transcript model and suspicious-phrase explanations.
- Use per-tab recording carousels so Voice Recorder and Device Audio Monitor clips can be
  reviewed separately.
- Use `live_monitor_generation` like Voice Recorder's `live_recorder_generation`: capturing
  another sample resets only the active input/playback widget while preserving analysed clips.
- Continue audio-only analysis if Whisper cannot transcribe because `ffmpeg` is missing.
- Save a session summary into the AI Report Generator history.

Limitations:

- Requires a local internal audio source such as WASAPI output, BlackHole, Stereo Mix, a
  monitor source, or a virtual cable.
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
