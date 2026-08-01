# Curated Demo Evidence

This folder contains non-personal demo evidence for repeatable dashboard,
notebook, SQL history, and lecturer walkthroughs. These files are presentation
fixtures, not claims about real victims, real callers, or live provider results.

The dataset and processing proof remains in `data/raw/` and `data/processed/`.
Keep those folders when demonstrating that the project used source datasets and
processed them into deployable artifacts.

## Upload-Ready Files

- `demo_emails.csv`: email examples with expected dashboard outcomes.
- `demo_transcripts.csv`: voice transcript examples with expected outcomes.
- `demo_phone_numbers.csv`: reserved phone examples split into scam and innocent
  groups.
- `phone_demo_dataset.csv`: local phone reputation fixture used by the phone
  lookup workflow.
- `uploads/`: individual `.txt` files that can be uploaded directly in the
  Email and Voice Transcript tabs.

## Scenario Labels

- `email_innocent_*` and `transcript_innocent_*`: ordinary communications that
  should land near No Immediate Action.
- `email_malicious_*` and `transcript_malicious_*`: clear scam or phishing
  evidence that should land near Immediate Action Required.
- `email_review_*` and `transcript_review_*`: middle-ground evidence where the
  safest result is Review Required.

## SQL History Note

CSV batch analysis is useful for demonstrating multiple rows, but the dashboard
creates the richest report artifacts when a single upload is analyzed through
the Email or Voice Transcript tab. Use the individual files in `uploads/` when
you want SQL history rows with model-comparison charts and report evidence.

## Phone Number Privacy

Phone demo rows use reserved drama/demo numbers instead of personal Malaysian
numbers. They are for repeatable report evidence only and should not be
described as live Veriphone.io or PenipuMY results.

Marker used in generated demo data:

```text
CURATED_DEMO_EVIDENCE_RESERVED_OR_FICTIONAL
```
