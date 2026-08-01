# Curated Demo Evidence

This folder contains non-personal, reserved demo evidence for repeatable
dashboard, notebook, SQL history, and lecturer walkthroughs. These files are
presentation fixtures with documented expected outcomes.

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

Current upload-ready coverage:

| Category | Email Files | Transcript Files | Expected Dashboard Action |
| --- | ---: | ---: | --- |
| Innocent | 3 | 3 | No Immediate Action |
| Malicious | 2 | 3 | Immediate Action Required |
| Review | 3 | 2 | Review Required |

## Scenario Labels

- `email_innocent_*` and `transcript_innocent_*`: ordinary communications that
  should land near No Immediate Action.
- `email_malicious_*` and `transcript_malicious_*`: clear scam or phishing
  evidence that should land near Immediate Action Required.
- `email_review_*` and `transcript_review_*`: middle-ground evidence where the
  safest result is Review Required.

## SQL History Note

CSV batch analysis demonstrates multiple records. Individual upload files in
`uploads/` create the richest dashboard artifacts for SQL history, model
comparison charts, highlighted evidence, and generated Student Brief reports.

## Phone Number Privacy

Phone demo rows use reserved drama/demo numbers in place of personal Malaysian
numbers. They provide repeatable report evidence for the phone workflow and are
labelled as curated demo records.

Marker used in generated demo data:

```text
CURATED_DEMO_EVIDENCE_RESERVED_OR_FICTIONAL
```
