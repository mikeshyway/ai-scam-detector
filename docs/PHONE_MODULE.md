# Phone Module

## Flow

```text
User phone number
  -> normalize to canonical international format
  -> configured live provider checks
  -> Omkar metadata evidence
  -> PenipuMY reputation/report evidence where configured
  -> local/demo fallback where allowed
  -> unknown result when evidence is unavailable
  -> shared normalized record
  -> transparent rules and explanation
  -> Streamlit result/history
```

The final prototype evidence notebook treats phone checks as provider evidence,
not local ML training. Omkar supplies carrier/line metadata. PenipuMY supplies
reputation or report-oriented fields where a valid key is configured. Older
IPQualityScore client files may remain in `src/phone/` for compatibility or
history, but they are not the primary documented final flow.

## Phone Number Normalization

The UI accepts common local and international formats, then converts them into
one canonical E.164-style internal format before lookup.

Accepted examples:

```text
012-345 6789
0123456789
+60 12-345 6789
60123456789
(03) 1234 5678
```

Canonical internal examples:

```text
+60123456789
+60312345678
```

Omkar receives the canonical E.164-style value, such as `+60123456789`. The app
rejects clearly invalid text, repeated plus signs, and alphabetic input.

## Live Provider

### Omkar Carrier Lookup

Omkar is used as a carrier and number metadata provider. It can return
validity, carrier, line type, E.164 phone number, national formatting, country
code, calling country code, mobile country code, and mobile network code.

Documentation: <https://github.com/omkarcloud/phone-lookup-api>

Account verification: <https://www.omkar.cloud/account/verify-phone>

Carrier metadata is not scam reputation. A valid phone number does not prove a
caller is safe, and a VoIP/mobile/landline classification does not prove fraud
by itself.

### PenipuMY

PenipuMY is used as a reputation/report evidence provider when configured. Its
fields can support the phone concern explanation, but provider reports are still
evidence, not a trained local caller-risk model and not legal proof.

## API Key

Configure the key through an environment variable, Streamlit secrets, or the
temporary session input in the Phone Number tab.

```powershell
$env:OMKAR_API_KEY="your-omkar-key"
$env:PENIPUMY_API_KEY="your-penipumy-key"
```

Streamlit secrets are also supported:

```toml
OMKAR_API_KEY = "..."
PENIPUMY_API_KEY = "..."

[omkar]
api_key = "..."

[penipumy]
api_key = "..."
```

Never commit `.env`, `.streamlit/secrets.toml`, or real API keys. The repository
includes `.env.example` with a blank Omkar placeholder only.

## Setup Guide

A standalone Omkar setup guide is available at:

```text
docs/omkar_api_setup_guide.html
```

The Phone Number tab offers this file as a download so the main page stays
concise.

## Account Verification Handling

If Omkar returns a message asking you to verify your phone number, the provider
is reachable and the key may be accepted, but the Omkar account has not enabled
free-plan carrier lookups yet. Complete verification at:

```text
https://www.omkar.cloud/account/verify-phone
```

The app labels this as `account_phone_verification_required`, not invalid phone
format.

## Local Fallback And Demo Evidence

Path: `data/processed/phone/phone_dataset.csv`

This file is for real, traceable fallback records only. Do not place synthetic
demo records in this file. If Omkar does not succeed and no real local row
matches, the correct result is Unknown.

Required columns:

```text
phone
police_report_count
verified_report_count
spam
fraud
business_tier
business_name
spoofing_report_count
source
record_type
is_demo
source_reference
last_verified
```

Rows in this file should use `record_type=real` and `is_demo=false`.

Path: `data/demo/phone_demo_dataset.csv`

This file contains fictional capstone examples only. The Phone Number tab will
not search it unless Demo Mode is explicitly enabled. Demo results are labelled
as demonstration data and excluded from dashboard headline KPIs.

Path: `data/demo/demo_phone_numbers.csv`

This smaller demo file may be used by notebook/demo workflows as repeatable
presentation input. It should not be described as trained phone-risk data.

Fallback order:

```text
Configured live providers
  -> real local processed phone dataset where allowed
  -> demo phone dataset only when demo flow is enabled
  -> unknown result
```

## Output Principles

- `Valid` means number format/routing appears valid.
- `Metadata available` means carrier or line information was returned.
- `Unknown` means no reputation conclusion is available.
- `High Risk` appears only when real reputation evidence or explicit fallback
  records support it.

The UI shows provenance for each result:

```text
Live provider: Omkar Carrier Lookup / PenipuMY where configured
Fallback used: Yes/No
Provider returned: Carrier or validation metadata / No usable carrier metadata
Scam reputation available: Yes/No
```

## Charts

The Phone Number tab may show:

- Lookup Evidence Coverage
- Caller Claim Consistency
- Provider Response Completeness
- Session Lookup History after multiple phone lookups

These charts summarize available evidence. They are not ML probabilities and do
not change the final lookup result.

## No Additional Phone ML Model

The phone module intentionally remains:

```text
Omkar API
+ PenipuMY API where configured
+ normalization
+ local fallback
+ transparent consistency rules
+ explainability
```

Do not add a phone-specific machine-learning model unless a sufficiently large,
labelled, traceable phone-metadata dataset becomes available. Adding a model
without that dataset would create weak or misleading evidence.

## Unknown Result

If neither Omkar nor the local dataset contains the number, the module returns an
Unknown result. Unknown does not mean safe. The UI should continue to advise
verification and never sharing OTPs, passwords, banking details, or personal
information.

## History Database

Phone rows saved in `data/session_history.db` are report evidence rows. They can
contain masked phone numbers, provider status, provider-derived fields, concern
labels, concern scores, and recommended verification actions.

This database is not a locally trained phone-risk database. Stored phone
outcomes come from provider-pulled evidence plus deterministic rules at lookup
time. A future lookup can differ if provider data, API keys, quota, rate limits,
or network availability change.

## Module Responsibilities

- `omkar_client.py`: Omkar Carrier Lookup HTTP communication and response parsing
- `penipumy_client.py`: PenipuMY HTTP communication and response parsing
- `phone_lookup.py`: provider -> local -> demo/unknown orchestration
- `phone_rules.py`: transparent evidence-based reputation/context level
- `phone_explainability.py`: readable evidence and recommendations
- `ipqs_client.py`: deprecated provider client kept out of the active UI
