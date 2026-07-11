# Phone Module

## Flow

```text
User phone number
  -> normalize to canonical international format
  -> Omkar Carrier Lookup
  -> local fallback CSV
  -> unknown result
  -> shared normalized record
  -> transparent rules and explanation
  -> Streamlit result/history
```

Omkar Carrier Lookup is the only visible live provider in the Phone Number tab.
Older PenipuMY and IPQualityScore client files may remain in `src/phone/` for
compatibility/history, but they are not part of the active UI flow.

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

## API Key

Configure the key through an environment variable, Streamlit secrets, or the
temporary session input in the Phone Number tab.

```powershell
$env:OMKAR_API_KEY="your-omkar-key"
```

Streamlit secrets are also supported:

```toml
OMKAR_API_KEY = "..."

[omkar]
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

## Local Fallback

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

## Demo Fallback

Path: `data/demo/phone_demo_dataset.csv`

This file contains fictional capstone examples only. The Phone Number tab will
not search it unless Demo Mode is explicitly enabled. Demo results are labelled
as demonstration data and excluded from dashboard headline KPIs.

Fallback order:

```text
Omkar Carrier Lookup
  -> real local processed phone dataset
  -> demo phone dataset only when Demo Mode is enabled
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
Live provider: Omkar Carrier Lookup
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

## Module Responsibilities

- `omkar_client.py`: Omkar Carrier Lookup HTTP communication and response parsing
- `phone_lookup.py`: Omkar -> local -> demo/unknown orchestration
- `phone_rules.py`: transparent evidence-based reputation/context level
- `phone_explainability.py`: readable evidence and recommendations
- `penipumy_client.py`: deprecated provider client kept out of the active UI
- `ipqs_client.py`: deprecated provider client kept out of the active UI
