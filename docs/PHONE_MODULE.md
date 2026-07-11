# Phone Module

## Flow

```text
User phone number
  -> user-selected live provider
       -> PenipuMY
       -> IPQualityScore
       -> Carrier Lookup
  -> local fallback CSV
  -> unknown result
  -> shared normalized record
  -> shared rules and explanation
  -> Streamlit result/history
```

The app calls only the provider selected by the user. It does not automatically
call the other live providers, because PenipuMY, IPQualityScore, and Carrier
Lookup return different evidence types and use separate API quotas.

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

Provider-specific conversion happens after this step:

- PenipuMY receives digits only, such as `60123456789`
- IPQualityScore receives canonical E.164-style text, such as `+60123456789`
- Carrier Lookup receives canonical E.164-style text, such as `+60123456789`

The app rejects clearly invalid text, repeated plus signs, and alphabetic input.

## Live Providers

### PenipuMY

PenipuMY is used as a Malaysian scam-report and caller-reputation source.
It can return police report counts, verified community reports, spam/fraud
flags, business-directory information, and spoofing-related business context.

Documentation: <https://penipu.my/api/v1/docs>

### IPQualityScore

IPQualityScore is used as a phone validation and fraud-risk metadata source.
It can return validity, active status, fraud score, recent abuse, risky/spammer
flags, carrier, line type, country, region, city, VoIP, prepaid, leaked, and
Do Not Call metadata.

Documentation: <https://www.ipqualityscore.com/documentation/phone-number-validation-api/overview>

IPQS metadata is not treated as PenipuMY report evidence. For example, an IPQS
fraud score is preserved as `fraud_score`; it is not converted into fake police
or verified report counts.

### Carrier Lookup

Carrier Lookup is used as a carrier and number metadata provider. It can return
validity, carrier, line type, E.164 phone number, national formatting, country
code, calling country code, mobile country code, and mobile network code.

Documentation: <https://github.com/omkarcloud/phone-lookup-api>

Carrier Lookup metadata is not treated as scam reputation. For example, a VoIP
line type is preserved as context; it is not converted into fake police reports,
fraud flags, or IPQS-style fraud scores.

## API Keys

Configure keys through environment variables, Streamlit secrets, or the
temporary session input in the Phone Number tab.

```powershell
$env:PENIPUMY_API_KEY="your-penipumy-key"
$env:IPQS_API_KEY="your-ipqs-key"
$env:OMKAR_API_KEY="your-omkar-key"
```

`PENIPUMY_API_KEY` is the standard name. The app still checks the older
`PENIPU_API_KEY` name as a temporary backward-compatible fallback.

Streamlit secrets are also supported:

```toml
PENIPUMY_API_KEY = "..."
IPQS_API_KEY = "..."
OMKAR_API_KEY = "..."

[penipumy]
api_key = "..."

[ipqs]
api_key = "..."

[omkar]
api_key = "..."
```

Never commit `.env`, `.streamlit/secrets.toml`, or real API keys. The
repository includes `.env.example` with blank placeholders only.

## Setup Guide

A standalone HTML setup guide is available at:

```text
docs/phone_api_setup_guide.html
```

The Phone Number tab offers this file as a download so the main page stays
concise.

## Local Fallback

Path: `data/processed/phone/phone_dataset.csv`

This file is for real, traceable fallback records only. Do not place synthetic
demo records in this file. If no live provider succeeds and no real local row
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
Selected live provider
  -> real local processed phone dataset
  -> demo phone dataset only when Demo Mode is enabled
  -> unknown result
```

## Omkar Account Verification

If Carrier Lookup returns a message asking you to verify your phone number, the
provider is reachable and the key may be accepted, but the Omkar account has not
enabled free-plan carrier lookups yet. Complete verification at:

```text
https://www.omkar.cloud/account/verify-phone
```

The app labels this as account verification required, not invalid phone format.

## Unknown Result

If neither the selected provider nor the local dataset contains the number, the
module returns an unknown result. Unknown does not mean safe. The UI should
continue to advise verification and never sharing OTPs, passwords, banking
details, or personal information.

## Module Responsibilities

- `penipumy_client.py`: PenipuMY HTTP communication and response parsing
- `ipqs_client.py`: IPQualityScore HTTP communication and response parsing
- `omkar_client.py`: Omkar Carrier Lookup HTTP communication and response parsing
- `phone_lookup.py`: selected-provider -> local -> unknown orchestration
- `phone_rules.py`: evidence-based reputation level
- `phone_explainability.py`: readable evidence and recommendations
